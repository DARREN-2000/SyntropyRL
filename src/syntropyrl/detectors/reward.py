"""Reward family: the policy is optimizing something you did not intend.

This is the family people feel worst about, because the run looks like a
success. Reward goes up and to the right. The model gets worse.
"""

from __future__ import annotations

from typing import Optional

from .. import stats
from ..core import Detector, Diagnosis, RunHistory, Severity


class RewardHacking(Detector):
    """RLD-031 - reward rises while the thing you actually care about does not.

    The load-bearing idea: reward is a proxy. If you log even one ground-truth
    metric next to it, divergence between the two is measurable and damning.
    """

    code = "RLD-031"
    title = "Reward hacking (proxy/truth divergence)"
    family = "reward"
    min_steps = 25
    cooldown = 60
    requires = ("reward_mean",)

    def check(self, h: RunHistory) -> Optional[Diagnosis]:
        rewards = h.series("reward_mean")
        if len(rewards) < 25:
            return None
        r_first, r_second = stats.split_halves(rewards)
        r_change = stats.relative_change(stats.mean(r_first), stats.mean(r_second))
        if r_change < 0.12:
            return None

        # Supporting signal 1: a ground-truth metric that refuses to move.
        truth_key = next(
            (k for k in ("accuracy", "task_accuracy", "eval_accuracy", "pass_rate") if h.has(k, 20)),
            None,
        )
        truth_change = None
        if truth_key:
            t = h.series(truth_key)
            t_first, t_second = stats.split_halves(t)
            truth_change = stats.relative_change(stats.mean(t_first), stats.mean(t_second))

        # Supporting signal 2: a phrase whose presence *pays*.
        # Not "a phrase common in good answers" -- that fires on stock domain
        # vocabulary. The exploit signature is that rollouts containing the
        # phrase earn materially more reward than rollouts that do not.
        recent = h.recent_rollouts(4)
        rewards_r = [float(r.reward) for r in recent]
        best = stats.reward_correlated_ngram(
            [r.completion for r in recent], rewards_r, n=4
        )
        spread = (max(rewards_r) - min(rewards_r)) if rewards_r else 0.0
        # t >= 6 is deliberately conservative: we search ~40 candidate phrases,
        # so a nominal 95% threshold would fire on healthy runs constantly.
        phrase_exploit = bool(
            best
            and spread > 0
            and best["t"] >= 6.0
            and best["delta"] >= max(0.15, 0.25 * spread)
        )

        diverging = truth_change is not None and truth_change < 0.04
        if not diverging and not phrase_exploit:
            return None

        parts = [
            "Reward is up {} across this window".format(stats.pct(r_change, 0)),
        ]
        if diverging:
            parts.append(
                "while {} moved only {}".format(truth_key, stats.pct(truth_change or 0.0, 1))
            )
        if phrase_exploit and best:
            parts.append(
                'rollouts containing "{}" score {} versus {} without it, and {} of '
                "the batch now contains it".format(
                    best["gram"],
                    stats.fmt(best["with"], 2),
                    stats.fmt(best["without"], 2),
                    stats.pct(best["prevalence"], 0),
                )
            )
        severity = Severity.CRITICAL if (diverging and phrase_exploit) or diverging else Severity.WARN
        return self._dx(
            h,
            severity,
            ". ".join(parts) + ". The policy is optimizing the proxy, not the task.",
            evidence={
                "reward_change": stats.pct(r_change, 1),
                "truth_metric": truth_key,
                "truth_change": None if truth_change is None else stats.pct(truth_change, 1),
                "paying_ngram": None if not best else best["gram"],
                "reward_with_ngram": None if not best else round(best["with"], 3),
                "reward_without_ngram": None if not best else round(best["without"], 3),
                "ngram_prevalence": None if not best else round(best["prevalence"], 3),
                "ngram_effect_t": None if not best else round(best["t"], 1),
            },
            causes=[
                "Verifier matches a prefix or a format marker instead of the answer",
                "Regex or string-equality grader accepting a superset of correct answers",
                "LLM judge rewarding confidence, length, or politeness",
                "Reward model overoptimization on off-distribution outputs",
            ],
            fixes=[
                "Hold out a verifier the policy never trains against and evaluate on it",
                "Read the 20 highest-reward completions right now, by hand. This always works",
                "Add an adversarial unit test: does an empty or nonsense answer score > 0?",
                "Log proxy reward and ground truth on the same chart, permanently",
            ],
        )


class LengthExploit(Detector):
    """RLD-033 - the policy discovered that longer is better."""

    code = "RLD-033"
    title = "Length exploit"
    family = "reward"
    min_steps = 15
    cooldown = 60

    def applicable(self, h: RunHistory) -> bool:
        return len(h) >= self.min_steps and len(h.recent_rollouts(3)) >= 12

    def check(self, h: RunHistory) -> Optional[Diagnosis]:
        rollouts = h.recent_rollouts(4)
        lengths = [float(r.length) for r in rollouts]
        rewards = [float(r.reward) for r in rollouts]
        corr = stats.pearson(lengths, rewards)
        lens = h.series("seq_len_mean")
        if len(lens) < 10:
            return None
        l_first, l_second = stats.split_halves(lens)
        growth = stats.relative_change(stats.mean(l_first), stats.mean(l_second))
        if corr < 0.45 or growth < 0.25:
            return None
        return self._dx(
            h,
            Severity.WARN,
            "Completion length correlates with reward (r={}) and mean length grew {} "
            "across the window ({} -> {} tokens). The policy is being paid by the "
            "word.".format(
                stats.fmt(corr, 2),
                stats.pct(growth, 0),
                int(stats.mean(l_first)),
                int(stats.mean(l_second)),
            ),
            evidence={
                "corr(length, reward)": round(corr, 3),
                "length_growth": stats.pct(growth, 1),
                "mean_len_now": int(stats.mean(l_second)),
            },
            causes=[
                "Judge or reward model prefers verbose answers",
                "Reward summed over tokens instead of averaged or applied at sequence level",
                "Repetition earning partial credit from a fuzzy matcher",
                "No length penalty while the context budget still has headroom",
            ],
            fixes=[
                "Normalize reward per sequence, never per token, unless you mean it",
                "Add an explicit length penalty or a token budget to the reward",
                "Score length-matched pairs to test whether the judge is length-biased",
            ],
        )


class DeadReward(Detector):
    """RLD-060 - there is no learning signal at all."""

    code = "RLD-060"
    title = "Dead reward signal"
    family = "reward"
    min_steps = 10
    cooldown = 80
    requires = ("reward_std",)

    def check(self, h: RunHistory) -> Optional[Diagnosis]:
        sd = stats.mean(h.series("reward_std", 12))
        mu = stats.mean(h.series("reward_mean", 12))
        if sd > 1e-4:
            return None
        flavor = (
            "every rollout is scoring exactly {}".format(stats.fmt(mu, 3))
            if abs(mu) > 1e-9
            else "every rollout is scoring exactly zero"
        )
        return self._dx(
            h,
            Severity.CRITICAL,
            "Reward has zero variance across the batch: {}. No signal means no learning, "
            "regardless of how healthy the loss looks.".format(flavor),
            evidence={"reward_std": round(sd, 8), "reward_mean": round(mu, 5)},
            causes=[
                "Reward function raising and being swallowed by a bare except",
                "Verifier receiving the wrong field (prompt instead of completion)",
                "All completions truncated before reaching the answer",
                "Reward computed on a padded tensor and averaged away to a constant",
            ],
            fixes=[
                "Unit-test the reward function on three known-good and three known-bad strings",
                "Print one full (prompt, completion, reward) triple per 50 steps and read it",
                "Remove the try/except around the verifier and let it crash loudly",
            ],
        )


class TruncationBias(Detector):
    """RLD-042 - truncated sequences are silently poisoning the gradient."""

    code = "RLD-042"
    title = "Truncation bias"
    family = "reward"
    min_steps = 12
    cooldown = 60
    requires = ("truncated_frac",)

    def check(self, h: RunHistory) -> Optional[Diagnosis]:
        frac = stats.mean(h.series("truncated_frac", 10))
        if frac < 0.12:
            return None
        rollouts = h.recent_rollouts(4)
        trunc = [r.reward for r in rollouts if r.truncated]
        full = [r.reward for r in rollouts if not r.truncated]
        if len(trunc) < 4 or len(full) < 4:
            return None
        t_mu, f_mu = stats.mean(trunc), stats.mean(full)
        if f_mu - t_mu < 0.15 * max(abs(f_mu), 1e-6):
            return None
        severity = Severity.CRITICAL if frac > 0.3 else Severity.WARN
        return self._dx(
            h,
            severity,
            "{} of rollouts hit the length cap, and truncated rollouts average {} reward "
            "versus {} for complete ones. You are training the model that being cut off "
            "is a mistake it made.".format(stats.pct(frac), stats.fmt(t_mu, 3), stats.fmt(f_mu, 3)),
            evidence={
                "truncated_frac": round(frac, 3),
                "reward_truncated": round(t_mu, 4),
                "reward_complete": round(f_mu, 4),
            },
            causes=[
                "max_new_tokens too small for the task's natural answer length",
                "Truncated sequences given reward 0 instead of being masked out",
                "Reasoning traces growing past the cap as training progresses",
            ],
            fixes=[
                "Mask truncated sequences out of the loss instead of scoring them",
                "Raise max_new_tokens above the 99th percentile of successful completions",
                "Track truncated_frac as a first-class metric with an alert threshold",
            ],
        )
