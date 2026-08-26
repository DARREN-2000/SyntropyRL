"""Numerics family: the bugs where your math is quietly wrong.

These are the expensive ones. Nothing in the loss curve looks unusual, the run
burns a week of GPU time, and the policy learns nothing because the importance
ratios were garbage from step one.
"""

from __future__ import annotations

from typing import List, Optional

from .. import stats
from ..core import Detector, Diagnosis, RunHistory, Severity

# A well-behaved setup keeps generator and trainer logprobs within ~1e-4.
HEALTHY_GAP = 1e-3


class TrainerGeneratorDivergence(Detector):
    """RLD-014 - the single most costly silent failure in async RL.

    The generator (vLLM / SGLang) scores the tokens it sampled. The trainer
    recomputes logprobs for those same tokens. If the two disagree, every
    importance ratio is wrong, PPO clipping silently becomes a no-op, and the
    gradient points somewhere arbitrary.
    """

    code = "RLD-014"
    title = "Trainer/generator logprob divergence"
    family = "numerics"
    min_steps = 5
    cooldown = 60
    requires = ("logprob_gap",)

    def check(self, h: RunHistory) -> Optional[Diagnosis]:
        recent = h.series("logprob_gap", 10)
        gap = stats.mean(recent)
        if gap <= HEALTHY_GAP * 3:
            return None

        rollouts = [r for r in h.recent_rollouts(3) if r.logprob_gap() is not None]
        gaps = [r.logprob_gap() or 0.0 for r in rollouts]
        lengths = [float(r.length) for r in rollouts]
        corr = stats.pearson(lengths, gaps)
        bad_share = stats.mean([1.0 if g > HEALTHY_GAP * 10 else 0.0 for g in gaps])

        causes = [
            "Chunked prefill or a different attention kernel between rollout and training",
            "dtype mismatch: bf16 generation vs fp32 recomputation (or the reverse)",
            "Sampling parameters applied to logprobs: temperature/top-p folded into the "
            "generator's returned logprobs but not the trainer's",
            "Off-by-one token alignment between prompt and completion boundaries",
            "Weights not actually synced to the inference engine before the rollout",
        ]
        if corr > 0.4:
            causes.insert(
                0,
                "Divergence grows with sequence length (r={}), which strongly suggests "
                "chunked prefill or a paged-attention boundary bug".format(stats.fmt(corr, 2)),
            )
        fixes = [
            "Run syntropyrl.check_logprob_parity(model, engine, prompts) before training",
            "Disable chunked prefill (enable_chunked_prefill=False) and re-measure",
            "Recompute the ratio as exp(trainer_lp - trainer_lp.detach()) so the ratio is "
            "exactly 1.0 at step 0, then assert it",
            "Log max|gap| per batch as a hard training assertion, not a chart",
        ]
        severity = Severity.CRITICAL if gap > HEALTHY_GAP * 10 else Severity.WARN
        return self._dx(
            h,
            severity,
            "Generator and trainer disagree on logprobs by {} per token (healthy < {}). "
            "Importance ratios are invalid, so PPO/GRPO clipping is currently a no-op.".format(
                stats.fmt(gap), stats.fmt(HEALTHY_GAP)
            ),
            evidence={
                "mean_abs_logprob_gap": round(gap, 6),
                "healthy_threshold": HEALTHY_GAP,
                "rollouts_above_10x_threshold": stats.pct(bad_share),
                "corr(seq_len, gap)": round(corr, 3),
            },
            causes=causes,
            fixes=fixes,
        )


class TemplateMismatch(Detector):
    """RLD-092 - divergence concentrated at the start of the sequence.

    When only the first few tokens disagree, the numbers are usually fine and
    the *prompt* is wrong: a different chat template, a doubled BOS, or a
    system prompt present at generation and missing at training.
    """

    code = "RLD-092"
    title = "Tokenizer or chat-template mismatch"
    family = "numerics"
    min_steps = 5
    cooldown = 80

    HEAD = 8

    def applicable(self, h: RunHistory) -> bool:
        if len(h) < self.min_steps:
            return False
        return any(
            r.rollout_logprobs and r.trainer_logprobs for r in h.recent_rollouts(2)
        )

    def check(self, h: RunHistory) -> Optional[Diagnosis]:
        head_gaps: List[float] = []
        tail_gaps: List[float] = []
        for r in h.recent_rollouts(3):
            a, b = r.rollout_logprobs, r.trainer_logprobs
            if not a or not b:
                continue
            m = min(len(a), len(b))
            if m < self.HEAD * 3:
                continue
            diffs = [abs(float(a[i]) - float(b[i])) for i in range(m)]
            head_gaps.append(stats.mean(diffs[: self.HEAD]))
            tail_gaps.append(stats.mean(diffs[self.HEAD :]))
        if len(head_gaps) < 3:
            return None
        head, tail = stats.mean(head_gaps), stats.mean(tail_gaps)
        if head <= HEALTHY_GAP * 5:
            return None
        ratio = head / max(tail, 1e-9)
        if ratio < 8:
            return None
        return self._dx(
            h,
            Severity.CRITICAL,
            "Logprob divergence is {}x larger in the first {} tokens than in the rest of "
            "the sequence. The tensors are fine; the prompt is not.".format(
                stats.fmt(ratio, 1), self.HEAD
            ),
            evidence={
                "head_gap": round(head, 6),
                "tail_gap": round(tail, 6),
                "head_tail_ratio": round(ratio, 2),
            },
            causes=[
                "apply_chat_template used at generation but not when tokenizing for training",
                "Double BOS: template adds it and the tokenizer adds it again",
                "System prompt present in the rollout but stripped in the training batch",
                "add_special_tokens defaulting differently on the two paths",
            ],
            fixes=[
                "Assert token-id equality, not just text equality, between the two paths",
                "Tokenize once at rollout time and pass token ids to the trainer verbatim",
                "Print repr() of the first 32 decoded tokens from both paths and diff them",
            ],
        )


class GradientPathology(Detector):
    """RLD-071 - NaN, inf, or spiking gradient norms."""

    code = "RLD-071"
    title = "Gradient norm pathology"
    family = "numerics"
    min_steps = 10
    cooldown = 40

    def applicable(self, h: RunHistory) -> bool:
        return len(h) >= self.min_steps and (
            h.has("grad_norm", 5) or any("grad_norm" in s.metrics for s in h.steps)
        )

    def check(self, h: RunHistory) -> Optional[Diagnosis]:
        raw = [s.metrics.get("grad_norm") for s in h.steps if "grad_norm" in s.metrics]
        nonfinite = sum(1 for v in raw if not stats.finite(v))
        series = h.series("grad_norm", 60)
        if nonfinite:
            return self._dx(
                h,
                Severity.CRITICAL,
                "{} of the last {} steps produced a non-finite gradient norm. Those steps "
                "either corrupted your weights or were silently skipped.".format(
                    nonfinite, len(raw)
                ),
                evidence={"nonfinite_steps": nonfinite, "observed_steps": len(raw)},
                causes=[
                    "log(0) in the ratio because a token had probability 0 under the new policy",
                    "Reward or advantage containing inf/NaN and propagating through the loss",
                    "Overflow in fp16 without a loss scaler",
                ],
                fixes=[
                    "Clamp logratios before exp(): torch.clamp(logratio, -20, 20)",
                    "Assert torch.isfinite on rewards and advantages at batch construction",
                    "Switch fp16 to bf16, or verify the grad scaler is actually attached",
                ],
            )
        if len(series) < 12:
            return None
        med = stats.median(series)
        peak = max(series[-12:])
        if med > 0 and peak > 20 * med and peak > 1.0:
            return self._dx(
                h,
                Severity.WARN,
                "Gradient norm spiked to {} against a median of {} ({}x). Clipping is "
                "absorbing the update, so these steps contribute direction without "
                "magnitude.".format(stats.fmt(peak, 2), stats.fmt(med, 2), int(peak / med)),
                evidence={
                    "median_grad_norm": round(med, 4),
                    "peak_grad_norm": round(peak, 4),
                    "spike_ratio": round(peak / med, 1),
                },
                causes=[
                    "A handful of very high-advantage sequences dominating the batch",
                    "Advantage normalization computed over too small a batch",
                    "Learning rate warmup ended too abruptly",
                ],
                fixes=[
                    "Normalize advantages per batch and clip them to +/- 5",
                    "Lower max_grad_norm and check what fraction of steps are clipped",
                    "Inspect the top-advantage sequences in the spiking batch by hand",
                ],
            )
        return None
