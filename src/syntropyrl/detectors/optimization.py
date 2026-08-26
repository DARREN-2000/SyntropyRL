"""Optimization family: the update itself is broken or wasted."""

from __future__ import annotations

from typing import Optional

from .. import stats
from ..core import Detector, Diagnosis, RunHistory, Severity


class AdvantageCollapse(Detector):
    """RLD-001 - the batch carries no gradient.

    In GRPO-style methods, a group where every rollout earns the same reward
    produces exactly zero advantage. If most groups are degenerate you are
    paying full price for compute that teaches nothing.
    """

    code = "RLD-001"
    title = "Advantage collapse (zero-gradient batches)"
    family = "optimization"
    min_steps = 8
    cooldown = 50

    def applicable(self, h: RunHistory) -> bool:
        return len(h) >= self.min_steps and (
            h.has("advantage_std", 4) or h.has("degenerate_group_frac", 4)
        )

    def check(self, h: RunHistory) -> Optional[Diagnosis]:
        adv = stats.mean(h.series("advantage_std", 10))
        dead = stats.mean(h.series("degenerate_group_frac", 10))
        if dead < 0.55 and adv > 0.15:
            return None
        acc = h.series("accuracy", 10)
        acc_mean = stats.mean(acc) if acc else None
        hint = ""
        if acc_mean is not None:
            if acc_mean < 0.05:
                hint = " Accuracy is near 0: the task is too hard for the base policy."
            elif acc_mean > 0.95:
                hint = " Accuracy is near 1: the task is too easy and there is nothing left to learn."
        severity = Severity.CRITICAL if dead > 0.75 else Severity.WARN
        return self._dx(
            h,
            severity,
            "{} of groups have identical rewards across all rollouts, so their advantages "
            "are exactly zero. Effective batch size is a fraction of what you think.{}".format(
                stats.pct(dead), hint
            ),
            evidence={
                "degenerate_group_frac": round(dead, 3),
                "advantage_std": round(adv, 4),
                "accuracy": None if acc_mean is None else round(acc_mean, 3),
            },
            causes=[
                "Prompt difficulty is not matched to current policy ability",
                "Binary reward with no partial credit on an all-or-nothing task",
                "Group size too small to produce reward variance",
                "Temperature too low, so rollouts within a group are near-identical",
            ],
            fixes=[
                "Filter prompts online: keep only those with 0 < pass_rate < 1 for this policy",
                "Raise group size or sampling temperature to recover intra-group variance",
                "Add dense or partial-credit reward terms alongside the binary signal",
                "Curriculum: sort prompts by measured pass rate and advance as accuracy climbs",
            ],
        )


class KLBlowup(Detector):
    """RLD-021 - the policy has left the reference model behind."""

    code = "RLD-021"
    title = "KL divergence blowup"
    family = "optimization"
    min_steps = 12
    cooldown = 50
    requires = ("kl",)

    def check(self, h: RunHistory) -> Optional[Diagnosis]:
        series = h.series("kl")
        if len(series) < 12:
            return None
        early = stats.median(series[: max(4, len(series) // 4)])
        recent = stats.mean(series[-6:])
        if recent < 0.5:
            return None
        growth = recent / max(early, 1e-6)
        if growth < 5 and recent < 2.0:
            return None
        severity = Severity.CRITICAL if recent > 2.0 else Severity.WARN
        return self._dx(
            h,
            severity,
            "KL to the reference policy reached {} (was {} early on, {}x growth). The "
            "policy is drifting into a region where the reward model was never "
            "calibrated.".format(stats.fmt(recent, 3), stats.fmt(early, 4), int(growth)),
            evidence={
                "kl_recent": round(recent, 4),
                "kl_early": round(early, 5),
                "growth_factor": round(growth, 1),
            },
            causes=[
                "KL coefficient too low, or adaptive KL controller not actually updating",
                "Reward model overoptimization: the policy found an off-distribution exploit",
                "Learning rate too high for the effective batch size",
                "KL estimated with the biased k1 estimator and reported far below the truth",
            ],
            fixes=[
                "Raise the KL penalty or re-enable the adaptive controller and log its beta",
                "Switch to the k3 unbiased KL estimator for reporting",
                "Sample completions now and read them: overoptimization is obvious to a human",
                "Refresh the reference policy, or hard-stop on a KL budget",
            ],
        )


class ClipSaturation(Detector):
    """RLD-055 - too much of the batch is being clipped away."""

    code = "RLD-055"
    title = "Clip saturation from stale off-policy data"
    family = "optimization"
    min_steps = 10
    cooldown = 50
    requires = ("clip_frac",)

    def check(self, h: RunHistory) -> Optional[Diagnosis]:
        recent = stats.mean(h.series("clip_frac", 10))
        if recent < 0.3:
            return None
        severity = Severity.CRITICAL if recent > 0.5 else Severity.WARN
        return self._dx(
            h,
            severity,
            "{} of tokens are hitting the PPO clip boundary. Most of your batch is being "
            "discarded, and the surviving gradient is biased toward whatever happened to "
            "stay in range.".format(stats.pct(recent)),
            evidence={"clip_frac": round(recent, 3), "healthy_range": "0.02 - 0.20"},
            causes=[
                "Rollout data is too stale: too many optimizer steps per generation batch",
                "Async pipeline running many steps behind the inference weights",
                "Learning rate too high, so one step moves the policy past the trust region",
                "Clip epsilon set too tight for the amount of off-policyness",
            ],
            fixes=[
                "Reduce inner epochs / minibatch reuse per rollout batch",
                "Bound staleness explicitly: drop rollouts older than N weight versions",
                "Apply truncated importance sampling (TIS) to correct for staleness",
                "Log the ratio distribution, not just clip_frac, and look at the tails",
            ],
        )


class ValueDivergence(Detector):
    """RLD-084 - the critic is getting worse, not better."""

    code = "RLD-084"
    title = "Value function divergence"
    family = "optimization"
    min_steps = 20
    cooldown = 60
    requires = ("value_loss",)

    def check(self, h: RunHistory) -> Optional[Diagnosis]:
        series = h.series("value_loss")
        if len(series) < 20:
            return None
        first, second = stats.split_halves(series)
        a, b = stats.mean(first), stats.mean(second)
        if b <= a * 1.5 or stats.slope(series) <= 0:
            return None
        return self._dx(
            h,
            Severity.WARN,
            "Value loss is growing: {} -> {} between the first and second half of the "
            "window. Your advantages are being computed against a critic that is "
            "actively diverging.".format(stats.fmt(a, 3), stats.fmt(b, 3)),
            evidence={
                "value_loss_first_half": round(a, 4),
                "value_loss_second_half": round(b, 4),
                "slope_per_step": round(stats.slope(series), 6),
            },
            causes=[
                "Value head learning rate too high, or shared trunk fighting the policy loss",
                "Reward scale drifted and the critic is chasing a moving target",
                "Value clipping range too small relative to actual returns",
                "Critic initialized from a model that never saw this reward scale",
            ],
            fixes=[
                "Normalize or whiten returns before fitting the critic",
                "Give the value head its own lower learning rate",
                "Warm up the critic for a few hundred steps with the policy frozen",
                "Consider a critic-free method (GRPO/RLOO) if the critic never stabilizes",
            ],
        )
