"""Distribution family: the policy stopped being a distribution."""

from __future__ import annotations

from typing import Optional

from .. import stats
from ..core import Detector, Diagnosis, RunHistory, Severity


class EntropyCollapse(Detector):
    """RLD-007 - exploration is gone and the run is now decoration.

    Entropy decay is normal and desirable. Entropy *collapse* means the policy
    became deterministic, every rollout in a group is identical, advantages go
    to zero, and the remaining steps cannot recover.
    """

    code = "RLD-007"
    title = "Entropy collapse"
    family = "distribution"
    min_steps = 15
    cooldown = 50
    requires = ("entropy",)

    def check(self, h: RunHistory) -> Optional[Diagnosis]:
        series = h.series("entropy")
        if len(series) < 15:
            return None
        baseline = stats.median(series[: max(5, len(series) // 5)])
        recent = stats.mean(series[-6:])
        if baseline <= 0:
            return None
        drop = 1.0 - (recent / baseline)
        if drop < 0.55 or recent > 0.4:
            return None
        adv = h.series("advantage_std", 8)
        adv_mean = stats.mean(adv) if adv else None
        severity = Severity.CRITICAL if recent < 0.15 else Severity.WARN
        tail = ""
        if adv_mean is not None and adv_mean < 0.2:
            tail = " Advantage std has already fallen to {}, so the collapse is now " \
                   "self-reinforcing.".format(stats.fmt(adv_mean, 3))
        return self._dx(
            h,
            severity,
            "Entropy fell {} from its early baseline ({} -> {}). The policy is close to "
            "deterministic and can no longer explore.{}".format(
                stats.pct(drop, 0), stats.fmt(baseline, 3), stats.fmt(recent, 3), tail
            ),
            evidence={
                "entropy_baseline": round(baseline, 4),
                "entropy_now": round(recent, 4),
                "relative_drop": stats.pct(drop, 1),
                "advantage_std": None if adv_mean is None else round(adv_mean, 4),
            },
            causes=[
                "No entropy bonus, or an entropy coefficient of exactly 0.0",
                "Learning rate high enough that the policy sharpens faster than it learns",
                "Reward saturating: once everything scores 1.0 the policy just locks in",
                "Repeated epochs over the same rollout batch amplifying a single mode",
            ],
            fixes=[
                "Add a small entropy bonus (1e-3 to 1e-2) and watch whether entropy stabilizes",
                "Cap the KL to the reference policy to slow down sharpening",
                "Lower the number of inner epochs per rollout batch",
                "Checkpoint before collapse and restart from there with a lower LR",
            ],
        )


class ModeCollapse(Detector):
    """RLD-018 - the policy is emitting the same string over and over."""

    code = "RLD-018"
    title = "Mode collapse (duplicate completions)"
    family = "distribution"
    min_steps = 10
    cooldown = 60
    requires = ("duplicate_frac",)

    def check(self, h: RunHistory) -> Optional[Diagnosis]:
        dup = stats.mean(h.series("duplicate_frac", 8))
        if dup < 0.35:
            return None
        severity = Severity.CRITICAL if dup > 0.6 else Severity.WARN
        return self._dx(
            h,
            severity,
            "{} of rollouts are exact duplicates of another rollout in the same batch. "
            "You are sampling one sequence and paying for many.".format(stats.pct(dup)),
            evidence={"duplicate_frac": round(dup, 3)},
            causes=[
                "Sampling temperature too low, or greedy decoding left on in the rollout path",
                "Entropy collapse already in progress",
                "Same seed reused across the group, so all samples follow one trajectory",
                "top_k / top_p set so tight that only one continuation is reachable",
            ],
            fixes=[
                "Verify the generation config actually used at rollout time (print it)",
                "Give each rollout in a group a distinct seed",
                "Raise temperature or loosen top_p, then confirm duplicate_frac drops",
            ],
        )


class DistributionalDrift(Detector):
    """RLD-019 - Distributional Drift.

    Tracks KL divergence spikes alongside policy entropy collapse.
    """

    code = "RLD-019"
    title = "Distributional Drift (KL + Entropy)"
    family = "distribution"
    min_steps = 10
    cooldown = 50
    requires = ("kl", "entropy")

    def check(self, h: RunHistory) -> Optional[Diagnosis]:
        kl_series = h.series("kl")
        ent_series = h.series("entropy")
        
        if len(kl_series) < 10 or len(ent_series) < 10:
            return None
            
        kl_baseline = stats.median(kl_series[: max(5, len(kl_series) // 5)])
        kl_recent = stats.mean(kl_series[-5:])
        ent_baseline = stats.median(ent_series[: max(5, len(ent_series) // 5)])
        ent_recent = stats.mean(ent_series[-5:])
        
        if kl_baseline <= 0 or ent_baseline <= 0:
            return None
            
        kl_spike = kl_recent / kl_baseline
        ent_drop = 1.0 - (ent_recent / ent_baseline)
        
        if kl_spike > 3.0 and ent_drop > 0.3:
            severity = Severity.CRITICAL if kl_spike > 5.0 and ent_drop > 0.5 else Severity.WARN
            return self._dx(
                h,
                severity,
                f"Severe distributional drift detected. KL spiked {kl_spike:.1f}x while entropy dropped {stats.pct(ent_drop)}.",
                evidence={
                    "kl_baseline": round(kl_baseline, 4),
                    "kl_recent": round(kl_recent, 4),
                    "entropy_baseline": round(ent_baseline, 4),
                    "entropy_recent": round(ent_recent, 4)
                },
                causes=["KL penalty coefficient too low", "Learning rate too high"],
                fixes=["Increase KL penalty", "Lower learning rate"]
            )
        return None
