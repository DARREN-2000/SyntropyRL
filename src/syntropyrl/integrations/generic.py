"""Generic adapter: map arbitrary framework metric names onto syntropyrl's.

Every RL framework calls the same quantity something different. This module owns
that translation table so detectors can assume one vocabulary.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from ..core import Doctor, Rollout, Severity, Step

# Canonical name -> aliases seen in the wild (verl, TRL, OpenRLHF, SB3, RLlib).
ALIASES: Dict[str, Sequence[str]] = {
    "entropy": (
        "entropy", "entropy_loss", "policy/entropy", "actor/entropy",
        "objective/entropy", "train/entropy", "entropy_bonus", "ent",
    ),
    "kl": (
        "kl", "kl_div", "kl_divergence", "objective/kl", "policy/approxkl",
        "actor/kl", "train/kl", "approx_kl", "approxkl", "kl_to_ref",
    ),
    "clip_frac": (
        "clip_frac", "clipfrac", "policy/clipfrac", "actor/pg_clipfrac",
        "train/clip_ratio", "clip_ratio", "ratio_clipped",
    ),
    "grad_norm": (
        "grad_norm", "gradient_norm", "train/grad_norm", "actor/grad_norm",
        "grad_norm_before_clip",
    ),
    "value_loss": (
        "value_loss", "critic/vf_loss", "vf_loss", "train/value_loss",
        "loss/value", "critic_loss",
    ),
    "reward_mean": (
        "reward_mean", "reward", "rewards/mean", "critic/rewards/mean",
        "train/reward", "ep_rew_mean", "episode_reward_mean", "score",
    ),
    "reward_std": ("reward_std", "rewards/std", "critic/rewards/std"),
    "accuracy": (
        "accuracy", "acc", "task_accuracy", "eval_accuracy", "pass_rate",
        "val/accuracy", "reward/accuracy", "verifier_accuracy",
    ),
    "seq_len_mean": (
        "seq_len_mean", "response_length", "completion_length",
        "response_length/mean", "train/completion_length", "mean_response_length",
    ),
    "truncated_frac": (
        "truncated_frac", "clip_ratio/truncated", "response_length/clip_ratio",
        "truncation_rate",
    ),
    "logprob_gap": (
        "logprob_gap", "logprob_diff", "actor/logprob_diff",
        "rollout_vs_actor_logprob_diff", "training_inference_mismatch",
    ),
    "advantage_std": ("advantage_std", "advantages/std", "critic/advantages/std"),
    "lr": ("lr", "learning_rate", "train/learning_rate", "actor_lr"),
}

_LOOKUP: Dict[str, str] = {}
for _canon, _alts in ALIASES.items():
    for _a in _alts:
        _LOOKUP[_a.lower()] = _canon


def from_metrics(raw: Mapping[str, Any]) -> Dict[str, float]:
    """Translate a framework metrics dict into syntropyrl's vocabulary.

    Unknown keys are dropped rather than guessed at. Known keys win over
    unknown ones, and an exact canonical name always wins over an alias.
    """
    out: Dict[str, float] = {}
    for key, value in raw.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        canon = _LOOKUP.get(str(key).lower())
        if canon is None:
            continue
        if canon in raw and str(key).lower() != canon:
            continue  # the canonical key is present verbatim; prefer it
        out[canon] = float(value)
    return out


def rollouts_from_records(records: Iterable[Mapping[str, Any]]) -> List[Rollout]:
    """Build Rollouts from loosely-shaped dicts (common field names accepted)."""
    def pick(d: Mapping[str, Any], *names: str, default: Any = None) -> Any:
        for n in names:
            if n in d and d[n] is not None:
                return d[n]
        return default

    out: List[Rollout] = []
    for rec in records:
        out.append(
            Rollout(
                reward=float(pick(rec, "reward", "score", "return", default=0.0)),
                completion=str(pick(rec, "completion", "response", "output", "text", default="")),
                prompt=str(pick(rec, "prompt", "query", "input", default="")),
                n_tokens=int(pick(rec, "n_tokens", "response_length", "length", default=0) or 0),
                truncated=bool(pick(rec, "truncated", "is_truncated", "hit_max_len", default=False)),
                correct=pick(rec, "correct", "is_correct", "passed"),
                group_id=pick(rec, "group_id", "prompt_id", "uid", "question_id"),
                rollout_logprobs=pick(rec, "rollout_logprobs", "vllm_logprobs", "gen_logprobs"),
                trainer_logprobs=pick(rec, "trainer_logprobs", "actor_logprobs", "old_logprobs"),
            )
        )
    return out


class SyntropyRLCallback:
    """Duck-typed callback usable from any loop that can call a function.

    Deliberately not a subclass of anything, so it imports with zero deps::

        cb = SyntropyRLCallback()
        cb.on_step(step=i, metrics=trainer_metrics, rollouts=batch_records)
        print(cb.doctor.report())
    """

    def __init__(
        self,
        doctor: Optional[Doctor] = None,
        min_severity: Severity = Severity.WARN,
        verbose: bool = True,
    ) -> None:
        self.doctor = doctor or Doctor(min_severity=min_severity, verbose=verbose)

    def on_step(
        self,
        step: Optional[int] = None,
        metrics: Optional[Mapping[str, Any]] = None,
        rollouts: Optional[Iterable[Mapping[str, Any]]] = None,
    ):
        record = Step(
            step=step if step is not None else len(self.doctor.history),
            rollouts=rollouts_from_records(rollouts or []),
            metrics=from_metrics(metrics or {}),
        )
        return self.doctor.observe_step(record)

    def report(self) -> str:
        return self.doctor.report()
