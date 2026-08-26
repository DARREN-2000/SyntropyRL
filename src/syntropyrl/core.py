"""Core data model and detector framework.

The mental model is deliberately small:

    Rollout   -> one generated sequence and what happened to it
    Step      -> one optimizer step: a batch of rollouts + scalar metrics
    RunHistory-> a sliding window of Steps
    Detector  -> reads RunHistory, optionally returns a Diagnosis
    Doctor    -> owns the history, runs detectors, dedupes, reports

Detectors never mutate state and never raise: a broken detector must not be
able to kill a training run that costs real money.
"""

from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence

from opentelemetry import trace

from . import stats

tracer = trace.get_tracer(__name__)

ATLAS_BASE = "https://syntropyrl.dev/atlas.html"


class Severity(Enum):
    INFO = "info"
    WARN = "warn"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return {"info": 0, "warn": 1, "critical": 2}[self.value]

    def __lt__(self, other: "Severity") -> bool:
        return self.rank < other.rank

    def __ge__(self, other: "Severity") -> bool:
        return self.rank >= other.rank


@dataclass
class Rollout:
    """One generated sequence.

    Only `reward` is required. Every additional field unlocks more detectors,
    which is the whole adoption strategy: partial instrumentation still works.
    """

    reward: float = 0.0
    completion: str = ""
    prompt: str = ""
    n_tokens: int = 0
    truncated: bool = False
    correct: Optional[bool] = None
    group_id: Optional[str] = None
    # Logprobs of the *same* tokens as scored by the generator (vLLM/SGLang)
    # and as recomputed by the trainer. The gap between them is the single
    # most expensive silent bug in modern async RL.
    rollout_logprobs: Optional[List[float]] = None
    trainer_logprobs: Optional[List[float]] = None

    @property
    def length(self) -> int:
        if self.n_tokens:
            return self.n_tokens
        if self.rollout_logprobs:
            return len(self.rollout_logprobs)
        return len(self.completion.split())

    def logprob_gap(self) -> Optional[float]:
        """Mean absolute per-token divergence between generator and trainer."""
        a, b = self.rollout_logprobs, self.trainer_logprobs
        if not a or not b:
            return None
        m = min(len(a), len(b))
        if m == 0:
            return None
        diffs = [abs(float(a[i]) - float(b[i])) for i in range(m) if stats.finite(a[i]) and stats.finite(b[i])]
        return stats.mean(diffs) if diffs else None


@dataclass
class Step:
    """One optimizer step."""

    step: int
    rollouts: List[Rollout] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    _merged_cache: Optional[Dict[str, float]] = field(default=None, init=False, repr=False)
    _groups_cache: Optional[List[List[Rollout]]] = field(default=None, init=False, repr=False)

    def merged(self) -> Dict[str, float]:
        """Metrics explicitly logged, plus everything we can derive ourselves.

        Explicit metrics always win, so a framework that already computes
        entropy correctly is never second-guessed.
        """
        if self._merged_cache is not None:
            return self._merged_cache
        derived: Dict[str, float] = {}
        rs = self.rollouts
        if rs:
            rewards = [r.reward for r in rs]
            derived["reward_mean"] = stats.mean(rewards)
            derived["reward_std"] = stats.std(rewards)
            derived["seq_len_mean"] = stats.mean([float(r.length) for r in rs])
            derived["truncated_frac"] = stats.mean([1.0 if r.truncated else 0.0 for r in rs])
            graded = [r for r in rs if r.correct is not None]
            if graded:
                derived["accuracy"] = stats.mean([1.0 if r.correct else 0.0 for r in graded])
            gaps = [g for g in (r.logprob_gap() for r in rs) if g is not None]
            if gaps:
                derived["logprob_gap"] = stats.mean(gaps)
            derived["advantage_std"] = self._group_advantage_std()
            derived["degenerate_group_frac"] = self._degenerate_group_frac()
            derived["duplicate_frac"] = stats.duplicate_share([r.completion for r in rs])
        out = dict(derived)
        out.update({k: v for k, v in self.metrics.items() if stats.finite(v)})
        self._merged_cache = out
        return out

    def _groups(self) -> List[List[Rollout]]:
        if self._groups_cache is not None:
            return self._groups_cache
        if not self.rollouts:
            self._groups_cache = []
            return self._groups_cache
        if not any(r.group_id for r in self.rollouts):
            self._groups_cache = [list(self.rollouts)]
            return self._groups_cache
        buckets: Dict[str, List[Rollout]] = {}
        for r in self.rollouts:
            buckets.setdefault(r.group_id or "__none__", []).append(r)
        self._groups_cache = list(buckets.values())
        return self._groups_cache

    def _group_advantage_std(self) -> float:
        """Std of group-normalized advantages (the GRPO-style signal)."""
        advs: List[float] = []
        for g in self._groups():
            rewards = [r.reward for r in g]
            mu, sd = stats.mean(rewards), stats.std(rewards)
            if sd == 0:
                advs.extend([0.0] * len(rewards))
            else:
                advs.extend([(r - mu) / sd for r in rewards])
        return stats.std(advs)

    def _degenerate_group_frac(self) -> float:
        """Fraction of groups where every rollout got the same reward.

        These groups contribute exactly zero gradient in GRPO. If most of your
        batch is degenerate you are paying for compute that teaches nothing.
        """
        groups = self._groups()
        if not groups:
            return 0.0
        dead = sum(1 for g in groups if len(g) > 1 and stats.std([r.reward for r in g]) == 0)
        return dead / len(groups)

    def top_rollouts(self, k: int = 8) -> List[Rollout]:
        return sorted(self.rollouts, key=lambda r: r.reward, reverse=True)[:k]


class RunHistory:
    """Sliding window over Steps with convenient series access."""

    def __init__(self, window: int = 250) -> None:
        self.window = window
        self.steps: deque = deque(maxlen=window)
        self._cache: deque = deque(maxlen=window)
        self._series_memo: Dict[str, List[float]] = {}

    def add(self, step: Step) -> None:
        self.steps.append(step)
        self._cache.append(step.merged())
        self._series_memo.clear()

    def __len__(self) -> int:
        return len(self.steps)

    @property
    def last(self) -> Optional[Step]:
        return self.steps[-1] if self.steps else None

    @property
    def step_index(self) -> int:
        return self.steps[-1].step if self.steps else 0

    def series(self, key: str, n: Optional[int] = None) -> List[float]:
        """All finite values of `key`, oldest first. `n` limits to the tail."""
        if key not in self._series_memo:
            self._series_memo[key] = [m[key] for m in self._cache if key in m and stats.finite(m[key])]
        vals = self._series_memo[key]
        return vals[-n:] if n else vals

    def has(self, key: str, min_points: int = 1) -> bool:
        return len(self.series(key)) >= min_points

    def recent_rollouts(self, n_steps: int = 3) -> List[Rollout]:
        out: List[Rollout] = []
        for st in list(self.steps)[-n_steps:]:
            out.extend(st.rollouts)
        return out


@dataclass
class Diagnosis:
    code: str
    title: str
    severity: Severity
    summary: str
    step: int = 0
    family: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    causes: List[str] = field(default_factory=list)
    fixes: List[str] = field(default_factory=list)

    @property
    def url(self) -> str:
        return "{}#{}".format(ATLAS_BASE, self.code)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "title": self.title,
            "severity": self.severity.value,
            "summary": self.summary,
            "step": self.step,
            "family": self.family,
            "evidence": self.evidence,
            "causes": self.causes,
            "fixes": self.fixes,
            "url": self.url,
        }


class Detector:
    """Base class. Subclasses implement `check`."""

    code: str = "RLD-000"
    title: str = "unnamed"
    family: str = "general"
    min_steps: int = 8
    cooldown: int = 40  # steps to wait before re-firing the same code
    requires: Sequence[str] = ()

    def applicable(self, h: RunHistory) -> bool:
        if len(h) < self.min_steps:
            return False
        return all(h.has(k, 2) for k in self.requires)

    def check(self, h: RunHistory) -> Optional[Diagnosis]:  # pragma: no cover
        raise NotImplementedError

    # -- helper ------------------------------------------------------------
    def _dx(
        self,
        h: RunHistory,
        severity: Severity,
        summary: str,
        evidence: Optional[Dict[str, Any]] = None,
        causes: Optional[Iterable[str]] = None,
        fixes: Optional[Iterable[str]] = None,
    ) -> Diagnosis:
        return Diagnosis(
            code=self.code,
            title=self.title,
            severity=severity,
            summary=summary,
            step=h.step_index,
            family=self.family,
            evidence=evidence or {},
            causes=list(causes or []),
            fixes=list(fixes or []),
        )


class Doctor:
    """Runs detectors over a training run and reports diagnoses.

    Usage inside any training loop::

        doc = Doctor()
        for step in range(n):
            ...
            doc.observe(step=step, rollouts=rollouts, entropy=e, kl=kl)
    """

    def __init__(
        self,
        detectors: Optional[Sequence[Detector]] = None,
        window: int = 250,
        min_severity: Severity = Severity.WARN,
        verbose: bool = True,
        raise_on_critical: bool = False,
        color: Optional[bool] = None,
    ) -> None:
        from .detectors import all_detectors

        self.detectors: List[Detector] = list(detectors if detectors is not None else all_detectors())
        self.history = RunHistory(window=window)
        self.min_severity = min_severity
        self.verbose = verbose
        self.raise_on_critical = raise_on_critical
        self.diagnoses: List[Diagnosis] = []
        self._last_fired: Dict[str, int] = {}
        self._errors: Dict[str, str] = {}
        if color is None:
            color = os.environ.get("NO_COLOR") is None
        self.color = bool(color)

    # -- ingestion ---------------------------------------------------------
    def observe(
        self,
        step: Optional[int] = None,
        rollouts: Optional[Sequence[Rollout]] = None,
        **metrics: float,
    ) -> List[Diagnosis]:
        idx = step if step is not None else len(self.history)
        record = Step(step=idx, rollouts=list(rollouts or []), metrics=dict(metrics))
        return self.observe_step(record)

    def observe_step(self, record: Step) -> List[Diagnosis]:
        with tracer.start_as_current_span("syntropyrl.observe_step", attributes={"step": record.step}):
            self.history.add(record)
            fired: List[Diagnosis] = []
            for det in self.detectors:
                try:
                    if not det.applicable(self.history):
                        continue
                    last = self._last_fired.get(det.code)
                    if last is not None and record.step - last < det.cooldown:
                        continue
                    dx = det.check(self.history)
                except Exception as exc:  # a detector must never kill a run
                    self._errors[det.code] = repr(exc)
                    continue
                if dx is None or dx.severity.rank < self.min_severity.rank:
                    continue
                
                # Trace when a detector catches an anomaly
                with tracer.start_as_current_span(
                    "syntropyrl.detector.anomaly", 
                    attributes={
                        "detector.code": det.code,
                        "detector.title": det.title,
                        "diagnosis.severity": dx.severity.value,
                        "step": record.step
                    }
                ):
                    pass

                self._last_fired[det.code] = record.step
                self.diagnoses.append(dx)
                fired.append(dx)
            if fired and self.verbose:
                from .report import render

                print(render(fired, color=self.color))
            if self.raise_on_critical:
                crit = [d for d in fired if d.severity is Severity.CRITICAL]
                if crit:
                    raise RLFailure(crit)
            return fired

    # -- output ------------------------------------------------------------
    def report(self, color: Optional[bool] = None) -> str:
        from .report import render_summary

        return render_summary(
            self.diagnoses,
            steps=len(self.history),
            color=self.color if color is None else color,
        )

    def counts(self) -> Dict[str, int]:
        out = {"critical": 0, "warn": 0, "info": 0}
        for d in self.diagnoses:
            out[d.severity.value] += 1
        return out

    def unique_codes(self) -> List[str]:
        seen: List[str] = []
        for d in self.diagnoses:
            if d.code not in seen:
                seen.append(d.code)
        return seen

    def to_json(self, path: Optional[str] = None) -> str:
        payload = {
            "steps_observed": len(self.history),
            "counts": self.counts(),
            "diagnoses": [d.to_dict() for d in self.diagnoses],
            "detector_errors": self._errors,
        }
        text = json.dumps(payload, indent=2)
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
        return text


class RLFailure(RuntimeError):
    """Raised when raise_on_critical=True and a critical diagnosis fires."""

    def __init__(self, diagnoses: Sequence[Diagnosis]) -> None:
        self.diagnoses = list(diagnoses)
        codes = ", ".join(d.code for d in diagnoses)
        super().__init__("syntropyrl halted training: {}".format(codes))
