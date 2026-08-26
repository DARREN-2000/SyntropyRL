"""TRL / transformers adapter.

`transformers` is never imported at module load time, so `import syntropyrl` stays
free of heavy dependencies. The callback is built dynamically and subclasses
`TrainerCallback` only if it is importable.

Usage::

    from syntropyrl.integrations import trl_callback

    trainer = GRPOTrainer(..., callbacks=[trl_callback()])
"""

from __future__ import annotations

from typing import Any, Optional

from ..core import Doctor, Severity
from .generic import from_metrics


def make_trl_callback(
    doctor: Optional[Doctor] = None,
    min_severity: Severity = Severity.WARN,
    verbose: bool = True,
) -> Any:
    try:
        from transformers import TrainerCallback  # type: ignore

        base = TrainerCallback
    except Exception:  # transformers not installed - stay usable anyway
        base = object  # type: ignore[assignment]

    doc = doctor or Doctor(min_severity=min_severity, verbose=verbose)

    class SyntropyRLTrainerCallback(base):  # type: ignore[misc, valid-type]
        """Reads whatever the trainer logs and diagnoses it.

        TRL logs scalars only, so this yields the metric-based detectors
        (entropy collapse, KL blowup, clip saturation, value divergence).
        For the rollout-level detectors -- reward hacking, length exploit,
        truncation bias, logprob divergence -- pass rollouts explicitly via
        `doctor.observe(...)` inside your reward function, where you have the
        completions in hand.
        """

        def __init__(self) -> None:
            self.doctor = doc
            self._step = 0

        def on_log(self, args: Any, state: Any, control: Any, logs: Optional[dict] = None, **kw: Any):
            if not logs:
                return control
            step = int(getattr(state, "global_step", self._step) or self._step)
            self._step = step
            metrics = from_metrics(logs)
            if metrics:
                self.doctor.observe(step=step, **metrics)
            return control

        def on_train_end(self, args: Any, state: Any, control: Any, **kw: Any):
            print(self.doctor.report())
            return control

    return SyntropyRLTrainerCallback()
