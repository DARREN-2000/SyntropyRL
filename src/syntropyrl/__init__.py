"""syntropyrl - find out why your RL run is not learning.

    import syntropyrl

    doctor = syntropyrl.Doctor()
    for step, batch in enumerate(train_loop):
        doctor.observe(step=step, rollouts=batch.rollouts, **batch.metrics)
    print(doctor.report())

Zero dependencies, no telemetry, no config file. Every detector explains itself,
cites the number that triggered it, and links to a failure-mode writeup in the
RL Failure Atlas.
"""

from .atlas import ATLAS_URL, ENTRIES as ATLAS_ENTRIES
from .atlas import entries_by_family as atlas_families
from .atlas import entry as atlas_entry
from .atlas import render_text as atlas_text
from .core import (
    Detector,
    Diagnosis,
    Doctor,
    RLFailure,
    Rollout,
    RunHistory,
    Severity,
    Step,
)
from .detectors import DETECTOR_CLASSES, all_detectors, by_code, catalog
from .integrations import SyntropyRLCallback, from_metrics, trl_callback
from .integrations.generic import rollouts_from_records
from .parity import check_logprob_parity, parity_report
from .report import render, render_summary
from .simulate import SCENARIOS, run_scenario, simulate_run

__version__ = "0.1.0"

__all__ = [
    # the training-loop surface
    "Doctor",
    "Detector",
    "Diagnosis",
    "Rollout",
    "RunHistory",
    "Severity",
    "Step",
    "RLFailure",
    # detector registry
    "DETECTOR_CLASSES",
    "all_detectors",
    "by_code",
    "catalog",
    # the atlas
    "ATLAS_URL",
    "ATLAS_ENTRIES",
    "atlas_entry",
    "atlas_families",
    "atlas_text",
    # the five-minute parity check
    "check_logprob_parity",
    "parity_report",
    # adapters
    "SyntropyRLCallback",
    "trl_callback",
    "from_metrics",
    "rollouts_from_records",
    # simulation and rendering
    "simulate_run",
    "run_scenario",
    "SCENARIOS",
    "render",
    "render_summary",
    "__version__",
]
