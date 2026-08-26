# SyntropyRL | Reinforcement Learning Failure Diagnostics

A lightweight diagnostic library for detecting common failure modes in RL and LLM post-training runs. It analyzes rewards, rollouts, and training metrics, then reports likely causes, evidence, and suggested fixes.

SyntropyRL is designed with a strict failure-isolation principle: **a broken detector must never kill an expensive training run**. 

## Core Capabilities

- **14 Pre-built Detectors**: Identifies reward hacking, KL/log-probability divergence, advantage collapse, and gradient/value pathologies.
- **Robust Telemetry**: Native OpenTelemetry integration for tracing components and diagnosing distributed failures.
- **Framework Agnostic**: Integrates natively with TRL, CleanRL, or generic custom training loops.
- **Diagnostic Export**: JSON reporting and documented RL Failure Atlas for team-wide observability.

## Installation

```bash
pip install syntropyrl
```

For development (includes testing and formatting tools):
```bash
pip install -e .[dev]
```

## Quick Start

Wrap your optimizer step with the `Doctor` to safely catch and diagnose trajectory anomalies:

```python
from syntropyrl.core import Doctor, Step, Rollout

doctor = Doctor()

# During your training loop
step_data = Step(
    step=100,
    rollouts=[Rollout(reward=1.2, n_tokens=50)],
    metrics={"loss": 0.45}
)

# observe_step is guaranteed never to raise exceptions 
# that would crash the training run.
doctor.observe_step(step_data)

# Print a diagnostic report if anomalies are detected
if doctor.has_critical_issues():
    print(doctor.generate_report())
```

## Architecture

SyntropyRL separates observation from diagnosis:
1. **RunHistory**: A sliding window of optimization steps and rollouts.
2. **Detectors**: Pure, stateless functions that read from the `RunHistory` and return optional `Diagnosis` objects.
3. **Doctor**: The orchestrator that safely executes detectors, dedupes alerts, and generates the final JSON/text reports.

## Contributing

We welcome contributions. Please read `CONTRIBUTING.md` for details on our code of conduct, and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
