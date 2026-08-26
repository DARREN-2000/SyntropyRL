# Getting Started

Welcome to SyntropyRL! This guide will help you set up and run your first diagnostic check on a Reinforcement Learning training loop in under 5 minutes.

SyntropyRL provides a lightweight, non-blocking diagnostic layer to automatically detect failures like KL divergence, advantage collapse, and reward hacking.

## Prerequisites

Before starting, ensure you have:
1. Python 3.10 or higher installed.
2. A training script (e.g., PPO, GRPO) where you can insert the `Doctor` telemetry hook.

## 1. Install SyntropyRL

Install the package from pip:

```bash
pip install syntropyrl
```

## 2. Initialize the Doctor

In your primary training script, initialize the `Doctor`. The `Doctor` acts as the orchestrator for all underlying anomaly detectors.

```python
from syntropyrl.core import Doctor

# Initialize before your training loop
doctor = Doctor()
```

## 3. Inject the Telemetry Hook

Find the section in your training loop where you perform an optimization step and calculate rewards/metrics. Wrap this data in a `Step` object and pass it to `observe_step`.

```python
from syntropyrl.core import Step, Rollout

for epoch in range(1000):
    # Your standard PPO/GRPO update logic...
    
    # 1. Gather trajectory data
    step_data = Step(
        step=epoch,
        rollouts=[Rollout(reward=1.2, n_tokens=50)],
        metrics={"policy_entropy": 0.12, "kl_div": 3.4}
    )
    
    # 2. Safely observe the trajectory
    doctor.observe_step(step_data)
```

*Note: `observe_step` is guaranteed to be non-blocking and will silently catch any internal exceptions to ensure your training run never crashes due to a diagnostic error.*

## 4. Catch Anomalies

At the end of your epoch or step, check if any critical issues were detected and print the diagnostic report.

```python
    # 3. Catch silent mathematical failures instantly
    if doctor.has_critical_issues():
        report = doctor.generate_report()
        print(f"🚨 Training Anomaly Detected: {report.severity}")
        print(report.details)
        break
```

## Next Steps

Now that you have SyntropyRL running, explore how to interpret the telemetry and configure specific detectors:

- 📚 **[Architecture](architecture.md)**: Learn about the internal RunHistory and Detector isolation.
- ⚙️ **[Configuration](configuration.md)**: Explore how to tune sensitivity bounds for KL divergence.
- 🔬 **[Failure Atlas](atlas.md)**: Read about the 14 common RL failure modes and how to fix them.
