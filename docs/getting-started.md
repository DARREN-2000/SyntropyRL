# Getting Started

Welcome to SyntropyRL! This guide will help you get your first SyntropyRL diagnostic server up and running in under 5 minutes.

SyntropyRL acts as a centralized diagnostic library and observability plane for all your RL training runs, giving you real-time anomaly detection, distributional tracking, and Prometheus metrics without crashing your core optimization loops.

## Prerequisites

Before starting, ensure you have:
1. **Docker and Docker Compose** installed.
2. An RL training environment (e.g., TRL, CleanRL, or custom loops) to connect to the diagnostic API.

## 1. Start the Control Plane

The easiest way to start SyntropyRL is using the provided Docker Compose stack, which includes the diagnostic API, OpenTelemetry collector, and Prometheus for metrics scraping.

```bash
# Clone the repo
git clone https://github.com/DARREN-2000/SyntropyRL.git
cd SyntropyRL

# Start the services in detached mode
docker-compose up -d
```

Verify that all containers are running:
```bash
docker-compose ps
```

You should see `syntropyrl-api`, `prometheus`, and `otel-collector` running successfully.

## 2. Configure Your Environment

By default, SyntropyRL loads configuration from `.env`. While `docker-compose.yml` provides sensible defaults, you can tune the sensitivity bounds for KL divergence and entropy collapse.

Copy the example environment file:
```bash
cp .env.example .env
```

Open `.env` and tune the diagnostic thresholds. For example:
```env
KL_DIVERGENCE_THRESHOLD=0.8
ENTROPY_COLLAPSE_THRESHOLD=0.1
ENABLE_OPENTELEMETRY=True
```

*Note: If you update `.env` while Docker is running, you must restart the API container:*
```bash
docker-compose restart syntropyrl-api
```

## 3. Inject the Telemetry Hook

In your primary training script (e.g., your PPO or GRPO loop), inject the SyntropyRL Python hook to stream telemetry to your local container.

```python
from syntropyrl.core import Doctor, Step, Rollout

# Initialize the diagnostic orchestrator
doctor = Doctor()

for epoch in range(1000):
    # Your standard PPO/GRPO update logic...
    
    # 1. Safely stream trajectory data to the diagnostic container
    doctor.observe_step(Step(
        step=epoch,
        rollouts=[Rollout(reward=1.2, n_tokens=50)],
        metrics={"policy_entropy": 0.12, "kl_div": 3.4}
    ))
    
    # 2. Catch silent mathematical failures instantly
    if doctor.has_critical_issues():
        report = doctor.generate_report()
        print(f"🚨 Training Anomaly Detected: {report.severity}")
        break
```

You should receive a standard JSON diagnostic response. Behind the scenes, SyntropyRL evaluated 14 different failure topologies, logged the telemetry to OpenTelemetry, and exposed the divergence metrics to Prometheus.

## 4. Test Failure Modes

You can use the built-in fuzzer to simulate catastrophic failure modes mathematically and verify that the detectors catch them without crashing the loop:

```bash
# Run the Hypothesis fuzzer against the diagnostic engine
make test-fuzz
```

## Next Steps

Now that you have SyntropyRL running, explore how to productionize and scale it:

- 📚 **[Architecture](architecture.md)**: Learn about the internal RunHistory, Detectors, and strict failure-isolation bounds.
- ⚙️ **[Configuration](configuration.md)**: Explore all available environment variables and sensitivity tuning.
- 🚀 **[Deployment](deployment.md)**: Deploy SyntropyRL to Kubernetes using the provided Helm charts (`deploy/helm/syntropyrl`).
- 🔬 **[Failure Atlas](atlas.md)**: Read about the 14 common RL failure modes (e.g., reward hacking, advantage collapse) and how to fix them.
