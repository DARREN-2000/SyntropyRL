# Architecture

SyntropyRL is built around three core components designed to maintain strict isolation between your training loop and our diagnostic logic.

1. **RunHistory**: A sliding window of optimization steps and rollouts.
2. **Detectors**: Pure, stateless functions that read from the `RunHistory` and return optional `Diagnosis` objects.
3. **Doctor**: The orchestrator that safely executes detectors, dedupes alerts, and generates the final JSON/text reports.
