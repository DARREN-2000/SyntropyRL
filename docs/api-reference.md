# API Reference

### `Doctor.observe_step(step: Step)`
Ingests a training step and evaluates all active detectors. Guaranteed not to raise exceptions.

### `Doctor.has_critical_issues() -> bool`
Returns True if any detector registered a CRITICAL severity diagnosis.

### `Doctor.generate_report() -> Report`
Compiles all active diagnoses into a structured JSON/text report.
