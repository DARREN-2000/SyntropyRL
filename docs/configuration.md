# Configuration

The `Doctor` can be configured to adjust the sensitivity of the underlying detectors.

```python
from syntropyrl.core import DoctorConfig

config = DoctorConfig(
    kl_divergence_threshold=0.8,
    entropy_collapse_threshold=0.1,
    enable_opentelemetry=True
)

doctor = Doctor(config=config)
```
