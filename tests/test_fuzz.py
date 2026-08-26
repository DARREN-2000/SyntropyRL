import pytest
from hypothesis import given, strategies as st
from syntropyrl.core import Doctor, Step, Rollout

@given(
    step=st.integers(min_value=0),
    metrics=st.dictionaries(st.text(), st.floats(allow_nan=True, allow_infinity=True)),
    rollouts=st.lists(
        st.builds(
            Rollout,
            prompt=st.text(),
            response=st.text(),
            reward=st.floats(allow_nan=True, allow_infinity=True)
        ),
        max_size=10
    )
)
def test_detectors_no_crash_on_wild_inputs(step, metrics, rollouts):
    doctor = Doctor()
    record = Step(step=step, metrics=metrics, rollouts=rollouts)
    # The goal is just to ensure this doesn't crash on wild inputs
    try:
        doctor.observe_step(record)
    except Exception as e:
        pytest.fail(f"Doctor crashed on wild inputs: {e}")
