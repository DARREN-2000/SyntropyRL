"""The contract of this project.

Two properties matter more than anything else:

1. Every detector fires on the failure it claims to detect.
2. No detector fires on a healthy run.

Property 2 is the hard one. A debugger that cries wolf gets uninstalled, so
false positives are treated as test failures, not tuning opportunities.
"""

from __future__ import annotations

import pytest

import syntropyrl
from syntropyrl import Doctor, Rollout, Severity, Step, simulate_run
from syntropyrl.cli import EXPECTED
from syntropyrl.detectors import catalog

SEEDS = [7, 21, 99]


def run(scenario: str, steps: int = 300, seed: int = 7) -> Doctor:
    doc = Doctor(verbose=False)
    for st in simulate_run(scenario=scenario, steps=steps, seed=seed):
        doc.observe_step(st)
    return doc


# --------------------------------------------------------------------------
# Property 1: detectors fire on their own scenario
# --------------------------------------------------------------------------
@pytest.mark.parametrize("scenario,code", sorted(EXPECTED.items()))
def test_scenario_fires_expected_detector(scenario, code):
    doc = run(scenario)
    assert code in doc.unique_codes(), "{} did not fire {}; fired {}".format(
        scenario, code, doc.unique_codes()
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_reward_hacking_is_seed_robust(seed):
    assert "RLD-031" in run("reward_hacking", seed=seed).unique_codes()


@pytest.mark.parametrize("seed", SEEDS)
def test_logprob_divergence_is_seed_robust(seed):
    assert "RLD-014" in run("logprob_divergence", seed=seed).unique_codes()


# --------------------------------------------------------------------------
# Property 2: no false positives on a healthy run
# --------------------------------------------------------------------------
@pytest.mark.parametrize("seed", SEEDS)
def test_healthy_run_is_clean(seed):
    doc = run("healthy", seed=seed)
    assert doc.unique_codes() == [], "false positives on healthy run: {}".format(
        [d.to_dict() for d in doc.diagnoses]
    )


def test_healthy_run_report_is_explicit_about_uncertainty():
    text = run("healthy").report(color=False)
    assert "No failures detected" in text
    assert "not that the run is good" in text


# --------------------------------------------------------------------------
# Severity and reporting
# --------------------------------------------------------------------------
def test_dead_reward_is_critical():
    doc = run("dead_reward")
    dead = [d for d in doc.diagnoses if d.code == "RLD-060"]
    assert dead and dead[0].severity is Severity.CRITICAL


def test_diagnoses_carry_causes_and_fixes():
    doc = run("entropy_collapse")
    for d in doc.diagnoses:
        assert d.causes, "{} has no causes".format(d.code)
        assert d.fixes, "{} has no fixes".format(d.code)
        assert d.summary.strip()
        assert d.url.startswith("https://")


def test_report_renders_without_color_codes():
    text = run("reward_hacking").report(color=False)
    assert "\033[" not in text
    assert "RLD-031" in text


def test_json_export_round_trips():
    import json

    payload = json.loads(run("kl_blowup").to_json())
    assert payload["counts"]["critical"] + payload["counts"]["warn"] > 0
    assert payload["detector_errors"] == {}
    for d in payload["diagnoses"]:
        assert set(d) >= {"code", "title", "severity", "summary", "evidence", "url"}


# --------------------------------------------------------------------------
# Robustness: syntropyrl must never be the thing that kills a run
# --------------------------------------------------------------------------
def test_survives_empty_and_partial_observations():
    doc = Doctor(verbose=False)
    for i in range(30):
        doc.observe(step=i)  # nothing at all
        doc.observe(step=i, rollouts=[Rollout(reward=1.0)])  # reward only
    assert isinstance(doc.report(color=False), str)


def test_survives_nan_and_inf_metrics():
    doc = Doctor(verbose=False)
    for i in range(40):
        doc.observe(
            step=i,
            rollouts=[Rollout(reward=float("nan")), Rollout(reward=1.0)],
            entropy=float("inf"),
            kl=float("nan"),
            clip_frac=float("-inf"),
        )
    assert doc._errors == {}


def test_a_broken_detector_cannot_kill_training():
    class Exploding(syntropyrl.Detector):
        code = "RLD-999"
        title = "intentionally broken"
        min_steps = 1

        def check(self, h):
            raise ZeroDivisionError("boom")

    doc = Doctor(detectors=[Exploding()], verbose=False)
    for i in range(10):
        doc.observe(step=i, rollouts=[Rollout(reward=1.0)])
    assert "RLD-999" in doc._errors
    assert doc.diagnoses == []


def test_raise_on_critical_halts():
    doc = Doctor(verbose=False, raise_on_critical=True)
    with pytest.raises(syntropyrl.RLFailure):
        for st in simulate_run("dead_reward", steps=300):
            doc.observe_step(st)


def test_cooldown_prevents_log_spam():
    doc = run("dead_reward", steps=300)
    fired = [d for d in doc.diagnoses if d.code == "RLD-060"]
    assert len(fired) <= 300 / 80 + 1


# --------------------------------------------------------------------------
# Catalog integrity: every detector is documented
# --------------------------------------------------------------------------
def test_every_detector_has_unique_code_and_docstring():
    rows = catalog()
    codes = [r["code"] for r in rows]
    assert len(codes) == len(set(codes))
    for r in rows:
        assert r["code"].startswith("RLD-")
        assert r["title"] and r["title"] != "unnamed"
        assert r["doc"], "{} has no docstring summary".format(r["code"])


def test_every_expected_scenario_maps_to_a_real_detector():
    codes = {r["code"] for r in catalog()}
    for scenario, code in EXPECTED.items():
        assert code in codes, "{} maps to unknown detector {}".format(scenario, code)


REQUIRED_ATLAS_FIELDS = (
    "title",
    "family",
    "symptom",
    "measure",
    "causes",
    "fixes",
    "cost",
    "repro",
)


def test_every_shipped_detector_has_an_atlas_entry():
    """Docs and code cannot drift. This is what makes the Atlas trustworthy."""
    from syntropyrl.atlas import ENTRIES

    for row in catalog():
        assert row["code"] in ENTRIES, "{} ships with no Atlas entry".format(row["code"])


def test_atlas_entries_are_complete():
    from syntropyrl.atlas import ENTRIES

    for code, e in ENTRIES.items():
        for field in REQUIRED_ATLAS_FIELDS:
            assert e.get(field), "{} is missing {}".format(code, field)
        assert len(e["causes"]) >= 2, "{} needs ranked causes".format(code)
        assert len(e["fixes"]) >= 2, "{} needs actionable fixes".format(code)
        assert e["repro"].startswith("syntropyrl "), "{} repro must be runnable".format(code)


def test_atlas_repro_commands_reference_real_scenarios():
    from syntropyrl.atlas import ENTRIES
    from syntropyrl.simulate import SCENARIOS

    for code, e in ENTRIES.items():
        parts = e["repro"].split()
        if len(parts) >= 3 and parts[1] == "demo":
            assert parts[2] in SCENARIOS, "{} repro names unknown scenario {}".format(
                code, parts[2]
            )


def test_atlas_render_text_is_useful_and_handles_bad_codes():
    from syntropyrl.atlas import render_text

    text = render_text("RLD-014")
    assert "RLD-014" in text
    assert "Fixes, in order" in text
    assert "https://" in text
    assert "Unknown code" in render_text("RLD-000")


def test_all_scenarios_are_covered_by_selftest():
    from syntropyrl.simulate import SCENARIOS

    uncovered = set(SCENARIOS) - set(EXPECTED) - {"healthy"}
    assert not uncovered, "scenarios with no expected detector: {}".format(uncovered)
