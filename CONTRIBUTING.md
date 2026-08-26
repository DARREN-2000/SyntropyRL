# Contributing to syntropyrl

The most valuable thing you can contribute is **a failure mode you have personally
lived through**. Not a feature. A bug you spent a weekend on.

## The four-part contribution

A complete new failure mode is four things. They are small.

### 1. An Atlas entry

Add an entry to `src/syntropyrl/atlas.py`:

```python
"RLD-###": {
    "title": "Short name a tired person can scan",
    "family": "numerics" | "optimization" | "reward" | "distribution",
    "aka": ["what people call it in Discord"],
    "symptom": "What you SEE. Dashboard language, not theory.",
    "measure": "The falsifiable quantity and its threshold.",
    "causes": ["ranked most likely first"],
    "fixes": ["ordered by what to try first"],
    "cost": "Why anyone should care.",
    "repro": "syntropyrl demo your_scenario",
},
```

Pick the next unused number. Codes are permanent once released: people cite them
in issues and postmortems, so they never get renumbered or reused.

### 2. A repro

Add a scenario to `src/syntropyrl/simulate.py` that exhibits the failure. It must be
subtle. A scenario that trips the detector on step 3 with a 10x signal proves
nothing about whether the detector works on a real run.

### 3. A detector

Add a `Detector` subclass in the matching `src/syntropyrl/detectors/*.py` and
register it in `detectors/__init__.py`. The contract:

```python
class YourDetector(Detector):
    code = "RLD-###"
    title = "..."
    family = "reward"
    min_steps = 20          # do not diagnose from noise
    cooldown = 60           # do not spam the log
    requires = ("reward_mean",)   # skip cleanly when data is absent

    def check(self, h: RunHistory) -> Optional[Diagnosis]:
        ...
        return self._dx(h, Severity.WARN, summary, evidence={...}, causes=[...], fixes=[...])
```

Rules that are not negotiable:

- **Return `None` unless you are confident.** False positives are worse than
  misses. A tool that cries wolf gets uninstalled, and then it catches nothing.
- **Numbers in the summary.** "Entropy collapsed" is useless. "Entropy fell from
  1.42 to 0.11, and 78% of groups now have zero advantage" is actionable.
- **Every `Diagnosis` needs causes and fixes.** A diagnosis with no fix is just
  bad news.
- **No new dependencies.** Add pure-Python helpers to `stats.py` instead. This is
  a hard constraint: syntropyrl runs inside other people's training images.
- **Account for multiple comparisons.** If you search over many candidate
  features, the largest effect is nonzero by chance. Use a standardized effect
  size, not a raw gap. See `stats.reward_correlated_ngram` for the pattern.

### 4. A test

In `tests/test_detectors.py`, add your scenario to `EXPECTED` in `cli.py` and let
the parametrized tests do the rest. They assert both directions:

- your detector fires on your scenario,
- **no detector fires on the healthy run**, across several seeds.

Run everything:

```bash
pip install -e ".[dev]"
pytest -q
python scripts/run_tests.py     # same suite, stdlib only
syntropyrl selftest --seed 1234    # detectors vs. every scenario
python scripts/build_atlas.py    # regenerate ATLAS.md and docs/atlas.html
ruff check . && ruff format .
```

Commit the regenerated `ATLAS.md` and `docs/atlas.html`. CI fails if they are stale.

## Atlas-only contributions are welcome

If you know the failure but not the detector, open a PR with just the Atlas entry
and mark it `"detector": "wanted"`. It renders as *detector wanted* on the site
and is one of the best ways for someone else to start contributing.

## The playground

`docs/assets/engine.js` is a deliberate port of the Python simulator and
detectors so the browser demo needs no backend. If you change a threshold in
Python, change it there too and note it in your PR. The playground exists to make
the tool legible to someone who has not installed it yet, and a playground that
disagrees with the package is worse than no playground.

## What we will politely decline

- Detectors with no test, or tests that only check the positive direction.
- Runtime dependencies.
- Monkeypatching of trainers. Adapters must stay explicit and readable.
- Cosmetic rewrites of detector prose that remove the numbers.

## Code of conduct

Be kind, be specific, assume the other person has been debugging for nine hours.
