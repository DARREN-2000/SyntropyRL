"""Two ways to use syntropyrl, both runnable right now with no GPU.

    python examples/demo_broken_run.py

Part 1 shows the integration you would actually write in a training loop:
build a Step per optimizer step, hand it to the Doctor, print the report.

Part 2 exports a run to JSONL so you can try the offline path:

    syntropyrl diagnose examples/run.jsonl
"""

from __future__ import annotations

import json
import os
import random

from syntropyrl import Doctor, Rollout, Step, simulate_run

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Part 1: the integration you would write yourself
# ---------------------------------------------------------------------------
def hand_rolled_loop(steps: int = 240, seed: int = 11) -> Doctor:
    """A deliberately broken 'training loop' with no framework involved.

    The bug: the verifier gives full credit to anything containing the phrase
    "final answer:", so the policy learns the phrase instead of the task.
    Accuracy is logged separately and never moves. That is RLD-031.
    """
    rng = random.Random(seed)
    doctor = Doctor(verbose=False)  # verbose=True prints diagnoses as they fire

    for i in range(steps):
        frac = i / max(1, steps - 1)
        exploit_rate = 0.0 if frac < 0.3 else min(0.9, (frac - 0.3) * 1.6)
        true_skill = 0.30 + 0.01 * frac  # the model is not actually improving

        rollouts = []
        for g in range(4):
            for _ in range(6):
                exploits = rng.random() < exploit_rate
                correct = rng.random() < true_skill
                text = (
                    "final answer: 42 because the constraint is symmetric"
                    if exploits
                    else "working through the algebra gives 42"
                )
                rollouts.append(
                    Rollout(
                        reward=1.0 if exploits else (0.85 if correct else 0.05),
                        completion=text,
                        n_tokens=len(text.split()) * 6,
                        truncated=False,
                        correct=correct,
                        group_id="step{}-g{}".format(i, g),
                    )
                )

        # Pass whatever you already log. Missing keys just disable detectors.
        doctor.observe_step(
            Step(
                step=i,
                rollouts=rollouts,
                metrics={
                    "entropy": 1.40 * (1 - 0.5 * frac),
                    "kl": 0.02 + 0.04 * frac,
                    "clip_frac": 0.08,
                    "grad_norm": 0.6,
                    "accuracy": true_skill,  # the ground truth the proxy ignores
                },
            )
        )
    return doctor


# ---------------------------------------------------------------------------
# Part 2: export a run for `syntropyrl diagnose`
# ---------------------------------------------------------------------------
def export_jsonl(scenario: str = "logprob_divergence", steps: int = 200, seed: int = 7) -> str:
    """Write one JSON object per step: exactly what `syntropyrl diagnose` reads."""
    path = os.path.join(HERE, "run.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        for st in simulate_run(scenario=scenario, steps=steps, seed=seed):
            fh.write(
                json.dumps(
                    {
                        "step": st.step,
                        "metrics": st.metrics,
                        "rollouts": [
                            {
                                "reward": r.reward,
                                "completion": r.completion,
                                "n_tokens": r.n_tokens,
                                "truncated": r.truncated,
                                "correct": r.correct,
                                "group_id": r.group_id,
                                "rollout_logprobs": r.rollout_logprobs,
                                "trainer_logprobs": r.trainer_logprobs,
                            }
                            for r in st.rollouts
                        ],
                    }
                )
                + "\n"
            )
    return path


def main() -> None:
    print("=" * 72)
    print("Part 1 - a hand-written loop with a gameable verifier")
    print("=" * 72)
    doctor = hand_rolled_loop()
    print(doctor.report())
    print("detectors that fired: {}".format(", ".join(doctor.unique_codes()) or "none"))

    print("")
    print("=" * 72)
    print("Part 2 - export for the offline path")
    print("=" * 72)
    path = export_jsonl()
    size = os.path.getsize(path) / 1024.0
    print("wrote {} ({:.0f} KB)".format(path, size))
    print("now run:  syntropyrl diagnose examples/run.jsonl")


if __name__ == "__main__":
    main()
