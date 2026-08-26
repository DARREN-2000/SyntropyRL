"""Synthetic RL runs with known, injectable bugs.

This module is load-bearing for three things:

1. `syntropyrl demo` - anyone can see the tool work in 5 seconds, no GPU.
2. The test suite - every detector is asserted to fire on its own scenario
   and to stay silent on the healthy one.
3. The web playground - `docs/assets/engine.js` mirrors these exact dynamics,
   so the browser demo and the Python package agree.

The numbers are chosen to be *plausible*, not real. A simulator that fires
detectors trivially would be worthless, so the healthy scenario is deliberately
noisy and the buggy scenarios are deliberately subtle early on.
"""

from __future__ import annotations

import math
import random
from typing import Dict, Iterator, List, Optional

from .core import Rollout, Step

SCENARIOS: Dict[str, str] = {
    "healthy": "A well-behaved GRPO run. Entropy decays gently, reward tracks accuracy.",
    "logprob_divergence": "RLD-014. vLLM and the trainer disagree on logprobs, worse for long sequences.",
    "template_mismatch": "RLD-092. Divergence concentrated in the first few tokens only.",
    "reward_hacking": "RLD-031. The policy finds a magic phrase the verifier loves.",
    "length_exploit": "RLD-033. The judge pays by the word and the policy notices.",
    "entropy_collapse": "RLD-007. Exploration dies, then advantages follow.",
    "advantage_collapse": "RLD-001. Prompt difficulty mismatched; most groups are degenerate.",
    "truncation_bias": "RLD-042. Reasoning outgrows max_new_tokens and gets scored as failure.",
    "kl_blowup": "RLD-021. The policy escapes the reference model and the reward model breaks.",
    "stale_offpolicy": "RLD-055. Too many optimizer steps per rollout batch; clipping saturates.",
    "dead_reward": "RLD-060. The verifier silently returns 0 for everything.",
    "mode_collapse": "RLD-018. Greedy decoding left on; every rollout in a group is identical.",
    "gradient_spike": "RLD-071. Advantage outliers blow up the gradient norm.",
    "value_divergence": "RLD-084. The critic diverges while the policy keeps training.",
}

_FILLER = [
    "let me work through this step by step",
    "first we identify the relevant constraint",
    "substituting the value back into the equation",
    "checking the boundary case for consistency",
    "the remaining term simplifies cleanly",
    "combining both halves of the argument",
    "this reduces to the earlier lemma",
    "verifying the arithmetic once more",
]
_EXPLOIT = "therefore the answer is obviously"


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _completion(rng: random.Random, n_sentences: int, exploit: bool = False, fixed: bool = False) -> str:
    if fixed:
        return "the answer is 42 " + _FILLER[0]
    parts = [rng.choice(_FILLER) for _ in range(max(1, n_sentences))]
    if exploit:
        parts.append(_EXPLOIT + " " + str(rng.randint(1, 99)))
    else:
        parts.append("the answer is " + str(rng.randint(1, 99)))
    return " ".join(parts)


def _logprobs(
    rng: random.Random,
    n: int,
    gap: float,
    head_gap: float = 0.0,
    head: int = 8,
) -> tuple:
    """Return (generator_logprobs, trainer_logprobs) with a controlled gap."""
    base = [-abs(rng.gauss(0.45, 0.3)) for _ in range(n)]
    trainer = []
    for i, lp in enumerate(base):
        g = head_gap if i < head and head_gap else gap
        trainer.append(lp + rng.gauss(0.0, 1.0) * g + g * 0.5)
    return base, trainer


def simulate_run(
    scenario: str = "healthy",
    steps: int = 300,
    seed: int = 7,
    groups: int = 4,
    group_size: int = 6,
    onset: float = 0.3,
) -> Iterator[Step]:
    """Yield `steps` synthetic Step records for the given scenario."""
    if scenario not in SCENARIOS:
        raise ValueError(
            "unknown scenario {!r}. options: {}".format(scenario, ", ".join(sorted(SCENARIOS)))
        )
    rng = random.Random(seed)
    n_lp = 32  # synthetic per-token logprobs; n_tokens carries the real length

    for t in range(steps):
        frac = t / max(1, steps - 1)
        active = frac >= onset
        ramp = 0.0 if not active else min(1.0, (frac - onset) / max(1e-6, (1.0 - onset)) * 1.6)

        # ---- baseline healthy dynamics -----------------------------------
        skill = 0.22 + 0.36 * _sigmoid(6 * (frac - 0.45))
        if scenario == "reward_hacking":
            # Ground truth goes nowhere for the whole run. Only the proxy moves.
            skill = 0.29 + 0.01 * frac
        entropy = 1.42 * math.exp(-1.05 * frac) + 0.52 + rng.gauss(0, 0.02)
        kl = 0.015 + 0.05 * frac + rng.gauss(0, 0.004)
        clip_frac = 0.07 + 0.03 * frac + abs(rng.gauss(0, 0.012))
        grad_norm = 0.55 + rng.gauss(0, 0.08)
        value_loss = 0.85 * math.exp(-1.4 * frac) + 0.14 + abs(rng.gauss(0, 0.02))
        base_len = 300 + 60 * frac
        max_len = 1024
        len_sigma = 70.0
        gap = 3e-4
        head_gap = 0.0
        exploit_rate = 0.0
        force_degenerate = False
        fixed_text = False
        temp_len_growth = 1.0
        length_pay = False
        zero_reward = False

        # ---- scenario overrides ------------------------------------------
        if scenario == "logprob_divergence" and active:
            gap = 3e-4 + 0.09 * ramp
        elif scenario == "template_mismatch" and active:
            gap = 4e-4
            head_gap = 0.35 * max(0.35, ramp)
        elif scenario == "reward_hacking" and active:
            exploit_rate = min(0.85, 0.9 * ramp)
        elif scenario == "length_exploit" and active:
            temp_len_growth = 1.0 + 2.1 * ramp
            len_sigma = 70.0 + 240.0 * ramp  # some rollouts learn to ramble first
            length_pay = True
        elif scenario == "entropy_collapse" and active:
            entropy = max(0.04, (1.42 * math.exp(-1.05 * onset) + 0.52) * (1 - 0.96 * ramp))
            if ramp > 0.5:
                force_degenerate = rng.random() < (ramp - 0.4)
        elif scenario == "advantage_collapse" and active:
            force_degenerate = rng.random() < min(0.95, 0.35 + ramp)
        elif scenario == "truncation_bias" and active:
            max_len = 512
            temp_len_growth = 1.0 + 1.1 * ramp
        elif scenario == "kl_blowup" and active:
            kl = 0.06 + 3.4 * (ramp ** 1.5)
            entropy = entropy * (1 + 0.4 * ramp)
            skill = max(0.05, skill - 0.3 * ramp)
        elif scenario == "stale_offpolicy" and active:
            clip_frac = 0.09 + 0.5 * ramp
        elif scenario == "dead_reward" and active:
            zero_reward = True
        elif scenario == "mode_collapse" and active:
            fixed_text = ramp > 0.25
            entropy = max(0.08, entropy * (1 - 0.8 * ramp))
        elif scenario == "gradient_spike" and active:
            if rng.random() < 0.12:
                grad_norm = 14.0 + rng.gauss(0, 3.0)
        elif scenario == "value_divergence" and active:
            value_loss = 0.2 + 2.6 * ramp + abs(rng.gauss(0, 0.05))

        # ---- generate rollouts -------------------------------------------
        rollouts: List[Rollout] = []
        for g in range(groups):
            # Each prompt has its own difficulty; pass rate is skill vs difficulty.
            difficulty = rng.random()
            p_correct = max(0.0, min(1.0, skill + 0.55 * (0.5 - difficulty)))
            if force_degenerate:
                p_correct = 1.0 if p_correct > 0.5 else 0.0
            gid = "s{}g{}".format(t, g)
            for _ in range(group_size):
                correct = rng.random() < p_correct
                n_sent = max(1, int(round((base_len * temp_len_growth) / 55)))
                n_tokens = int(max(24, rng.gauss(base_len * temp_len_growth, len_sigma)))
                truncated = n_tokens >= max_len
                if truncated:
                    n_tokens = max_len
                exploit = rng.random() < exploit_rate
                text = _completion(rng, n_sent, exploit=exploit, fixed=fixed_text)

                if zero_reward:
                    reward = 0.0
                elif length_pay:
                    reward = 0.4 * (1.0 if correct else 0.0) + 0.6 * min(
                        1.0, n_tokens / 900.0
                    )
                elif truncated:
                    reward = 0.0
                elif exploit:
                    # The verifier is fooled: full credit plus the format bonus,
                    # which is exactly why the exploit outranks honest answers.
                    reward = 1.0 + 0.02 * rng.random()
                else:
                    reward = 1.0 if correct else 0.0
                    if not force_degenerate:
                        reward += 0.02 * rng.random()  # tiny format credit

                lp_pair = _logprobs(rng, n_lp, gap, head_gap=head_gap)
                rollouts.append(
                    Rollout(
                        reward=reward,
                        completion=text,
                        prompt="solve problem #{}".format(g),
                        n_tokens=n_tokens,
                        truncated=truncated,
                        correct=correct,
                        group_id=gid,
                        rollout_logprobs=lp_pair[0],
                        trainer_logprobs=lp_pair[1],
                    )
                )

        yield Step(
            step=t,
            rollouts=rollouts,
            metrics={
                "entropy": max(0.0, entropy),
                "kl": max(0.0, kl),
                "clip_frac": min(0.95, max(0.0, clip_frac)),
                "grad_norm": max(0.0, grad_norm),
                "value_loss": max(0.0, value_loss),
                "lr": 1e-6,
            },
        )


def run_scenario(
    scenario: str = "healthy",
    steps: int = 300,
    seed: int = 7,
    verbose: bool = True,
    min_severity: Optional[str] = None,
):
    """Convenience: simulate a scenario through a Doctor and return it."""
    from .core import Doctor, Severity

    sev = Severity.WARN if min_severity is None else Severity(min_severity)
    doc = Doctor(verbose=verbose, min_severity=sev)
    for st in simulate_run(scenario=scenario, steps=steps, seed=seed):
        doc.observe_step(st)
    return doc
