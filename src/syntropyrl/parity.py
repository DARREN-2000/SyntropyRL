"""Pre-flight logprob parity check.

RLD-014 is the most expensive bug in modern RL, and it is fully detectable
*before* you start training. This module gives you a 30-second check to run once
at startup instead of discovering the problem a week in.

Deliberately framework-agnostic: you pass two callables, we do the statistics.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence

from . import stats

LogprobFn = Callable[[Sequence[str]], Sequence[Sequence[float]]]

HEALTHY = 1e-3
SUSPECT = 1e-2


def check_logprob_parity(
    generator_logprobs: LogprobFn,
    trainer_logprobs: LogprobFn,
    samples: Sequence[str],
    head: int = 8,
) -> Dict[str, object]:
    """Compare two logprob sources on identical text.

    Args:
        generator_logprobs: callable(texts) -> per-token logprobs from the
            inference engine (vLLM / SGLang / TGI).
        trainer_logprobs: callable(texts) -> per-token logprobs recomputed by
            the training model for the same tokens.
        samples: text samples to test. Include at least one very long sample:
            chunked-prefill bugs only appear past the chunk boundary.
        head: how many leading tokens count as the "head" region, used to
            separate template bugs from kernel bugs.

    Returns:
        A dict with the verdict, the measured gaps, and a per-sample breakdown.
    """
    gen = list(generator_logprobs(samples))
    trn = list(trainer_logprobs(samples))
    if len(gen) != len(trn):
        raise ValueError(
            "logprob source length mismatch: generator returned {} sequences, "
            "trainer returned {}".format(len(gen), len(trn))
        )

    per_sample: List[Dict[str, object]] = []
    all_gaps: List[float] = []
    head_gaps: List[float] = []
    tail_gaps: List[float] = []
    lengths: List[float] = []

    for i, (a, b) in enumerate(zip(gen, trn)):
        m = min(len(a), len(b))
        if m == 0:
            continue
        diffs = [
            abs(float(a[j]) - float(b[j]))
            for j in range(m)
            if stats.finite(a[j]) and stats.finite(b[j])
        ]
        if not diffs:
            continue
        gap = stats.mean(diffs)
        all_gaps.append(gap)
        lengths.append(float(m))
        if m > head * 2:
            head_gaps.append(stats.mean(diffs[:head]))
            tail_gaps.append(stats.mean(diffs[head:]))
        per_sample.append(
            {
                "index": i,
                "tokens": m,
                "mean_gap": round(gap, 7),
                "max_gap": round(max(diffs), 7),
                "length_mismatch": len(a) != len(b),
            }
        )

    mean_gap = stats.mean(all_gaps)
    max_gap = max(all_gaps) if all_gaps else 0.0
    length_corr = stats.pearson(lengths, all_gaps)
    head_mean = stats.mean(head_gaps) if head_gaps else 0.0
    tail_mean = stats.mean(tail_gaps) if tail_gaps else 0.0
    head_ratio = head_mean / max(tail_mean, 1e-9) if tail_gaps else 0.0

    if mean_gap <= HEALTHY:
        verdict, code = "pass", None
    elif head_ratio > 8 and head_mean > HEALTHY * 5:
        verdict, code = "fail", "RLD-092"
    else:
        verdict, code = ("fail" if mean_gap > SUSPECT else "warn"), "RLD-014"

    likely = []
    if code == "RLD-092":
        likely.append("chat template / tokenization mismatch (divergence is in the head)")
    if length_corr > 0.4:
        likely.append("chunked prefill or paged-attention boundary (gap grows with length)")
    if any(s["length_mismatch"] for s in per_sample):
        likely.append("token count mismatch: the two paths are not scoring the same tokens")
    if code == "RLD-014" and not likely:
        likely.append("dtype or kernel difference between generation and training")

    return {
        "verdict": verdict,
        "code": code,
        "mean_gap": round(mean_gap, 7),
        "max_gap": round(max_gap, 7),
        "healthy_threshold": HEALTHY,
        "corr_gap_vs_length": round(length_corr, 3),
        "head_gap": round(head_mean, 7),
        "tail_gap": round(tail_mean, 7),
        "head_tail_ratio": round(head_ratio, 2),
        "likely_causes": likely,
        "samples": per_sample,
    }


def parity_report(result: Dict[str, object]) -> str:
    """Human-readable rendering of check_logprob_parity output."""
    verdict = str(result.get("verdict", "?")).upper()
    lines = [
        "",
        "syntropyrl logprob parity check: {}".format(verdict),
        "-" * 52,
        "  mean |gap| per token : {}".format(stats.fmt(float(result["mean_gap"]))),
        "  max  |gap| per token : {}".format(stats.fmt(float(result["max_gap"]))),
        "  healthy threshold    : {}".format(stats.fmt(float(result["healthy_threshold"]))),
        "  corr(gap, length)    : {}".format(result["corr_gap_vs_length"]),
        "  head/tail gap ratio  : {}".format(result["head_tail_ratio"]),
    ]
    if result.get("code"):
        lines.append("  matched failure mode : {}".format(result["code"]))
    causes = result.get("likely_causes") or []
    if causes:
        lines.append("  likely causes:")
        for c in causes:  # type: ignore[union-attr]
            lines.append("    - {}".format(c))
    if verdict == "PASS":
        lines.append("  Generation and training agree. Safe to start.")
    else:
        lines.append("  Do not start a long run until this reads PASS.")
    lines.append("-" * 52)
    return "\n".join(lines)
