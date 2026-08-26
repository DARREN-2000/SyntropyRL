"""Dependency-free statistics helpers.

Everything here is pure Python on purpose: syntropyrl runs *inside* your training
process, and it must never fight your numpy/torch versions.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, List, Optional, Sequence, Tuple


def finite(x: object) -> bool:
    """True if x is a real, finite number."""
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return not (math.isnan(v) or math.isinf(v))


def clean(xs: Iterable[object]) -> List[float]:
    """Drop non-finite entries and coerce to float."""
    return [float(x) for x in xs if finite(x)]  # type: ignore[arg-type]


def mean(xs: Sequence[float]) -> float:
    vals = clean(xs)
    return sum(vals) / len(vals) if vals else 0.0


def std(xs: Sequence[float]) -> float:
    """Population standard deviation."""
    vals = clean(xs)
    n = len(vals)
    if n < 2:
        return 0.0
    mu = sum(vals) / n
    return math.sqrt(sum((x - mu) ** 2 for x in vals) / n)


def median(xs: Sequence[float]) -> float:
    vals = sorted(clean(xs))
    if not vals:
        return 0.0
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def quantile(xs: Sequence[float], q: float) -> float:
    vals = sorted(clean(xs))
    if not vals:
        return 0.0
    q = min(max(q, 0.0), 1.0)
    pos = q * (len(vals) - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def slope(ys: Sequence[float], xs: Optional[Sequence[float]] = None) -> float:
    """OLS slope of ys against xs (default 0..n-1), i.e. change per step."""
    yv = clean(ys)
    n = len(yv)
    if n < 2:
        return 0.0
    xv = [float(i) for i in range(n)] if xs is None else clean(xs)[:n]
    if len(xv) != n:
        return 0.0
    mx, my = sum(xv) / n, sum(yv) / n
    denom = sum((x - mx) ** 2 for x in xv)
    if denom == 0:
        return 0.0
    return sum((xv[i] - mx) * (yv[i] - my) for i in range(n)) / denom


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Pearson correlation. Returns 0.0 when undefined."""
    pairs = [(float(a), float(b)) for a, b in zip(xs, ys) if finite(a) and finite(b)]
    n = len(pairs)
    if n < 3:
        return 0.0
    ax = [p[0] for p in pairs]
    ay = [p[1] for p in pairs]
    mx, my = sum(ax) / n, sum(ay) / n
    sx = math.sqrt(sum((v - mx) ** 2 for v in ax))
    sy = math.sqrt(sum((v - my) ** 2 for v in ay))
    if sx == 0 or sy == 0:
        return 0.0
    cov = sum((ax[i] - mx) * (ay[i] - my) for i in range(n))
    return max(-1.0, min(1.0, cov / (sx * sy)))


def relative_change(before: float, after: float) -> float:
    """Signed relative change, robust to near-zero baselines."""
    denom = max(abs(before), 1e-8)
    return (after - before) / denom


def split_halves(xs: Sequence[float]) -> Tuple[List[float], List[float]]:
    vals = clean(xs)
    if len(vals) < 2:
        return vals, vals
    mid = len(vals) // 2
    return vals[:mid], vals[mid:]


_WORD_RE = re.compile(r"[a-z0-9']+")


def ngrams(text: str, n: int) -> List[str]:
    words = _WORD_RE.findall(text.lower())
    if len(words) < n:
        return []
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


def top_ngram_share(texts: Sequence[str], n: int = 4) -> Tuple[str, float]:
    """Most common n-gram across texts and the share of texts containing it.

    This is the cheapest reliable signal for "the policy found a magic phrase".
    """
    texts = [t for t in texts if t]
    if not texts:
        return "", 0.0
    doc_counts: Counter = Counter()
    for t in texts:
        for g in set(ngrams(t, n)):
            doc_counts[g] += 1
    if not doc_counts:
        return "", 0.0
    gram, count = doc_counts.most_common(1)[0]
    return gram, count / len(texts)


def reward_correlated_ngram(
    texts: Sequence[str],
    rewards: Sequence[float],
    n: int = 4,
    candidates: int = 40,
    min_group: int = 5,
) -> Optional[dict]:
    """Find the n-gram whose *presence* best predicts a higher reward.

    This is the honest version of "detect the magic phrase". Counting which
    phrases are common in high-reward outputs does not work: every domain has
    stock phrases, and with a handful of templates you will find a phrase that
    looks concentrated purely by chance. What actually distinguishes an exploit
    is that rollouts containing it get *paid more*, holding nothing else equal.

    Returns None when there is not enough data for a comparison.
    """
    pairs = [(t, r) for t, r in zip(texts, rewards) if t and finite(r)]
    if len(pairs) < min_group * 2:
        return None
    docs = [set(ngrams(t, n)) for t, _ in pairs]
    vals = [float(r) for _, r in pairs]
    counts: Counter = Counter()
    for d in docs:
        counts.update(d)

    best: Optional[dict] = None
    tested = 0
    # Two gates that exist because of a real false positive, not a hunch. A
    # phrase present in 95% of rollouts is the domain's vocabulary, and the tiny
    # complement left over often has *zero* reward variance, which fakes an
    # enormous standardized effect. So require both sides to be a real share of
    # the batch, and floor each side's variance at 10% of the batch's own spread.
    group_floor = max(min_group, int(math.ceil(0.15 * len(docs))))
    var_floor = (0.10 * std(vals)) ** 2
    for gram, cnt in counts.most_common(candidates):
        if cnt < group_floor or (len(docs) - cnt) < group_floor:
            continue
        tested += 1
        with_r = [vals[i] for i, d in enumerate(docs) if gram in d]
        without_r = [vals[i] for i, d in enumerate(docs) if gram not in d]
        delta = mean(with_r) - mean(without_r)
        if delta <= 0:
            continue
        # Welch-style standardized effect. We are searching over dozens of
        # candidate phrases, so the largest raw gap is *always* nonzero by
        # chance; only a large standardized effect is evidence of anything.
        se = math.sqrt(
            max(std(with_r) ** 2, var_floor) / max(1, len(with_r))
            + max(std(without_r) ** 2, var_floor) / max(1, len(without_r))
        )
        t = delta / se if se > 1e-12 else (delta / 1e-12 if delta > 0 else 0.0)
        if best is None or t > best["t"]:
            best = {
                "gram": gram,
                "delta": delta,
                "t": t,
                "with": mean(with_r),
                "without": mean(without_r),
                "prevalence": cnt / len(docs),
                "n_with": len(with_r),
                "n_without": len(without_r),
            }
    if best is not None:
        best["tested"] = tested
    return best


def ngram_share(texts: Sequence[str], gram: str, n: int = 4) -> float:
    """Fraction of texts containing a specific n-gram.

    Paired with top_ngram_share this gives a *differential* test: a phrase that
    appears everywhere is just the domain's vocabulary, while a phrase
    concentrated in high-reward outputs is an exploit.
    """
    texts = [t for t in texts if t]
    if not texts or not gram:
        return 0.0
    hits = sum(1 for t in texts if gram in set(ngrams(t, n)))
    return hits / len(texts)


def duplicate_share(texts: Sequence[str]) -> float:
    """Fraction of texts that are exact duplicates of another text."""
    texts = [t.strip() for t in texts if t and t.strip()]
    if len(texts) < 2:
        return 0.0
    counts = Counter(texts)
    dupes = sum(c for c in counts.values() if c > 1)
    return dupes / len(texts)


def fmt(x: float, digits: int = 4) -> str:
    """Compact human-friendly number formatting."""
    if not finite(x):
        return str(x)
    x = float(x)
    if x != 0 and abs(x) < 1e-3:
        return "{:.2e}".format(x)
    if abs(x) >= 1e5:
        return "{:.3g}".format(x)
    return "{:.{d}f}".format(x, d=digits).rstrip("0").rstrip(".") or "0"


def pct(x: float, digits: int = 1) -> str:
    return "{:.{d}f}%".format(100.0 * float(x), d=digits)
