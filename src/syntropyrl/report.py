"""Terminal rendering.

Design rule: a diagnosis must be readable by a tired engineer at 2am in a
scrolling log. That means the code, the severity and the one-line verdict come
first, and everything else is indented underneath.
"""

from __future__ import annotations

import os
import shutil
from typing import Any, Dict, List, Optional, Sequence

from .core import Diagnosis, Severity

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
COLORS = {
    Severity.CRITICAL: "\033[31m",
    Severity.WARN: "\033[33m",
    Severity.INFO: "\033[36m",
}
LABEL = {
    Severity.CRITICAL: "CRITICAL",
    Severity.WARN: "WARN",
    Severity.INFO: "INFO",
}


def _supports_color() -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    return True


class _Style:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def c(self, text: str, code: str) -> str:
        return "{}{}{}".format(code, text, RESET) if self.enabled else text

    def bold(self, text: str) -> str:
        return self.c(text, BOLD)

    def dim(self, text: str) -> str:
        return self.c(text, DIM)

    def sev(self, text: str, severity: Severity) -> str:
        return self.c(text, COLORS[severity] + BOLD)


def _wrap(text: str, width: int, indent: str) -> List[str]:
    words = text.split()
    lines: List[str] = []
    cur = ""
    for w in words:
        candidate = w if not cur else cur + " " + w
        if len(candidate) + len(indent) > width and cur:
            lines.append(indent + cur)
            cur = w
        else:
            cur = candidate
    if cur:
        lines.append(indent + cur)
    return lines


def _term_width(default: int = 92) -> int:
    try:
        return min(max(shutil.get_terminal_size((default, 24)).columns, 60), 110)
    except Exception:
        return default


def _fmt_evidence(evidence: Dict[str, Any]) -> str:
    parts = []
    for k, v in evidence.items():
        if v is None:
            continue
        parts.append("{}={}".format(k, v))
    return "  ".join(parts)


def render(
    diagnoses: Sequence[Diagnosis],
    color: Optional[bool] = None,
    show_fixes: bool = True,
    width: Optional[int] = None,
) -> str:
    """Render freshly fired diagnoses for inline printing during training."""
    if not diagnoses:
        return ""
    st = _Style(_supports_color() if color is None else bool(color))
    w = width or _term_width()
    counts: Dict[Severity, int] = {}
    for d in diagnoses:
        counts[d.severity] = counts.get(d.severity, 0) + 1
    header_bits = [
        "{} {}".format(n, LABEL[s].lower())
        for s, n in sorted(counts.items(), key=lambda kv: -kv[0].rank)
    ]
    step = diagnoses[0].step
    out: List[str] = []
    out.append("")
    out.append(
        "{}  {}  {}".format(
            st.bold("syntropyrl"),
            st.dim("step {}".format(step)),
            ", ".join(header_bits),
        )
    )
    for d in sorted(diagnoses, key=lambda x: -x.severity.rank):
        out.append("")
        out.append(
            "  {} {}  {}".format(
                st.sev("[" + LABEL[d.severity] + "]", d.severity),
                st.bold(d.code),
                d.title,
            )
        )
        out.extend(_wrap(d.summary, w, "      "))
        ev = _fmt_evidence(d.evidence)
        if ev:
            out.extend(_wrap(st.dim(ev) if not st.enabled else ev, w, "      "))
        if d.causes:
            out.append("      " + st.dim("likely causes:"))
            for c in d.causes[:3]:
                out.extend(_wrap("- " + c, w, "        "))
        if show_fixes and d.fixes:
            out.append("      " + st.dim("try:"))
            for f in d.fixes[:3]:
                out.extend(_wrap("- " + f, w, "        "))
        out.append("      " + st.dim(d.url))
    out.append("")
    return "\n".join(out)


def render_summary(
    diagnoses: Sequence[Diagnosis],
    steps: int = 0,
    color: Optional[bool] = None,
    width: Optional[int] = None,
) -> str:
    """End-of-run summary: one line per unique failure, worst first."""
    st = _Style(_supports_color() if color is None else bool(color))
    w = width or _term_width()
    rule = "-" * min(w, 78)
    out: List[str] = ["", st.bold("syntropyrl report"), rule]
    if not diagnoses:
        out.append("  No failures detected across {} observed steps.".format(steps))
        out.append(
            "  "
            + st.dim(
                "Note: a clean report means no *known* failure mode fired, not that the "
                "run is good."
            )
        )
        out.append(rule)
        return "\n".join(out)

    first_seen: Dict[str, Diagnosis] = {}
    occurrences: Dict[str, int] = {}
    for d in diagnoses:
        occurrences[d.code] = occurrences.get(d.code, 0) + 1
        if d.code not in first_seen:
            first_seen[d.code] = d
    ordered = sorted(
        first_seen.values(), key=lambda d: (-d.severity.rank, d.step)
    )
    out.append(
        "  {} distinct failure modes across {} steps".format(len(ordered), steps)
    )
    out.append("")
    for d in ordered:
        out.append(
            "  {} {}  {}  {}".format(
                st.sev(LABEL[d.severity].ljust(8), d.severity),
                st.bold(d.code),
                d.title,
                st.dim("(first seen step {}, {}x)".format(d.step, occurrences[d.code])),
            )
        )
    out.append("")
    worst = ordered[0]
    out.append("  " + st.bold("Start here: ") + "{} - {}".format(worst.code, worst.title))
    if worst.fixes:
        out.extend(_wrap("-> " + worst.fixes[0], w, "    "))
    out.append("  " + st.dim(worst.url))
    out.append(rule)
    return "\n".join(out)
