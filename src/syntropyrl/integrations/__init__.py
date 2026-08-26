"""Explicit framework adapters.

We do not monkeypatch. A tool whose job is finding subtle numerical bugs has no
business silently rewriting your training loop, and an adapter you can read is
an adapter you can trust.

Each adapter is a thin translation layer: framework metric names in, syntropyrl
`Step` records out.
"""

from __future__ import annotations

from .generic import SyntropyRLCallback, from_metrics

__all__ = ["SyntropyRLCallback", "from_metrics", "trl_callback"]


def trl_callback(**kwargs):
    """Return a TRL/transformers-compatible TrainerCallback.

    Imported lazily so `transformers` is never a hard dependency.
    """
    from .trl import make_trl_callback

    return make_trl_callback(**kwargs)
