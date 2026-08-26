"""Detector registry.

Every detector maps 1:1 to an entry in the RL Failure Atlas. Adding a detector
means adding an Atlas page, a repro script, and a test. That coupling is the
whole quality bar of this project.
"""

from __future__ import annotations

from typing import Dict, List, Type

from ..core import Detector
from .distribution import DistributionalDrift, EntropyCollapse, ModeCollapse
from .numerics import GradientPathology, TemplateMismatch, TrainerGeneratorDivergence
from .optimization import AdvantageCollapse, ClipSaturation, KLBlowup, ValueDivergence
from .reward import DeadReward, LengthExploit, RewardHacking, TruncationBias

DETECTOR_CLASSES: List[Type[Detector]] = [
    # numerics
    TrainerGeneratorDivergence,
    TemplateMismatch,
    GradientPathology,
    # optimization
    AdvantageCollapse,
    KLBlowup,
    ClipSaturation,
    ValueDivergence,
    # reward
    RewardHacking,
    LengthExploit,
    DeadReward,
    TruncationBias,
    # distribution
    DistributionalDrift,
    EntropyCollapse,
    ModeCollapse,
]


def all_detectors() -> List[Detector]:
    """Fresh instances of every registered detector."""
    return [cls() for cls in DETECTOR_CLASSES]


def by_code() -> Dict[str, Type[Detector]]:
    return {cls.code: cls for cls in DETECTOR_CLASSES}


def catalog() -> List[Dict[str, str]]:
    """Machine-readable catalog, used by the CLI and to build the Atlas page."""
    return [
        {
            "code": cls.code,
            "title": cls.title,
            "family": cls.family,
            "requires": ", ".join(cls.requires) or "rollouts",
            "doc": (cls.__doc__ or "").strip().split("\n")[0],
        }
        for cls in DETECTOR_CLASSES
    ]


__all__ = [
    "DETECTOR_CLASSES",
    "all_detectors",
    "by_code",
    "catalog",
    "TrainerGeneratorDivergence",
    "TemplateMismatch",
    "GradientPathology",
    "AdvantageCollapse",
    "KLBlowup",
    "ClipSaturation",
    "ValueDivergence",
    "RewardHacking",
    "LengthExploit",
    "DeadReward",
    "TruncationBias",
    "DistributionalDrift",
    "EntropyCollapse",
    "ModeCollapse",
]
