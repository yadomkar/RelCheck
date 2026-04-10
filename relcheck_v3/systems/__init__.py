"""Correction systems for the evaluation harness.

Exports the ``CorrectionSystem`` protocol and lightweight implementations
eagerly.  Heavy implementations (Woodpecker, RelCheck variants) that depend
on GPU libraries are available via lazy imports to avoid pulling in
``torchvision`` / ``transformers`` at package-import time.
"""

from __future__ import annotations

from relcheck_v3.systems.base import CorrectionSystem
from relcheck_v3.systems.raw_mllm import RawMLLM

__all__ = [
    "CorrectionSystem",
    "RawMLLM",
    "RelCheckClaimOnly",
    "RelCheckClaimGeom",
    "RelCheckFull",
    "WoodpeckerBaseline",
]


def __getattr__(name: str):  # noqa: ANN001
    """Lazy-import heavy correction systems on first access."""
    if name == "WoodpeckerBaseline":
        from relcheck_v3.systems.woodpecker import WoodpeckerBaseline
        return WoodpeckerBaseline
    if name == "RelCheckClaimOnly":
        from relcheck_v3.systems.relcheck_claim import RelCheckClaimOnly
        return RelCheckClaimOnly
    if name == "RelCheckClaimGeom":
        from relcheck_v3.systems.relcheck_claim_geom import RelCheckClaimGeom
        return RelCheckClaimGeom
    if name == "RelCheckFull":
        from relcheck_v3.systems.relcheck_full import RelCheckFull
        return RelCheckFull
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
