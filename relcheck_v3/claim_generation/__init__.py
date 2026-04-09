"""Claim generation pipeline — Woodpecker Stages 1–4 (Visual Knowledge Base construction)."""

from relcheck_v3.claim_generation.config import ClaimGenConfig
from relcheck_v3.claim_generation.models import SampleResult, VisualKnowledgeBase

# ClaimGenerationPipeline imported lazily to avoid requiring GPU-only deps
# (groundingdino, spacy) at import time. Use:
#   from relcheck_v3.claim_generation.pipeline import ClaimGenerationPipeline


def __getattr__(name: str):
    if name == "ClaimGenerationPipeline":
        from relcheck_v3.claim_generation.pipeline import ClaimGenerationPipeline
        return ClaimGenerationPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ClaimGenerationPipeline",
    "ClaimGenConfig",
    "SampleResult",
    "VisualKnowledgeBase",
]
