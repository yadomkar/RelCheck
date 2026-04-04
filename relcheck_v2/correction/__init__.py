"""
RelCheck v2 — Correction Subpackage
=====================================
Dispatches to the appropriate correction strategy based on caption length:

- **Short captions** (< 30 words, e.g. BLIP-2): full KB-guided enrichment
  that fixes errors and adds verified missing facts.
- **Long captions** (>= 30 words, e.g. LLaVA, Qwen): surgical per-triple
  verification and minimal span editing that never shortens captions.

Public API
----------
enrich_caption_v3   — auto-dispatch entry point (recommended)
enrich_short_caption — direct access to enrichment pipeline
correct_long_caption — direct access to surgical correction pipeline
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .._logging import log
from ..config import SHORT_CAPTION_THRESHOLD
from ..types import CorrectionResult
from ._enrichment import enrich_short_caption
from .surgical import correct_long_caption

if TYPE_CHECKING:
    from PIL import Image

__all__ = [
    "enrich_caption_v3",
    "enrich_short_caption",
    "correct_long_caption",
]


def enrich_caption_v3(
    img_id: str,
    caption: str,
    kb: dict,
    pil_image: "Image.Image | None" = None,
    cross_captions: dict[str, str] | None = None,
) -> CorrectionResult:
    """Auto-dispatch correction based on caption word count.

    Short captions (< SHORT_CAPTION_THRESHOLD words) get full KB-guided
    enrichment.  Long captions get surgical per-triple correction that
    preserves all non-hallucinated content.

    Args:
        img_id: Image identifier (for logging / checkpoints).
        caption: Original caption text from any captioning model.
        kb: Visual Knowledge Base dict (hard_facts, spatial_facts,
            visual_description, detections).
        pil_image: PIL image for crop-based VQA (required for long
            caption correction; ignored for enrichment).
        cross_captions: Mapping of {captioner_name: caption} from other
            models, used for consensus pre-filtering in long captions.

    Returns:
        CorrectionResult with corrected/enriched caption and metadata.
    """
    word_count = len(caption.split())

    if word_count < SHORT_CAPTION_THRESHOLD:
        log.info(
            "[%s] Short caption (%d words < %d) → enrichment mode",
            img_id, word_count, SHORT_CAPTION_THRESHOLD,
        )
        return enrich_short_caption(img_id, caption, kb)

    log.info(
        "[%s] Long caption (%d words >= %d) → surgical correction mode",
        img_id, word_count, SHORT_CAPTION_THRESHOLD,
    )
    return correct_long_caption(
        img_id, caption, kb,
        pil_image=pil_image,
        cross_captions=cross_captions,
    )
