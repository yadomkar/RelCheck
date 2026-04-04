"""
RelCheck v2 — Short Caption Enrichment
========================================
KB-guided full rewrite for captions under 30 words (e.g. BLIP-2).
Fixes errors and adds verified missing facts from the Visual KB.
"""

from __future__ import annotations

from collections import Counter

import json_repair
import pysbd
from .._logging import log
from ..api import llm_call
from ..config import ENRICHMENT_MAX_SENTENCES, VERIFY_KB_MIN_LENGTH
from ..entity import levenshtein_distance
from ..prompts import ANALYSIS_PROMPT, VERIFY_PROMPT
from ..types import CorrectionMode, CorrectionResult

_SEGMENTER = pysbd.Segmenter(language="en", clean=False)


def enrich_short_caption(
    img_id: str,
    caption: str,
    kb: dict,
) -> CorrectionResult:
    """Full KB-guided enrichment for captions under 30 words.

    Pipeline:
        1. Build prompt with KB evidence (hard facts, spatial, visual description)
        2. LLM analyzes: find errors + missing facts → improved caption in JSON
        3. Guard: reject if > ENRICHMENT_MAX_SENTENCES sentences
        4. Verify against KB (faithfulness + fluency + coherence)
        5. Keep if verified; revert if KB contradiction

    Args:
        img_id: Image identifier (for logging).
        caption: Original short caption text.
        kb: Visual KB dict with hard_facts, spatial_facts, visual_description, detections.

    Returns:
        CorrectionResult with enriched caption and metadata.
    """
    hard = "\n".join(f"- {f}" for f in kb.get("hard_facts", [])) or "- None detected"
    spatial = "\n".join(f"- {f}" for f in kb.get("spatial_facts", [])) or "- No spatial facts"
    visual = (kb.get("visual_description", "") or "")[:800] or "- No visual description"

    prompt = ANALYSIS_PROMPT.format(
        caption=caption,
        hard_facts=hard,
        spatial_facts=spatial,
        visual_description=visual,
    )

    improved = caption
    errors: list[dict] = []
    missing: list[dict] = []

    raw = llm_call([{"role": "user", "content": prompt}], max_tokens=1000)
    if raw:
        try:
            result = json_repair.loads(raw)
            if isinstance(result, dict):
                errors = result.get("errors", [])
                missing = result.get("missing", [])
                cand = result.get("improved_caption", "").strip().strip('"').strip("'")
                if cand and len(cand) >= VERIFY_KB_MIN_LENGTH:
                    improved = cand
        except Exception:
            log.debug("[%s] Failed to parse enrichment JSON", img_id)

    # Guard: reject overly long enrichment
    if improved != caption:
        n_sent = len(_SEGMENTER.segment(improved))
        if n_sent > ENRICHMENT_MAX_SENTENCES:
            log.debug("[%s] Enrichment rejected: %d sentences > max %d",
                      img_id, n_sent, ENRICHMENT_MAX_SENTENCES)
            improved = caption

    # Verify against KB
    if improved != caption:
        detections = kb.get("detections", [])
        # Handle both Detection objects and tuples
        if detections and hasattr(detections[0], "label"):
            counts = Counter(d.label for d in detections)
        else:
            counts = Counter(label for label, _, _ in detections)

        obj_str = ", ".join(f"{c}x {l}" for l, c in counts.most_common(10))
        rel_str = (kb.get("visual_description", "") or "")[:500]
        verdict = llm_call(
            [{"role": "user", "content": VERIFY_PROMPT.format(
                rewritten=improved, objects=obj_str, relationships=rel_str
            )}],
            max_tokens=50,
        )
        if verdict and verdict.upper().startswith("FAIL"):
            log.debug("[%s] Enrichment FAILED KB verification: %s", img_id, verdict)
            improved = caption

    edit_dist = levenshtein_distance(caption, improved)
    edit_rate_val = edit_dist / max(len(caption), len(improved), 1)

    return CorrectionResult(
        original=caption,
        corrected=improved,
        errors=[],  # Enrichment doesn't produce typed CorrectionError objects
        checks=[],
        mode=CorrectionMode.ENRICH,
        edit_rate=edit_rate_val,
        status="modified" if improved != caption else "unchanged",
    )
