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
from ._metrics import STAGE_FINAL, MetricsCollector

_SEGMENTER = pysbd.Segmenter(language="en", clean=False)


def enrich_short_caption(
    img_id: str,
    caption: str,
    kb: dict,
    metrics: MetricsCollector | None = None,
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
        metrics: Optional metrics collector for path logging.

    Returns:
        CorrectionResult with enriched caption and metadata.
    """
    hard_facts_list = kb.get("hard_facts", [])
    spatial_facts_list = kb.get("spatial_facts", [])
    visual_desc = kb.get("visual_description", "") or ""
    detections = kb.get("detections", [])

    if metrics is not None:
        n_det = len(detections) if detections else 0
        metrics.record_kb_content(
            img_id, hard_facts_list, spatial_facts_list, visual_desc, n_det,
        )

    hard = "\n".join(f"- {f}" for f in hard_facts_list) or "- None detected"
    spatial = "\n".join(f"- {f}" for f in spatial_facts_list) or "- No spatial facts"
    visual = visual_desc[:800] or "- No visual description"

    prompt = ANALYSIS_PROMPT.format(
        caption=caption,
        hard_facts=hard,
        spatial_facts=spatial,
        visual_description=visual,
    )

    improved = caption
    errors: list[dict] = []
    missing: list[dict] = []

    llm_analysis_success = False
    json_parse_success = False
    rejection_guard: str | None = None
    rejection_threshold: str | None = None

    raw = llm_call([{"role": "user", "content": prompt}], max_tokens=1000)
    if raw:
        llm_analysis_success = True
        try:
            result = json_repair.loads(raw)
            if isinstance(result, dict):
                json_parse_success = True
                errors = result.get("errors", [])
                missing = result.get("missing", [])
                cand = result.get("improved_caption", "").strip().strip('"').strip("'")
                if cand and len(cand) >= VERIFY_KB_MIN_LENGTH:
                    improved = cand
        except Exception:
            log.debug("[%s] Failed to parse enrichment JSON", img_id)

    # Guard: reject overly long enrichment
    sentence_guard_rejected = False
    candidate_sentence_count = 0
    if improved != caption:
        candidate_sentence_count = len(_SEGMENTER.segment(improved))
        if candidate_sentence_count > ENRICHMENT_MAX_SENTENCES:
            log.debug("[%s] Enrichment rejected: %d sentences > max %d",
                      img_id, candidate_sentence_count, ENRICHMENT_MAX_SENTENCES)
            sentence_guard_rejected = True
            rejection_guard = "sentence_count"
            rejection_threshold = str(ENRICHMENT_MAX_SENTENCES)
            improved = caption

    # Verify against KB
    kb_verification_passed: bool | None = None
    if improved != caption:
        # Handle both Detection objects and tuples
        if detections and hasattr(detections[0], "label"):
            counts = Counter(d.label for d in detections)
        else:
            counts = Counter(label for label, _, _ in detections)

        obj_str = ", ".join(f"{c}x {l}" for l, c in counts.most_common(10))
        rel_str = visual_desc[:500]
        verdict = llm_call(
            [{"role": "user", "content": VERIFY_PROMPT.format(
                rewritten=improved, objects=obj_str, relationships=rel_str
            )}],
            max_tokens=50,
        )
        if verdict and verdict.upper().startswith("FAIL"):
            log.debug("[%s] Enrichment FAILED KB verification: %s", img_id, verdict)
            kb_verification_passed = False
            rejection_guard = "kb_verification"
            rejection_threshold = "PASS/FAIL"
            improved = caption
        else:
            kb_verification_passed = True

    if metrics is not None:
        metrics.record_enrichment(
            img_id,
            llm_analysis_success=llm_analysis_success,
            json_parse_success=json_parse_success,
            n_errors_found=len(errors),
            n_missing_found=len(missing),
            candidate_sentence_count=candidate_sentence_count,
            sentence_guard_rejected=sentence_guard_rejected,
            sentence_guard_threshold=ENRICHMENT_MAX_SENTENCES,
            kb_verification_passed=kb_verification_passed,
            rejection_guard=rejection_guard,
            rejection_threshold=rejection_threshold,
            kb_hard_facts_count=len(hard_facts_list),
            kb_spatial_facts_count=len(spatial_facts_list),
            kb_visual_description_len=len(visual_desc),
            kb_detections_count=len(detections) if detections else 0,
        )
        metrics.record_caption_snapshot(img_id, STAGE_FINAL, improved)

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
