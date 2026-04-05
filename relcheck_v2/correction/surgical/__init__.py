"""
RelCheck v2 — Long Caption Surgical Correction (Orchestrator)
===============================================================
Thin orchestrator that wires together extraction, verification,
correction, and addendum modules for captions >= 30 words.

Key properties:
    - Never shortens captions (surgical edit only)
    - Type-aware verification: spatial → geometry, action → VQA
    - Cross-captioner consensus pre-filter (free signal)
    - Post-verification: checks corrected caption for new hallucinations
"""

from __future__ import annotations

from PIL import Image

from ..._logging import log
from ...config import ENABLE_NLI, SPATIAL_OPPOSITES
from ...entity import levenshtein_distance
from ...types import (
    CorrectionMode, CorrectionResult, CorrectionError,
    VerificationResult, RelationType, Verdict, Confidence,
)
from .._metrics import (
    STAGE_BATCH_CANDIDATE, STAGE_FALLBACK_DELETION,
    STAGE_FINAL, STAGE_INPUT, STAGE_MISSING_FACT_ADDENDUM,
    STAGE_POST_VERIFY_REVERT, STAGE_SPATIAL_ADDENDUM,
    MetricsCollector,
)
from ._addendum import build_spatial_addendum, add_missing_fact_addendum
from ._application import apply_batch_correction, apply_sequential_correction
from ._extraction import extract_triples
from .._utils import entity_matches
from ._verify import (
    check_spatial_contradictions,
    verify_spatial_triple,
    verify_action_attribute_triple,
)
from .._vqa import _parse_spatial_facts
from .._nli import nli_check_triples_batch, NLIResult

__all__ = ["correct_long_caption"]


def correct_long_caption(
    img_id: str,
    caption: str,
    kb: dict,
    pil_image: Image.Image | None = None,
    cross_captions: dict[str, str] | None = None,
    include_addendum: bool = True,
    metrics: MetricsCollector | None = None,
    enable_nli: bool = ENABLE_NLI,
) -> CorrectionResult:
    """Per-triple verification and surgical correction for detailed captions.

    Pipeline:
        1. Extract relational triples via LLM
        2. For each triple: type-aware verification (geometry + VQA)
        3. Batch correction of all confirmed errors
        4. Post-verification: check for introduced hallucinations
        5. Append missing spatial facts (skipped if include_addendum=False)

    Args:
        img_id: Image identifier (for logging).
        caption: Original caption text (>= 30 words).
        kb: Visual KB dict.
        pil_image: Full PIL image (required for VQA).
        cross_captions: Optional {captioner_name: caption_text} for consensus.
        include_addendum: If False, skip Step 5 (fact appending). Use this for
            the correction-only ablation to isolate hallucination correction
            from information enrichment.
        metrics: Optional metrics collector for path logging.

    Returns:
        CorrectionResult with corrected caption and metadata.
    """
    if pil_image is None:
        return CorrectionResult(
            original=caption, corrected=caption,
            mode=CorrectionMode.CORRECT_V2, status="unchanged",
        )

    # Record KB content
    if metrics is not None:
        metrics.record_caption_snapshot(img_id, STAGE_INPUT, caption)
        spatial_facts_list = kb.get("spatial_facts", [])
        hard_facts_list = kb.get("hard_facts", [])
        visual_desc = kb.get("visual_description", "") or ""
        detections = kb.get("detections", [])
        metrics.record_kb_content(
            img_id, hard_facts_list, spatial_facts_list, visual_desc,
            len(detections) if detections else 0,
        )

    # ── Step 1: Extract triples ──
    triples = extract_triples(caption)

    if metrics is not None:
        spatial_count = sum(1 for t in triples if t.rel_type == RelationType.SPATIAL)
        action_count = sum(1 for t in triples if t.rel_type == RelationType.ACTION)
        attribute_count = sum(1 for t in triples if t.rel_type == RelationType.ATTRIBUTE)
        metrics.record_extraction(
            img_id,
            total=len(triples),
            spatial=spatial_count,
            action=action_count,
            attribute=attribute_count,
            addendum_only=len(triples) == 0,
        )

    if not triples:
        log.debug("[%s] 0 triples extracted — addendum only", img_id)
        corrected, n_addendum = build_spatial_addendum(
            corrected_caption=caption, kb=kb,
            img_id=img_id, metrics=metrics,
        )
        if metrics is not None and corrected != caption:
            metrics.record_caption_snapshot(img_id, STAGE_SPATIAL_ADDENDUM, corrected)
        edit_dist = levenshtein_distance(caption, corrected)
        if metrics is not None:
            metrics.record_caption_snapshot(img_id, STAGE_FINAL, corrected)
        return CorrectionResult(
            original=caption, corrected=corrected,
            mode=CorrectionMode.CORRECT_V2,
            edit_rate=edit_dist / max(len(caption), 1),
            n_triples=0, n_addendum=n_addendum,
            status="modified" if corrected != caption else "unchanged",
        )

    # ── Step 2: Verify each triple ──
    spatial_facts = kb.get("spatial_facts", [])
    from ...config import SKIP_KB_GEOMETRY
    if not SKIP_KB_GEOMETRY:
        geo_contras = check_spatial_contradictions(caption, spatial_facts)
        geo_set = {c.lower() for c in geo_contras}
    else:
        geo_set = set()

    # ── Step 1.5: Batch NLI pre-filter (when enabled) ──
    nli_results: list[NLIResult] | None = None
    if enable_nli and triples:
        nli_results = nli_check_triples_batch(
            triples,
            spatial_facts=spatial_facts,
            visual_description=kb.get("visual_description", "") or "",
            hard_facts=kb.get("hard_facts", []),
        )

    errors: list[CorrectionError] = []
    all_checks: list[VerificationResult] = []

    for i, triple in enumerate(triples):
        nli_result = nli_results[i] if nli_results else None

        if triple.rel_type == RelationType.SPATIAL:
            verify_spatial_triple(
                triple, kb, pil_image, spatial_facts, geo_set,
                errors, all_checks, img_id,
                metrics=metrics,
                nli_result=nli_result,
            )
        else:
            verify_action_attribute_triple(
                triple, kb, pil_image, cross_captions,
                errors, all_checks, img_id,
                metrics=metrics,
                nli_result=nli_result,
            )

    # ── Step 3: Apply corrections ──
    corrected = caption
    applied: list[dict] = []

    if errors:
        corrected, applied = apply_sequential_correction(
            img_id, caption, errors, kb, pil_image,
            metrics=metrics,
        )

    # ── Step 4: Post-verification ──
    if corrected != caption:
        new_triples = extract_triples(corrected)
        introduced_errors: list[str] = []
        for nt in new_triples:
            if nt.rel_type == RelationType.SPATIAL:
                kb_triples = _parse_spatial_facts(kb.get("spatial_facts", []))
                for kb_s, kb_r, kb_o in kb_triples:
                    if entity_matches(nt.subject, kb_s) and entity_matches(nt.object, kb_o):
                        opp = SPATIAL_OPPOSITES.get(nt.relation.lower())
                        if opp and kb_r.lower() == opp:
                            introduced_errors.append(f"{nt.claim} (KB says {kb_r})")
                            break

        if introduced_errors:
            log.info("[%s] POST-CHECK: correction introduced errors: %s → reverting",
                     img_id, introduced_errors)
            corrected = caption
            applied = []
            if metrics is not None:
                metrics.record_post_verification(
                    img_id, n_contradictions=len(introduced_errors), reverted=True,
                )
                metrics.record_caption_snapshot(
                    img_id, STAGE_POST_VERIFY_REVERT, corrected,
                )
        else:
            if metrics is not None:
                metrics.record_post_verification(
                    img_id, n_contradictions=0, reverted=False,
                )
    else:
        if metrics is not None:
            metrics.record_post_verification(
                img_id, n_contradictions=0, reverted=False,
            )

    # ── Step 5: Addendum ──
    n_addendum = 0
    if include_addendum:
        pre_addendum = corrected
        corrected, n_spatial = build_spatial_addendum(
            corrected_caption=corrected, kb=kb,
            img_id=img_id, metrics=metrics,
        )
        if metrics is not None and corrected != pre_addendum:
            metrics.record_caption_snapshot(img_id, STAGE_SPATIAL_ADDENDUM, corrected)

        pre_missing = corrected
        corrected, n_missing = add_missing_fact_addendum(
            corrected_caption=corrected, kb=kb,
            img_id=img_id, metrics=metrics,
        )
        if metrics is not None and corrected != pre_missing:
            metrics.record_caption_snapshot(img_id, STAGE_MISSING_FACT_ADDENDUM, corrected)

        n_addendum = n_spatial + n_missing
        if n_addendum:
            log.debug("[%s] +%d fact(s) appended (%d spatial, %d missing)",
                      img_id, n_addendum, n_spatial, n_missing)

    if metrics is not None:
        metrics.record_caption_snapshot(img_id, STAGE_FINAL, corrected)

    edit_dist = levenshtein_distance(caption, corrected)
    edit_rate_val = edit_dist / max(len(caption), len(corrected), 1)

    return CorrectionResult(
        original=caption,
        corrected=corrected,
        errors=errors,
        checks=all_checks,
        mode=CorrectionMode.CORRECT_V2,
        edit_rate=edit_rate_val,
        n_triples=len(triples),
        n_addendum=n_addendum,
        status="modified" if corrected != caption else "unchanged",
    )
