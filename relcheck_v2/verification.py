"""
RelCheck v2 — Relation Verification
=====================================
Type-aware verification: spatial → deterministic geometry,
action/attribute → crop-based VQA with contrastive question.
"""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image

from .config import (
    SPATIAL_RELS, ACTION_WORDS, COUNTERFACTUAL_MAP,
    YES_SUPPORTED, YES_UNSUPPORTED,
)
from .api import vlm_call, encode_b64
from .detection import find_best_bbox, crop_to_bboxes
from .spatial import spatial_verdict
from .types import Detection, RelationType, VerificationResult, Triple, Verdict, Confidence
from ._logging import log


# ── Relation classification ──────────────────────────────────────────────

def classify_relation(rel: str) -> RelationType:
    """Classify a relation as SPATIAL, ACTION, or ATTRIBUTE.

    Uses keyword matching against curated sets.

    Args:
        rel: Relation string to classify.

    Returns:
        RelationType enum (SPATIAL, ACTION, or ATTRIBUTE).
    """
    r = rel.lower().strip()
    if r in SPATIAL_RELS:
        return RelationType.SPATIAL
    if any(w in r for w in ACTION_WORDS):
        return RelationType.ACTION
    return RelationType.ATTRIBUTE


# ── Main verifier ────────────────────────────────────────────────────────

def verify_triple(
    subject: str,
    relation: str,
    object_: str,
    detections: list[Detection],
    image: Image.Image,
) -> bool | None:
    """Full verification pipeline for a single (subject, relation, object) triple.

    Flow:
        1. If SPATIAL + both boxes found → try deterministic geometry
        2. If geometry returns None (ambiguous) → fall through to VQA
        3. VQA: 2 standard yes/no + 1 contrastive forced-choice
        4. Majority vote → True (supported) / False (hallucinated) / None (uncertain)

    Args:
        subject: The subject entity in the triple.
        relation: The relation word/phrase.
        object_: The object entity in the triple.
        detections: List of Detection objects from GroundingDINO.
        image: PIL Image to verify on.

    Returns:
        True if relation is supported, False if hallucinated, None if uncertain.
    """
    rel_type = classify_relation(relation)

    # ── Step 1: Spatial geometry (deterministic) ──
    if rel_type == RelationType.SPATIAL and detections:
        subj_box = find_best_bbox(subject, detections)
        obj_box = find_best_bbox(object_, detections)
        if subj_box and obj_box:
            verdict = spatial_verdict(subj_box, obj_box, relation)
            if verdict is not None:
                return verdict

    # ── Step 2: VQA fallback ──
    if image is None:
        log.debug("No image provided, returning uncertain verdict")
        return None

    subj_box = find_best_bbox(subject, detections) if detections else None
    obj_box = find_best_bbox(object_, detections) if detections else None

    using_crop = bool(subj_box and obj_box)
    crop = crop_to_bboxes(image, subj_box, obj_box, padding=0.15) if using_crop else image
    region_hint = "this cropped region" if using_crop else "the full image"
    crop_b64 = encode_b64(crop)

    yes_votes = 0
    no_votes = 0

    # ── 2 standard yes/no questions ──
    for i, question in enumerate([
        f"In this image, is the {subject} {relation} the {object_}? "
        f"Look carefully at {region_hint}. Answer YES or NO only.",
        f"Does the {subject} appear to be {relation} the {object_} here? "
        f"Observe {region_hint} closely. Answer YES or NO only.",
    ]):
        resp = vlm_call(
            [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{crop_b64}"}},
                {"type": "text", "text": question},
            ]}],
            max_tokens=5,
        )
        if not resp:
            log.debug("No response from VLM for question %d", i + 1)
            continue
        rl = resp.strip().lower()
        if "yes" in rl and "no" not in rl:
            yes_votes += 1
        elif "no" in rl:
            no_votes += 1

    # ── 1 contrastive forced-choice question ──
    cf_rel = COUNTERFACTUAL_MAP.get(relation.lower(), f"not {relation}")
    ab_flip = hash(f"{subject}{relation}{object_}") % 2 == 1
    opt_a, opt_b = (cf_rel, relation) if ab_flip else (relation, cf_rel)

    cf_question = (
        f"Look at {region_hint} carefully. Which is more accurate?\n"
        f"(A) The {subject} is {opt_a} the {object_}\n"
        f"(B) The {subject} is {opt_b} the {object_}\n"
        f"Answer ONLY the letter A or B."
    )
    cf_resp = vlm_call(
        [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{crop_b64}"}},
            {"type": "text", "text": cf_question},
        ]}],
        max_tokens=5,
    )
    if cf_resp:
        cr = cf_resp.strip().upper()
        chose_a = "A" in cr and "B" not in cr
        chose_b = "B" in cr and "A" not in cr
        if ab_flip:
            if chose_b:
                yes_votes += 1
            elif chose_a:
                no_votes += 1
        else:
            if chose_a:
                yes_votes += 1
            elif chose_b:
                no_votes += 1

    # ── Majority vote ──
    total = yes_votes + no_votes
    log.debug("VQA votes: %d yes, %d no for (%s, %s, %s)", yes_votes, no_votes, subject, relation, object_)
    if total == 0:
        return None
    if yes_votes > no_votes:
        return True
    if no_votes > yes_votes:
        return False
    return None


# ── Verdict decision (threshold-based) ───────────────────────────────────

def decide_verdict(
    triple: dict,
    geometry_result: dict | None = None,
    vqa_result: dict | None = None,
) -> VerificationResult:
    """Apply threshold rules to geometry and VQA results to produce a verdict.

    Used by stage-based verification pipelines (separate from
    the verify_triple all-in-one function above).

    Args:
        triple: The triple dict with 'subject', 'relation', 'object' keys.
        geometry_result: Optional dict with spatial verification results.
        vqa_result: Optional dict with VQA results.

    Returns:
        VerificationResult with verdict, confidence, reason, and evidence source.
    """
    # Convert dict triple to Triple dataclass if needed
    if isinstance(triple, dict):
        try:
            triple_obj = Triple.from_dict(triple)
        except Exception as e:
            log.warning("Failed to convert triple dict: %s", e)
            triple_obj = Triple(
                subject=triple.get("subject", ""),
                relation=triple.get("relation", ""),
                object=triple.get("object", ""),
            )
    else:
        triple_obj = triple

    # Rule 1: Geometry verdict takes priority
    if geometry_result and geometry_result.get("verdict") is not None:
        is_correct = bool(geometry_result["verdict"])
        verdict = Verdict.CORRECT if is_correct else Verdict.INCORRECT
        return VerificationResult(
            triple=triple_obj,
            verdict=verdict,
            confidence=Confidence.HIGH,
            reason="Deterministic spatial geometry verification",
            evidence_source="geometry",
        )

    # Rule 2: VQA-based decision
    if not vqa_result:
        log.debug("No VQA result provided for triple %s", triple_obj.claim)
        return VerificationResult(
            triple=triple_obj,
            verdict=Verdict.UNKNOWN,
            confidence=Confidence.LOW,
            reason="No verification evidence available",
            evidence_source="none",
        )

    vqa_answers = vqa_result.get("yes_no_answers", [])
    if not vqa_answers:
        log.debug("Empty VQA answers for triple %s", triple_obj.claim)
        return VerificationResult(
            triple=triple_obj,
            verdict=Verdict.UNKNOWN,
            confidence=Confidence.LOW,
            reason="No VQA answers collected",
            evidence_source="none",
        )

    yes_ratios = [a.get("yes_ratio", 0.5) for a in vqa_answers]
    avg_yes_ratio = float(np.mean(yes_ratios))

    # Rule 3: High yes_ratio → supported
    if avg_yes_ratio >= YES_SUPPORTED:
        log.debug("High yes_ratio (%.2f) for %s", avg_yes_ratio, triple_obj.claim)
        return VerificationResult(
            triple=triple_obj,
            verdict=Verdict.CORRECT,
            confidence=Confidence.HIGH if avg_yes_ratio >= 0.80 else Confidence.MEDIUM,
            reason=f"VQA consensus: {avg_yes_ratio:.2f} yes ratio",
            evidence_source="vqa",
        )

    # Rule 4: Low yes_ratio → hallucinated
    if avg_yes_ratio < YES_UNSUPPORTED:
        log.debug("Low yes_ratio (%.2f) for %s", avg_yes_ratio, triple_obj.claim)
        return VerificationResult(
            triple=triple_obj,
            verdict=Verdict.INCORRECT,
            confidence=Confidence.HIGH if avg_yes_ratio <= 0.20 else Confidence.MEDIUM,
            reason=f"VQA consensus: {avg_yes_ratio:.2f} yes ratio (below threshold {YES_UNSUPPORTED})",
            evidence_source="vqa",
        )

    # Rule 5: Uncertain zone — check contrastive
    contrastive = vqa_result.get("contrastive_answer")
    if contrastive:
        if contrastive.get("chose_original"):
            log.debug("Contrastive favored original for %s", triple_obj.claim)
            return VerificationResult(
                triple=triple_obj,
                verdict=Verdict.CORRECT,
                confidence=Confidence.MEDIUM,
                reason="Contrastive VQA favored original relation",
                evidence_source="vqa_contrastive",
            )
        elif contrastive.get("chose_alternative"):
            log.debug("Contrastive favored alternative for %s", triple_obj.claim)
            return VerificationResult(
                triple=triple_obj,
                verdict=Verdict.INCORRECT,
                confidence=Confidence.MEDIUM,
                reason="Contrastive VQA favored alternative relation",
                evidence_source="vqa_contrastive",
            )

    # Uncertain zone — no clear decision
    log.debug("Uncertain zone for %s: yes_ratio=%.2f", triple_obj.claim, avg_yes_ratio)
    return VerificationResult(
        triple=triple_obj,
        verdict=Verdict.UNKNOWN,
        confidence=Confidence.LOW,
        reason=f"Yes ratio {avg_yes_ratio:.2f} in uncertain zone [{YES_UNSUPPORTED}, {YES_SUPPORTED})",
        evidence_source="vqa_uncertain_zone",
    )


# ── Backward-compatibility helper ─────────────────────────────────────────

def verification_result_to_dict(result: VerificationResult) -> dict:
    """Convert a VerificationResult to a plain dict (backward compatibility).

    Args:
        result: The VerificationResult to convert.

    Returns:
        Plain dict with legacy format.
    """
    return {
        "verdict": result.verdict.value,
        "confidence": result.confidence.value,
        "evidence_used": result.evidence_source,
        "reason": result.reason,
        "claim": result.triple.claim,
    }
