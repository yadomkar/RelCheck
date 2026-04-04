"""
RelCheck v2 — Relation Verification
=====================================
Type-aware verification: spatial → deterministic geometry,
action/attribute → crop-based VQA with contrastive question.
"""

from __future__ import annotations

from io import BytesIO
from typing import Optional

import numpy as np
from PIL import Image

from .config import (
    SPATIAL_RELS, ACTION_WORDS, COUNTERFACTUAL_MAP,
    YES_SUPPORTED, YES_UNSUPPORTED,
)
from .api import vlm_call, encode_b64
from .detection import find_best_bbox, crop_to_bboxes, Detection
from .spatial import spatial_verdict


# ── Relation classification ──────────────────────────────────────────────

def classify_relation(rel: str) -> str:
    """Classify a relation as SPATIAL, ACTION, or ATTRIBUTE.

    Uses keyword matching against curated sets.
    """
    r = rel.lower().strip()
    if r in SPATIAL_RELS:
        return "SPATIAL"
    if any(w in r for w in ACTION_WORDS):
        return "ACTION"
    return "ATTRIBUTE"


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
    """
    rel_type = classify_relation(relation)

    # ── Step 1: Spatial geometry (deterministic) ──
    if rel_type == "SPATIAL" and detections:
        subj_box = find_best_bbox(subject, detections)
        obj_box = find_best_bbox(object_, detections)
        if subj_box and obj_box:
            verdict = spatial_verdict(subj_box, obj_box, relation)
            if verdict is not None:
                return verdict

    # ── Step 2: VQA fallback ──
    if image is None:
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
    for question in [
        f"In this image, is the {subject} {relation} the {object_}? "
        f"Look carefully at {region_hint}. Answer YES or NO only.",
        f"Does the {subject} appear to be {relation} the {object_} here? "
        f"Observe {region_hint} closely. Answer YES or NO only.",
    ]:
        resp = vlm_call(
            [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{crop_b64}"}},
                {"type": "text", "text": question},
            ]}],
            max_tokens=5,
        )
        if not resp:
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
) -> dict:
    """Apply threshold rules to geometry and VQA results to produce a verdict.

    Used by RelCheck_PlanA.ipynb's stage-based pipeline (separate from
    the verify_triple all-in-one function above).
    """
    # Rule 1: Geometry verdict takes priority
    if geometry_result and geometry_result.get("verdict") is not None:
        return {
            "verdict": "supported" if geometry_result["verdict"] else "unsupported",
            "confidence": 0.95,
            "evidence_used": "geometry",
            "correct_relation": geometry_result.get("correct_relation"),
        }

    # Rule 2: VQA-based decision
    if not vqa_result:
        return {
            "verdict": "uncertain",
            "confidence": 0.0,
            "evidence_used": "none",
        }

    vqa_answers = vqa_result.get("yes_no_answers", [])
    if not vqa_answers:
        return {
            "verdict": "uncertain",
            "confidence": 0.0,
            "evidence_used": "none",
        }

    yes_ratios = [a.get("yes_ratio", 0.5) for a in vqa_answers]
    avg_yes_ratio = float(np.mean(yes_ratios))

    # Rule 3: High yes_ratio → supported
    if avg_yes_ratio >= YES_SUPPORTED:
        return {
            "verdict": "supported",
            "confidence": avg_yes_ratio,
            "evidence_used": "vqa",
            "yes_ratios": yes_ratios,
        }

    # Rule 4: Low yes_ratio → hallucinated
    if avg_yes_ratio < YES_UNSUPPORTED:
        return {
            "verdict": "unsupported",
            "confidence": 1.0 - avg_yes_ratio,
            "evidence_used": "vqa",
            "yes_ratios": yes_ratios,
        }

    # Rule 5: Uncertain zone — check contrastive
    contrastive = vqa_result.get("contrastive_answer")
    if contrastive:
        if contrastive.get("chose_original"):
            return {
                "verdict": "supported",
                "confidence": 0.7,
                "evidence_used": "vqa_contrastive",
                "yes_ratios": yes_ratios,
            }
        elif contrastive.get("chose_alternative"):
            return {
                "verdict": "unsupported",
                "confidence": 0.7,
                "evidence_used": "vqa_contrastive",
                "yes_ratios": yes_ratios,
            }

    return {
        "verdict": "uncertain",
        "confidence": 0.5,
        "evidence_used": "vqa_uncertain_zone",
        "yes_ratios": yes_ratios,
        "reason": f"Yes ratio {avg_yes_ratio:.2f} in uncertain zone "
                  f"[{YES_UNSUPPORTED}, {YES_SUPPORTED})",
    }
