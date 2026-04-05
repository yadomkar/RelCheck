"""
RelCheck v2 — Correction VQA & Relation Querying
==================================================
VQA-based verification and relation correction used by
the surgical correction pipeline. Includes:

- Entity existence checks (full-image VQA)
- Action triple verification (crop-based multi-question voting)
- KB-first spatial relation correction
- KB-first action relation correction (3-stage cascade)
"""

from __future__ import annotations

import re
from io import BytesIO

from PIL import Image

from .._logging import log
from ..api import vlm_call, llm_call, encode_b64
from ..config import (
    SPATIAL_OPPOSITES, COUNTERFACTUAL_MAP, CROP_PADDING, CROP_PADDING_WIDE,
)
from ..detection import find_best_bbox_from_kb, crop_to_bboxes
from ..types import BBox
from ._utils import core_noun, entity_matches, normalize_entity


# ── Spatial fact parsing (local to correction) ───────────────────────────

def _parse_spatial_facts(spatial_facts: list[str]) -> list[tuple[str, str, str]]:
    """Parse KB spatial fact strings into (subj, rel, obj) triples."""
    from ..spatial import SPATIAL_TRIPLE_RE

    parsed: list[tuple[str, str, str]] = []
    for fact in spatial_facts:
        fact_clean = fact.replace("'", "").replace('"', "")
        for m in SPATIAL_TRIPLE_RE.finditer(fact_clean.lower()):
            subj = m.group(1).strip().rstrip(" ,;")
            rel = m.group(2).strip()
            obj = m.group(3).strip().rstrip(" ,;.")
            parsed.append((subj, rel, obj))
    return parsed


# ── Entity existence ─────────────────────────────────────────────────────


def check_entity_exists_vqa(
    entity: str,
    pil_image: Image.Image,
    retries: int = 2,
) -> bool | None:
    """Ask VLM whether an entity exists in the image.

    Uses two yes/no questions with majority vote.

    Args:
        entity: Entity name to check (e.g. "dog", "red car").
        pil_image: Full PIL image.
        retries: Unused (kept for backward compat).

    Returns:
        True (exists), False (absent), or None (uncertain).
    """
    b64 = encode_b64(pil_image)

    questions = [
        f"Is there a {entity} visible anywhere in this image? Answer only YES or NO.",
        f"Can you see a {entity} in this scene? Look carefully. Answer YES or NO only.",
    ]
    yes_v = no_v = 0
    for q in questions:
        r = vlm_call(
            [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": q},
            ]}],
            max_tokens=5,
        )
        if not r:
            continue
        rl = r.strip().lower()
        if "yes" in rl and "no" not in rl:
            yes_v += 1
        elif "no" in rl:
            no_v += 1

    total = yes_v + no_v
    if total == 0:
        return None
    if no_v > yes_v:
        return False
    if yes_v > no_v:
        return True
    return None


# ── Geometry hint builder for grounded VQA ───────────────────────────────

_GEO_HINTS: dict[str, dict[bool, str]] = {
    "mounting": {
        True: "Object detection shows the {subj} positioned above the {obj} with overlap. ",
        False: "Object detection shows the {subj} is NOT positioned above the {obj}. ",
    },
    "grasping": {
        True: "Pose estimation shows the {subj}'s hands near the {obj}. ",
        False: "Pose estimation shows the {subj}'s hands are far from the {obj} (no physical contact detected). ",
    },
    "consuming": {
        True: "Pose estimation shows the {subj}'s mouth area near the {obj}. ",
        False: "Pose estimation shows the {subj}'s mouth area is far from the {obj}. ",
    },
    "containment": {
        True: "Object detection shows the {subj} within the {obj}'s boundaries. ",
        False: "Object detection shows the {subj} is NOT within the {obj}'s boundaries. ",
    },
    "adjacency": {
        True: "Object detection shows the {subj} and {obj} are close together. ",
        False: "Object detection shows the {subj} and {obj} are far apart. ",
    },
}


def _build_geo_hint(
    subj: str,
    rel: str,
    obj: str,
    geo_family: str | None,
    geo_prereq: bool | None,
    subj_box: list[float] | None = None,
    obj_box: list[float] | None = None,
) -> str:
    """Build a geometry evidence prefix for VQA prompts.

    When geometry data is available, returns a short sentence describing
    what the detector/pose estimator found. For known action families,
    uses a specific template. For unknown verbs with bboxes available,
    generates a generic spatial context hint from bbox positions.

    Args:
        subj: Subject entity name.
        rel: Relation verb.
        obj: Object entity name.
        geo_family: Action geometry family or None.
        geo_prereq: Geometry prerequisite result (True/False) or None.
        subj_box: Normalized bbox [x1, y1, x2, y2] of subject, or None.
        obj_box: Normalized bbox [x1, y1, x2, y2] of object, or None.

    Returns:
        Geometry hint string (with trailing space) or empty string.
    """
    # Known family with geometry result → specific hint
    if geo_family is not None and geo_prereq is not None:
        templates = _GEO_HINTS.get(geo_family)
        if templates is not None:
            template = templates.get(geo_prereq)
            if template is not None:
                return template.format(subj=subj, obj=obj)

    # Generic bbox context when both entities are detected but no family matched
    if subj_box is not None and obj_box is not None:
        return _bbox_context_hint(subj, obj, subj_box, obj_box)

    return ""


def _bbox_context_hint(
    subj: str,
    obj: str,
    subj_box: list[float],
    obj_box: list[float],
) -> str:
    """Generate a rich spatial context hint from bounding box geometry.

    Computes multiple geometric features used in HOI detection literature:
    relative position, IoU overlap, relative scale, contact likelihood,
    and containment ratio. Produces a natural language summary for the VLM.

    Args:
        subj: Subject entity name.
        obj: Object entity name.
        subj_box: Normalized bbox [x1, y1, x2, y2] of subject.
        obj_box: Normalized bbox [x1, y1, x2, y2] of object.

    Returns:
        Multi-sentence spatial description string with trailing space.
    """
    sx1, sy1, sx2, sy2 = subj_box
    ox1, oy1, ox2, oy2 = obj_box

    # Centroids
    sx, sy = (sx1 + sx2) / 2, (sy1 + sy2) / 2
    ox, oy = (ox1 + ox2) / 2, (oy1 + oy2) / 2

    # Sizes
    s_w, s_h = sx2 - sx1, sy2 - sy1
    o_w, o_h = ox2 - ox1, oy2 - oy1
    s_area = max(s_w * s_h, 1e-6)
    o_area = max(o_w * o_h, 1e-6)

    # ── Relative position ────────────────────────────────────────────
    dx = ox - sx
    dy = oy - sy
    dist = (dx**2 + dy**2) ** 0.5

    if abs(dx) < 0.05 and abs(dy) < 0.05:
        pos_desc = "directly overlapping"
    else:
        v_part = "above" if dy > 0.05 else ("below" if dy < -0.05 else "")
        h_part = "to the right of" if dx > 0.05 else ("to the left of" if dx < -0.05 else "")
        if v_part and h_part:
            pos_desc = f"{v_part} and {h_part}"
        else:
            pos_desc = v_part or h_part or "near"

    # ── IoU overlap ──────────────────────────────────────────────────
    inter_x = max(0, min(sx2, ox2) - max(sx1, ox1))
    inter_y = max(0, min(sy2, oy2) - max(sy1, oy1))
    inter_area = inter_x * inter_y
    union_area = s_area + o_area - inter_area
    iou = inter_area / max(union_area, 1e-6)

    if iou > 0.3:
        overlap_desc = "significantly overlapping"
    elif iou > 0.1:
        overlap_desc = "partially overlapping"
    elif inter_area > 0:
        overlap_desc = "slightly touching"
    else:
        overlap_desc = "not touching"

    # ── Relative scale ───────────────────────────────────────────────
    scale_ratio = s_area / o_area
    if scale_ratio > 3.0:
        scale_desc = f"much larger than the {obj}"
    elif scale_ratio > 1.5:
        scale_desc = f"larger than the {obj}"
    elif scale_ratio < 0.33:
        scale_desc = f"much smaller than the {obj}"
    elif scale_ratio < 0.67:
        scale_desc = f"smaller than the {obj}"
    else:
        scale_desc = f"similar in size to the {obj}"

    # ── Contact / proximity ──────────────────────────────────────────
    # Edge-to-edge gap (0 if overlapping)
    gap_x = max(0, max(sx1, ox1) - min(sx2, ox2))
    gap_y = max(0, max(sy1, oy1) - min(sy2, oy2))
    edge_gap = (gap_x**2 + gap_y**2) ** 0.5

    if edge_gap == 0 and iou > 0.05:
        proximity = "in physical contact"
    elif edge_gap < 0.05:
        proximity = "very close (near-contact)"
    elif edge_gap < 0.15:
        proximity = "nearby"
    else:
        proximity = "separated"

    # ── Containment ──────────────────────────────────────────────────
    # How much of the smaller object is inside the larger
    containment = inter_area / min(s_area, o_area) if min(s_area, o_area) > 1e-6 else 0
    contain_note = ""
    if containment > 0.8:
        smaller = obj if o_area < s_area else subj
        larger = subj if o_area < s_area else obj
        contain_note = f" The {smaller} is mostly contained within the {larger}."

    return (
        f"Object detection: the {subj} is {pos_desc} the {obj}, "
        f"{overlap_desc}, {proximity}, and {scale_desc}.{contain_note} "
    )


# ── Action/attribute triple verification ─────────────────────────────────


def verify_action_triple(
    subj: str,
    rel: str,
    obj: str,
    kb: dict,
    pil_image: Image.Image,
    n_questions: int = 3,
    geo_family: str | None = None,
    geo_prereq: bool | None = None,
) -> tuple[bool | None, int, int, int, bool]:
    """Verify an action/attribute triple using crop-based VQA with multi-question voting.

    Asks 2 standard yes/no questions and 1 contrastive forced-choice question
    (with A/B position randomization to avoid bias). When geometry evidence is
    available, it is included in the VQA prompts to ground the VLM's decision.

    Args:
        subj: Subject entity.
        rel: Relation verb.
        obj: Object entity.
        kb: Visual KB dict.
        pil_image: Full PIL image.
        n_questions: Total number of VQA questions (default 3).
        geo_family: Action geometry family (e.g. "grasping", "mounting") if detected.
        geo_prereq: Geometry prerequisite result (True/False/None).

    Returns:
        Tuple of (verdict, yes_votes, no_votes, total_votes, contrastive_no).
        verdict is True (supported), False (hallucinated), or None (uncertain).
    """
    subj_box = find_best_bbox_from_kb(subj, kb)
    obj_box = find_best_bbox_from_kb(obj, kb)

    using_crop = bool(subj_box and obj_box)
    if using_crop:
        crop = crop_to_bboxes(pil_image, subj_box, obj_box, padding=CROP_PADDING)
        region_hint = "this cropped region"
    else:
        crop = pil_image
        region_hint = "the full image"
    crop_b64 = encode_b64(crop)

    # Build geometry evidence hint for VQA prompts
    geo_hint = _build_geo_hint(
        subj, rel, obj, geo_family, geo_prereq,
        subj_box=subj_box, obj_box=obj_box,
    )

    # Build counterfactual for contrastive question
    counterfactual_rel = COUNTERFACTUAL_MAP.get(rel.lower(), f"not {rel}")
    ab_flip = hash(f"{subj}{rel}{obj}") % 2 == 1
    if ab_flip:
        opt_a, opt_b = counterfactual_rel, rel
    else:
        opt_a, opt_b = rel, counterfactual_rel

    contrastive_q = (
        f"Look at {region_hint} carefully. Which description is more accurate?\n"
        f"(A) The {subj} is {opt_a} the {obj}\n"
        f"(B) The {subj} is {opt_b} the {obj}\n"
        f"Answer with ONLY the letter A or B."
    )

    question_templates = [
        f"{geo_hint}In this image, is the {subj} {rel} the {obj}? "
        f"Look carefully at {region_hint}. Answer only YES or NO.",
        f"{geo_hint}Does the {subj} appear to be {rel} the {obj} here? "
        f"Observe {region_hint} closely. Answer YES or NO only.",
    ]
    questions_yn = question_templates[: min(n_questions - 1, 2)]

    yes_votes = 0
    no_votes = 0

    # Standard yes/no questions
    for q in questions_yn:
        result = vlm_call(
            [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{crop_b64}"}},
                {"type": "text", "text": q},
            ]}],
            max_tokens=5,
        )
        if not result:
            continue
        r = result.strip().lower()
        if "yes" in r and "no" not in r:
            yes_votes += 1
        elif "no" in r:
            no_votes += 1

    # Contrastive forced-choice question
    contrastive_no = False
    contrastive_result = vlm_call(
        [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{crop_b64}"}},
            {"type": "text", "text": contrastive_q},
        ]}],
        max_tokens=5,
    )
    if contrastive_result:
        cr = contrastive_result.strip().upper()
        chose_a = "A" in cr and "B" not in cr
        chose_b = "B" in cr and "A" not in cr
        if ab_flip:
            if chose_b:
                yes_votes += 1
            elif chose_a:
                no_votes += 1
                contrastive_no = True
        else:
            if chose_a:
                yes_votes += 1
            elif chose_b:
                no_votes += 1
                contrastive_no = True

    # Majority vote
    total = yes_votes + no_votes
    if total == 0:
        verdict = None
    elif yes_votes > no_votes:
        verdict = True
    elif no_votes > yes_votes:
        verdict = False
    else:
        verdict = None

    return (verdict, yes_votes, no_votes, total, contrastive_no)


# ── Correct spatial relation (KB-first) ──────────────────────────────────


def query_correct_spatial_relation(
    subj: str,
    obj: str,
    kb: dict,
    pil_image: Image.Image,
) -> str | None:
    """Determine the correct spatial relation between subject and object.

    Two-stage cascade:
        1. KB lookup (deterministic geometry — free, no API cost)
        2. VLM query + verification (if KB has no answer)

    Args:
        subj: Subject entity name.
        obj: Object entity name.
        kb: Visual KB dict.
        pil_image: Full PIL image.

    Returns:
        Correct spatial relation string, or None if undetermined.
    """
    if pil_image is None:
        return None

    # Stage 1: KB lookup
    spatial_facts = kb.get("spatial_facts", [])
    kb_triples = _parse_spatial_facts(spatial_facts)
    for kb_s, kb_r, kb_o in kb_triples:
        if entity_matches(subj, kb_s) and entity_matches(obj, kb_o):
            log.debug("  [correct_spatial] KB hit: %s %s %s", kb_s, kb_r, kb_o)
            return kb_r
        if entity_matches(subj, kb_o) and entity_matches(obj, kb_s):
            rev = SPATIAL_OPPOSITES.get(kb_r.lower())
            if rev:
                log.debug("  [correct_spatial] KB hit (reversed): %s", rev)
                return rev

    # Stage 2: VLM query
    subj_box = find_best_bbox_from_kb(subj, kb)
    obj_box = find_best_bbox_from_kb(obj, kb)

    using_crop = bool(subj_box and obj_box)
    if using_crop:
        crop = crop_to_bboxes(pil_image, subj_box, obj_box, padding=CROP_PADDING_WIDE)
        img_b64 = encode_b64(crop)
        region_hint = "this cropped region"
    else:
        img_b64 = encode_b64(pil_image)
        region_hint = "the full image"

    prompt = (
        f"Look at this image. Where is the {obj} relative to the {subj}? "
        f"Reply with ONLY a short spatial phrase "
        f"(e.g. 'to the left of', 'to the right of', 'above', 'below', "
        f"'in front of', 'behind', 'next to'). "
        f"If you cannot determine the relationship clearly, reply 'unknown'."
    )
    raw = vlm_call(
        [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            {"type": "text", "text": prompt},
        ]}],
        max_tokens=15,
    )

    if raw:
        r = raw.strip().strip('"').lower()
        if r != "unknown" and len(r.split()) <= 5:
            # Verify the suggestion
            verify_q = (
                f"Look at {region_hint}. Is the {obj} {r} the {subj}? "
                f"Answer YES or NO only."
            )
            verify_raw = vlm_call(
                [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    {"type": "text", "text": verify_q},
                ]}],
                max_tokens=5,
            )
            if verify_raw and "no" in verify_raw.strip().lower():
                log.debug("  [correct_spatial] VLM suggestion '%s' REJECTED by verify", r)
                return "near"
            log.debug("  [correct_spatial] VLM verified: %s", r)
            return r
    return None


# ── Correct action relation (KB-first, 3-stage) ─────────────────────────


def query_correct_action_relation(
    subj: str,
    wrong_rel: str,
    obj: str,
    kb: dict,
    pil_image: Image.Image,
) -> str | None:
    """Determine the correct relation between subject and object.

    Three-stage cascade with decreasing cost:
        1. KB lookup (free — check spatial_facts and visual_description)
        2. Constrained selection (6 candidates, forced-choice VLM)
        3. Verification (yes/no confirmation of selected relation)

    Args:
        subj: Subject entity name.
        wrong_rel: The incorrect relation that was detected.
        obj: Object entity name.
        kb: Visual KB dict.
        pil_image: Full PIL image.

    Returns:
        Correct relation string, or None if undetermined.
    """
    if pil_image is None:
        return None

    # Stage 1a: KB spatial facts
    spatial_facts = kb.get("spatial_facts", [])
    kb_triples = _parse_spatial_facts(spatial_facts)
    for kb_s, kb_r, kb_o in kb_triples:
        if entity_matches(subj, kb_s) and entity_matches(obj, kb_o):
            log.debug("  [correct_rel] KB hit: %s %s %s", kb_s, kb_r, kb_o)
            return kb_r
        if entity_matches(subj, kb_o) and entity_matches(obj, kb_s):
            rev = SPATIAL_OPPOSITES.get(kb_r.lower())
            if rev:
                log.debug("  [correct_rel] KB hit (reversed): %s", rev)
                return rev

    # Stage 1b: KB visual description extraction
    vis_desc = kb.get("visual_description", "")
    if vis_desc:
        subj_core = core_noun(subj)
        obj_core = core_noun(obj)
        if subj_core in vis_desc.lower() and obj_core in vis_desc.lower():
            extract_prompt = (
                f"From this description, what is the relationship between "
                f"'{subj}' and '{obj}'? Description: \"{vis_desc}\"\n"
                f"Reply with ONLY a 1-4 word relation phrase. If not mentioned, reply 'unclear'."
            )
            raw = llm_call(
                [{"role": "user", "content": extract_prompt}], max_tokens=15
            )
            if raw:
                r = raw.strip().strip("\"'").lower()
                if r not in ("unclear", "unknown", "not mentioned") and 1 <= len(r.split()) <= 4:
                    log.debug("  [correct_rel] visual_description extraction: %s", r)
                    return r

    # Stage 2: Constrained selection via VLM
    subj_box = find_best_bbox_from_kb(subj, kb)
    obj_box = find_best_bbox_from_kb(obj, kb)

    using_crop = bool(subj_box and obj_box)
    if using_crop:
        crop = crop_to_bboxes(pil_image, subj_box, obj_box, padding=0.20)
        img_b64 = encode_b64(crop)
        region_hint = "this cropped region"
    else:
        img_b64 = encode_b64(pil_image)
        region_hint = "the full image"

    _ACTION_CANDIDATES = [
        "standing next to", "looking at", "walking toward", "sitting near",
        "reaching for", "talking to", "facing", "behind",
    ]
    _SPATIAL_CANDIDATES = [
        "to the left of", "to the right of", "above", "below",
        "next to", "in front of", "behind", "on top of",
    ]
    _spatial_kw = {
        "on", "above", "below", "left", "right", "next", "near",
        "behind", "front", "top", "under", "over", "beside",
    }
    is_spatial = any(kw in wrong_rel.lower().split() for kw in _spatial_kw)
    candidates = _SPATIAL_CANDIDATES if is_spatial else _ACTION_CANDIDATES
    candidates = [c for c in candidates if c.lower() != wrong_rel.lower()][:6]

    letters = "ABCDEF"
    options_str = "\n".join(
        f"({letters[i]}) The {subj} is {c} the {obj}"
        for i, c in enumerate(candidates)
    )

    selection_prompt = (
        f"Look at {region_hint} carefully. The caption incorrectly says "
        f"the {subj} is '{wrong_rel}' the {obj}. "
        f"Which of these alternatives BEST describes their actual relationship?\n"
        f"{options_str}\n"
        f"(Z) None of these — the relationship is something else entirely\n\n"
        f"Answer with ONLY the letter."
    )

    raw = vlm_call(
        [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            {"type": "text", "text": selection_prompt},
        ]}],
        max_tokens=5,
    )

    selected_rel = None
    if raw:
        r = raw.strip().upper()
        for i, letter in enumerate(letters[: len(candidates)]):
            if letter in r and "Z" not in r:
                selected_rel = candidates[i]
                break

    # Open-ended fallback if constrained selection failed
    if selected_rel is None:
        open_prompt = (
            f"Look at {region_hint} carefully. "
            f"The caption incorrectly claims the {subj} is '{wrong_rel}' the {obj}. "
            f"In 1-4 words, describe what the {subj} is ACTUALLY doing relative to the {obj}. "
            f"Reply ONLY with the short phrase. If truly unclear, reply 'unclear'."
        )
        raw2 = vlm_call(
            [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                {"type": "text", "text": open_prompt},
            ]}],
            max_tokens=15,
        )
        if raw2:
            r2 = raw2.strip().strip("\"'").lower()
            if r2 not in ("unclear", "unknown") and 1 <= len(r2.split()) <= 4:
                selected_rel = r2

    if selected_rel is None:
        log.debug("  [correct_rel] FAILED: no candidate selected")
        return None

    # Stage 3: Verification
    verify_prompt = (
        f"Look at {region_hint}. Is the {subj} {selected_rel} the {obj}? "
        f"Answer YES or NO only."
    )
    raw3 = vlm_call(
        [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            {"type": "text", "text": verify_prompt},
        ]}],
        max_tokens=5,
    )

    if raw3:
        r3 = raw3.strip().lower()
        if "yes" in r3 and "no" not in r3:
            log.debug("  [correct_rel] VERIFIED: %s", selected_rel)
            return selected_rel
        if "no" in r3:
            log.debug("  [correct_rel] REJECTED: %s — falling back to 'near'", selected_rel)
            return "near"

    log.debug("  [correct_rel] UNVERIFIED (VLM unclear): %s", selected_rel)
    return selected_rel
