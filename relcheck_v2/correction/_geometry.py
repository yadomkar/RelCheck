"""
RelCheck v2 — Action Geometry Verification
============================================
Physical-geometric prerequisites for action relations.
Uses bounding box overlap and optional ViTPose keypoints to
pre-screen actions before expensive VQA.

Families:
    mounting     — subject above object with horizontal overlap
    containment  — subject bbox inside object bbox
    adjacency    — bboxes within gap threshold
    grasping     — wrist keypoint near object bbox (requires ViTPose)
    consuming    — nose keypoint near object bbox (requires ViTPose)
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from .._logging import log
from ..config import (
    MOUNTING_TOP_RATIO,
    CONTAINMENT_OVERLAP_MIN,
    ADJACENCY_GAP_RATIO,
    KEYPOINT_CONFIDENCE_MIN,
)
from ..types import BBox

# ── Action family taxonomy ───────────────────────────────────────────────

ACTION_GEOMETRY_TAXONOMY: dict[str, dict] = {
    "mounting": {
        "verbs": frozenset({
            "riding", "sitting on", "standing on", "straddling",
            "mounted on", "perched on", "atop", "on top of",
            "perching on", "seated on", "crouching on",
        }),
        "geometric_rule": "subject_above_object",
        "needs_keypoints": False,
    },
    "containment": {
        "verbs": frozenset({
            "inside", "in", "enclosed by", "covered by",
            "contained in", "within", "trapped in", "wrapped in",
        }),
        "geometric_rule": "subject_inside_object",
        "needs_keypoints": False,
    },
    "adjacency": {
        "verbs": frozenset({
            "next to", "beside", "near", "alongside", "adjacent to",
            "close to", "leaning on", "leaning against",
        }),
        "geometric_rule": "bboxes_close",
        "needs_keypoints": False,
    },
    "grasping": {
        "verbs": frozenset({
            "holding", "carrying", "picking up", "pulling", "pushing",
            "grabbing", "gripping", "lifting", "dragging", "clutching",
            "catching", "throwing", "tossing",
        }),
        "geometric_rule": "wrist_near_object",
        "needs_keypoints": True,
    },
    "consuming": {
        "verbs": frozenset({
            "eating", "drinking", "tasting", "licking", "biting",
            "sipping", "chewing", "feeding on",
        }),
        "geometric_rule": "nose_near_object",
        "needs_keypoints": True,
    },
}

# Pre-computed verb → family lookup
_VERB_TO_FAMILY: dict[str, str] = {}
for _fam, _spec in ACTION_GEOMETRY_TAXONOMY.items():
    for _v in _spec["verbs"]:
        _VERB_TO_FAMILY[_v] = _fam

# COCO keypoint indices
KP_NOSE: int = 0
KP_LEFT_WRIST: int = 9
KP_RIGHT_WRIST: int = 10


# ── Family classification ────────────────────────────────────────────────


@lru_cache(maxsize=256)
def _lemmatize(verb: str) -> str:
    """Lemmatize a verb using spaCy (cached). Falls back to input on error."""
    try:
        import spacy
        nlp = spacy.blank("en")
        doc = nlp(verb)
        return doc[0].lemma_ if doc else verb
    except Exception:
        return verb


def classify_action_family(relation_verb: str) -> str | None:
    """Map a relation verb to its physical family (or None if no rule exists).

    Three-level cascade:
        1. Exact match on input form
        2. Multi-word verb containment
        3. spaCy lemmatization (catches inflected forms like "rode" → "ride")

    Args:
        relation_verb: The relation verb string (e.g. "riding", "sitting on").

    Returns:
        Family name ("mounting", "containment", etc.) or None.
    """
    rel = relation_verb.strip().lower()
    # Level 1: exact match
    if rel in _VERB_TO_FAMILY:
        return _VERB_TO_FAMILY[rel]
    # Level 2: multi-word verb containment
    for verb, fam in _VERB_TO_FAMILY.items():
        if len(verb.split()) >= 2 and verb in rel:
            return fam
    # Level 3: lemmatize and retry
    lemma = _lemmatize(rel.split()[0]) if rel else rel
    if lemma != rel and lemma in _VERB_TO_FAMILY:
        return _VERB_TO_FAMILY[lemma]
    return None


# ── ViTPose integration ──────────────────────────────────────────────────


def get_person_keypoints(
    pil_image: "Image.Image",
    person_box_norm: BBox,
) -> dict | None:
    """Run ViTPose on a detected person to get 17 COCO keypoints.

    Args:
        pil_image: PIL Image of the full scene.
        person_box_norm: Normalized bounding box [x1, y1, x2, y2] of the person.

    Returns:
        Dict with 'keypoints' (17x2 normalized pixel coords) and
        'scores' (17 confidence values), or None if detection fails.
    """
    try:
        import torch
        from transformers import ViTPoseProcessor, ViTPoseForPoseEstimation

        vitpose_model = ViTPoseForPoseEstimation.from_pretrained(
            "google/vitpose-base-simple-coco", device_map="auto"
        )
        vitpose_processor = ViTPoseProcessor.from_pretrained(
            "google/vitpose-base-simple-coco"
        )
    except Exception:
        return None

    W, H = pil_image.size
    x1, y1, x2, y2 = person_box_norm
    coco_box = [x1 * W, y1 * H, (x2 - x1) * W, (y2 - y1) * H]

    try:
        inputs = vitpose_processor(
            pil_image, boxes=[[coco_box]], return_tensors="pt"
        ).to(vitpose_model.device)

        with torch.no_grad():
            outputs = vitpose_model(**inputs)

        results = vitpose_processor.post_process_pose_estimation(
            outputs, boxes=[[coco_box]]
        )

        if results and results[0]:
            kp = results[0][0]
            keypoints = kp["keypoints"].cpu().numpy()
            scores = kp["scores"].cpu().numpy()
            keypoints[:, 0] /= W
            keypoints[:, 1] /= H
            return {"keypoints": keypoints, "scores": scores}
    except Exception as e:
        log.debug("ViTPose error: %s", e)
    return None


# ── Geometric prerequisite checks ────────────────────────────────────────


def check_action_geometry(
    family: str,
    subj_box: BBox,
    obj_box: BBox,
    keypoints: dict | None = None,
) -> bool | None:
    """Test geometric prerequisite for an action family.

    Args:
        family: Action family name (from ACTION_GEOMETRY_TAXONOMY).
        subj_box: Normalized bounding box of the subject.
        obj_box: Normalized bounding box of the object.
        keypoints: Optional ViTPose keypoint dict (required for grasping/consuming).

    Returns:
        True (prerequisite met), False (violated), or None (cannot check).
    """
    sx1, sy1, sx2, sy2 = subj_box
    ox1, oy1, ox2, oy2 = obj_box

    o_h = oy2 - oy1
    o_w = ox2 - ox1
    s_h = sy2 - sy1

    if family == "mounting":
        top_region = oy1 + MOUNTING_TOP_RATIO * o_h
        subject_bottom_in_top = sy2 <= top_region + 0.05
        x_overlap = min(sx2, ox2) - max(sx1, ox1)
        has_x_overlap = x_overlap > 0.02 * max(o_w, 0.01)
        return subject_bottom_in_top and has_x_overlap

    if family == "containment":
        inter_x = max(0, min(sx2, ox2) - max(sx1, ox1))
        inter_y = max(0, min(sy2, oy2) - max(sy1, oy1))
        inter_area = inter_x * inter_y
        subj_area = max((sx2 - sx1) * (sy2 - sy1), 1e-6)
        containment_ratio = inter_area / subj_area
        return containment_ratio > CONTAINMENT_OVERLAP_MIN

    if family == "adjacency":
        gap_x = max(0, max(sx1, ox1) - min(sx2, ox2))
        gap_y = max(0, max(sy1, oy1) - min(sy2, oy2))
        gap = (gap_x**2 + gap_y**2) ** 0.5
        avg_size = ((s_h + o_h) / 2 + ((sx2 - sx1) + o_w) / 2) / 2
        return gap < ADJACENCY_GAP_RATIO * max(avg_size, 0.01)

    # Keypoint-based families require pose data
    if keypoints is None:
        return None

    kp_xy = keypoints["keypoints"]
    kp_sc = keypoints["scores"]

    if family == "grasping":
        margin_x = max(0.5 * o_w, 0.03)
        margin_y = max(0.5 * o_h, 0.03)
        obj_expanded = [
            ox1 - margin_x, oy1 - margin_y,
            ox2 + margin_x, oy2 + margin_y,
        ]
        for wrist_idx in [KP_LEFT_WRIST, KP_RIGHT_WRIST]:
            if kp_sc[wrist_idx] < KEYPOINT_CONFIDENCE_MIN:
                continue
            wx, wy = kp_xy[wrist_idx]
            if (obj_expanded[0] <= wx <= obj_expanded[2]
                    and obj_expanded[1] <= wy <= obj_expanded[3]):
                return True
        return False

    if family == "consuming":
        if kp_sc[KP_NOSE] < KEYPOINT_CONFIDENCE_MIN:
            return None
        nx, ny = kp_xy[KP_NOSE]
        margin_x = max(0.75 * o_w, 0.04)
        margin_y = max(0.75 * o_h, 0.04)
        obj_expanded = [
            ox1 - margin_x, oy1 - margin_y,
            ox2 + margin_x, oy2 + margin_y,
        ]
        if (obj_expanded[0] <= nx <= obj_expanded[2]
                and obj_expanded[1] <= ny <= obj_expanded[3]):
            return True
        return False

    return True
