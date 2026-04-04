"""
RelCheck v2 — Spatial Geometry
================================
Deterministic spatial relation verification from bounding box centroids.
Zero hallucination risk — pure geometry, no model involved.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from .config import (
    NEAR_DISTANCE_THRESHOLD,
    SPATIAL_DEADZONE,
    SPATIAL_OPPOSITES,
)
from .entity import core_noun
from .types import BBox


# ── Regex for extracting spatial triples from text ───────────────────────

SPATIAL_TRIPLE_RE = re.compile(
    r"([a-z][a-z\s]{1,25}?)\s+(?:is\s+)?(?:to\s+the\s+)?"
    r"(left|right|above|below|on top of|under|over|in front of|behind)"
    r"(?:\s+of)?\s+([a-z][a-z\s]{1,25})",
    re.IGNORECASE,
)


# ── Core spatial verdict ─────────────────────────────────────────────────

def spatial_verdict(
    subj_box: BBox,
    obj_box: BBox,
    relation: str,
) -> bool | None:
    """Deterministic spatial relation check from bounding box centroids.

    Args:
        subj_box: Subject bounding box [x1, y1, x2, y2] normalized.
        obj_box: Object bounding box [x1, y1, x2, y2] normalized.
        relation: Relation word or phrase (e.g., "left of", "above").

    Returns:
        True  — relation is geometrically supported
        False — relation is geometrically contradicted
        None  — ambiguous (within SPATIAL_DEADZONE), needs VQA fallback
    """
    cx_s = (subj_box[0] + subj_box[2]) / 2
    cy_s = (subj_box[1] + subj_box[3]) / 2
    cx_o = (obj_box[0] + obj_box[2]) / 2
    cy_o = (obj_box[1] + obj_box[3]) / 2

    r = relation.lower().strip()
    T = SPATIAL_DEADZONE

    # ── Left of ──
    if r in ("left of", "to the left of", "to the left"):
        if cx_s < cx_o - T:
            return True
        if cx_s > cx_o + T:
            return False
        return None

    # ── Right of ──
    if r in ("right of", "to the right of", "to the right"):
        if cx_s > cx_o + T:
            return True
        if cx_s < cx_o - T:
            return False
        return None

    # ── Above / over ──
    if r in ("above", "over"):
        if cy_s < cy_o - T:
            return True
        if cy_s > cy_o + T:
            return False
        return None

    # ── Below / under / beneath ──
    if r in ("below", "under", "beneath", "underneath"):
        if cy_s > cy_o + T:
            return True
        if cy_s < cy_o - T:
            return False
        return None

    # ── On top of / on ──
    if r in ("on top of", "on"):
        horiz_overlap = subj_box[0] < obj_box[2] and subj_box[2] > obj_box[0]
        if cy_s < cy_o - T and horiz_overlap:
            return True
        if cy_s > cy_o + T:
            return False
        return None

    # ── In front of ──
    if r == "in front of":
        if cy_s > cy_o + T:
            return True
        if cy_s < cy_o - T:
            return False
        return None

    # ── Behind / in back of ──
    if r in ("behind", "in back of"):
        if cy_s < cy_o - T:
            return True
        if cy_s > cy_o + T:
            return False
        return None

    # ── Near / next to / beside ──
    if r in ("near", "next to", "beside"):
        dist = math.hypot(cx_s - cx_o, cy_s - cy_o)
        img_diag = math.hypot(1.0, 1.0)  # normalized coords
        rel_dist = dist / img_diag if img_diag > 0 else 1.0
        if rel_dist < NEAR_DISTANCE_THRESHOLD:
            return True
        else:
            return False

    # ── Far from ──
    if r in ("far from", "away from"):
        dist = math.hypot(cx_s - cx_o, cy_s - cy_o)
        img_diag = math.hypot(1.0, 1.0)
        rel_dist = dist / img_diag if img_diag > 0 else 1.0
        if rel_dist >= NEAR_DISTANCE_THRESHOLD:
            return True
        else:
            return False

    # ── Inside ──
    if r == "inside":
        contained = (
            subj_box[0] >= obj_box[0] - 0.05
            and subj_box[2] <= obj_box[2] + 0.05
            and subj_box[1] >= obj_box[1] - 0.05
            and subj_box[3] <= obj_box[3] + 0.05
        )
        if contained:
            return True
        return None

    # ── Unrecognized spatial relation → VQA fallback ──
    return None


# ── Pairwise spatial facts from detections ───────────────────────────────

def compute_spatial_facts(
    detections: list[tuple[str, float, BBox]],
    max_detections: int = 20,
) -> list[str]:
    """Compute pairwise spatial relationships from detections.

    Args:
        detections: List of (label, score, bbox) tuples from GroundingDINO.
        max_detections: Maximum number of detections to process.

    Returns:
        Human-readable spatial fact strings suitable for KB construction.

    Uses centroid positions and containment checks to produce
    human-readable spatial fact strings.
    """
    # Cap at top-N most confident
    dets = sorted(detections, key=lambda x: -x[1])[:max_detections]
    if not dets:
        return []

    facts: list[str] = []

    # Object counts
    counts = Counter(label for label, _, _ in dets)
    for label, count in counts.items():
        verb = "are" if count > 1 else "is"
        facts.append(f"There {verb} {count} '{label}'")

    # Pairwise spatial
    labels = list(counts.keys())
    for i, l1 in enumerate(labels):
        for j, l2 in enumerate(labels):
            if i >= j:
                continue
            # Best bbox for each label
            _, b1 = max(
                [(s, b) for l, s, b in dets if l == l1], key=lambda x: x[0]
            )
            _, b2 = max(
                [(s, b) for l, s, b in dets if l == l2], key=lambda x: x[0]
            )
            c1x = (b1[0] + b1[2]) / 2
            c1y = (b1[1] + b1[3]) / 2
            c2x = (b2[0] + b2[2]) / 2
            c2y = (b2[1] + b2[3]) / 2

            # Containment check
            contained = (
                b1[0] >= b2[0] - 0.05
                and b1[2] <= b2[2] + 0.05
                and b1[1] >= b2[1] - 0.05
                and b1[3] <= b2[3] + 0.05
            )
            if contained:
                facts.append(f"'{l1}' is on/inside '{l2}'")
                continue

            # Direction from centroid difference
            dx, dy = c1x - c2x, c1y - c2y
            if abs(dx) > abs(dy):
                pos = "to the left of" if dx < 0 else "to the right of"
            else:
                pos = "above" if dy < 0 else "below"
            facts.append(f"'{l1}' is {pos} '{l2}'")

    return facts


# ── Spatial triple extraction from text ──────────────────────────────────

def extract_spatial_triples(text: str) -> list[tuple[str, str, str]]:
    """Extract (subject, relation, object) spatial triples from free text.

    Args:
        text: Input text (caption or KB fact).

    Returns:
        List of (subj, rel, obj) tuples, all lowercase.
    """
    triples = []
    for m in SPATIAL_TRIPLE_RE.finditer(text.lower()):
        subj = m.group(1).strip().rstrip(" ,;")
        rel = m.group(2).strip()
        obj_ = m.group(3).strip().rstrip(" ,;.")
        triples.append((subj, rel, obj_))
    return triples


def parse_spatial_facts(spatial_facts: list[str]) -> list[tuple[str, str, str]]:
    """Parse KB spatial fact strings into (entity1, relation, entity2) triples.

    Args:
        spatial_facts: List of spatial fact strings from compute_spatial_facts.

    Returns:
        List of (subj, rel, obj) triples extracted from facts.
    """
    triples = []
    for fact in spatial_facts:
        found = extract_spatial_triples(fact)
        triples.extend(found)
    return triples


# ── Contradiction detection ──────────────────────────────────────────────

def check_spatial_contradictions(
    caption: str,
    spatial_facts: list[str],
) -> list[str]:
    """Find caption claims that contradict deterministic spatial KB facts.

    Args:
        caption: Input caption text to check.
        spatial_facts: List of verified spatial KB facts.

    Returns:
        List of human-readable contradiction strings.
    """
    cap_triples = extract_spatial_triples(caption)
    kb_triples = parse_spatial_facts(spatial_facts)

    contradictions: list[str] = []

    for c_subj, c_rel, c_obj in cap_triples:
        for k_subj, k_rel, k_obj in kb_triples:
            # Check if same entity pair (either order)
            subj_match = (
                core_noun(c_subj) == core_noun(k_subj)
                and core_noun(c_obj) == core_noun(k_obj)
            ) or (
                core_noun(c_subj) == core_noun(k_obj)
                and core_noun(c_obj) == core_noun(k_subj)
            )
            if not subj_match:
                continue

            # Check if relations are opposites
            opposite = SPATIAL_OPPOSITES.get(c_rel)
            if opposite and opposite == k_rel:
                contradictions.append(
                    f"Caption says '{c_subj} {c_rel} {c_obj}' "
                    f"but geometry shows: {k_subj} {k_rel} {k_obj}"
                )

    return contradictions
