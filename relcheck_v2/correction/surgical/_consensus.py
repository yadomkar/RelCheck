"""
RelCheck v2 — Cross-Captioner Consensus & Caption Text Helpers
================================================================
Functions for checking agreement across multiple captioning models
and for resolving entity names between KB and caption text.
"""

from __future__ import annotations

from ...config import ENTITY_SYNONYMS
from .._utils import core_noun


# ── Cross-captioner consensus ───────────────────────────────────────────


def consensus_confirms_triple(
    subj: str,
    rel: str,
    obj: str,
    cross_captions: dict[str, str],
) -> bool:
    """Check if all cross-captioners mention (subj rel obj).

    Args:
        subj: Subject entity.
        rel: Relation.
        obj: Object entity.
        cross_captions: {captioner_name: caption_text} dict.

    Returns:
        True if all captioners confirm the triple.
    """
    if not cross_captions or len(cross_captions) < 2:
        return False

    core_s = core_noun(subj)
    core_o = core_noun(obj)
    rel_lower = rel.lower()

    confirmed = 0
    for cap_text in cross_captions.values():
        if not cap_text:
            continue
        cap_lower = cap_text.lower()
        if rel_lower not in cap_lower:
            continue
        idx = cap_lower.find(rel_lower)
        while idx >= 0:
            context = cap_lower[max(0, idx - 80) : idx + 80]
            subj_near = (
                core_s in context
                or any(s in context for s in ENTITY_SYNONYMS.get(core_s, []))
            )
            obj_near = (
                core_o in context
                or any(s in context for s in ENTITY_SYNONYMS.get(core_o, []))
            )
            if subj_near and obj_near:
                confirmed += 1
                break
            idx = cap_lower.find(rel_lower, idx + 1)

    return confirmed >= len(cross_captions)


# ── Caption text helpers ────────────────────────────────────────────────


def caption_name_for(kb_entity: str, cap_lower: str) -> str:
    """Return the synonym of kb_entity that appears in cap_lower."""
    cn = core_noun(kb_entity)
    if not cn:
        return kb_entity
    if cn in cap_lower:
        return cn
    for syn in ENTITY_SYNONYMS.get(cn, []):
        if syn in cap_lower:
            return syn
    return cn


def relation_already_expressed(
    subj: str, rel: str, obj: str, cap_lower: str,
) -> bool:
    """Check if caption already states subj <rel> obj or equivalent."""
    rel_norm = rel.lower()
    rel_variants: dict[str, list[str]] = {
        "left": ["left"], "right": ["right"],
        "above": ["above", "on top of"],
        "below": ["below", "beneath", "under"],
        "behind": ["behind", "in back of"],
        "in front of": ["in front of", "front"],
    }
    keywords = rel_variants.get(rel_norm, [rel_norm])
    for kw in keywords:
        if kw not in cap_lower:
            continue
        idx = cap_lower.find(kw)
        context = cap_lower[max(0, idx - 80) : idx + 80]
        cn_s = core_noun(subj)
        cn_o = core_noun(obj)
        if cn_s in context or cn_o in context:
            return True
        for syn in ENTITY_SYNONYMS.get(cn_s, []):
            if syn in context:
                return True
        for syn in ENTITY_SYNONYMS.get(cn_o, []):
            if syn in context:
                return True
    return False


# ── Spatial synonym groups ──────────────────────────────────────────────

_SPATIAL_SYNONYM_GROUPS: list[set[str]] = [
    {"on", "above", "on top of", "over"},
    {"under", "below", "beneath", "underneath"},
    {"left", "to the left", "to the left of"},
    {"right", "to the right", "to the right of"},
    {"in front of", "before"},
    {"behind", "in back of"},
]


def spatial_synonyms(rel: str) -> set[str]:
    """Return the synonym set that contains rel, or {rel} if none."""
    rel_lower = rel.lower().strip()
    for group in _SPATIAL_SYNONYM_GROUPS:
        if rel_lower in group:
            return group
    return {rel_lower}
