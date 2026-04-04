"""
RelCheck v2 — Triple Extraction
=================================
LLM-based relational triple extraction from caption text,
with keyword fallback for type classification.
"""

from __future__ import annotations

import json_repair

from ..._logging import log
from ...api import llm_call
from ...prompts import TRIPLE_EXTRACT_PROMPT
from ...types import Triple, RelationType


_SPATIAL_WORDS: frozenset[str] = frozenset({
    "left", "right", "above", "below", "behind", "front",
    "beside", "next to", "on top of", "under", "over",
    "in front of", "near", "inside", "outside", "between",
})
_ACTION_WORDS: frozenset[str] = frozenset({
    "riding", "holding", "carrying", "eating", "drinking",
    "wearing", "pushing", "pulling", "walking", "running",
    "sitting", "standing", "playing", "using", "throwing",
    "catching", "driving", "leading",
})


def _classify_relation_type(relation: str) -> str:
    """Keyword-based fallback for relation type classification."""
    rel_lower = relation.lower()
    if any(w in rel_lower for w in _SPATIAL_WORDS):
        return "SPATIAL"
    if any(w in rel_lower for w in _ACTION_WORDS):
        return "ACTION"
    return "ACTION"


def extract_triples(caption: str) -> list[Triple]:
    """Use LLM to extract relational triples from a caption.

    Falls back to keyword-based type classification if the LLM
    does not provide a valid type.

    Args:
        caption: Caption text.

    Returns:
        List of Triple objects.
    """
    prompt = TRIPLE_EXTRACT_PROMPT.format(caption=caption)
    raw = llm_call([{"role": "user", "content": prompt}], max_tokens=600)
    if not raw:
        log.debug("[extract_triples] empty response for: %.80r", caption)
        return []

    log.debug("[extract_triples] raw=%.200r", raw[:200])

    try:
        parsed = json_repair.loads(raw)
    except Exception as e:
        log.debug("[extract_triples] json_repair failed: %s", e)
        return []

    # Unwrap dict wrapper (LLM sometimes wraps list in {"triples": [...]})
    if isinstance(parsed, dict):
        for key in ("triples", "relations", "result", "data", "output"):
            if key in parsed and isinstance(parsed[key], list):
                parsed = parsed[key]
                break
        else:
            return []

    if not isinstance(parsed, list):
        return []

    result: list[Triple] = []
    for t in parsed:
        if not isinstance(t, dict):
            continue
        if not all(k in t for k in ("subject", "relation", "object")):
            continue

        typ = str(t.get("type", "")).upper().strip()
        if typ not in ("SPATIAL", "ACTION", "ATTRIBUTE"):
            typ = _classify_relation_type(t.get("relation", ""))

        result.append(Triple(
            subject=t["subject"].strip(),
            relation=t["relation"].strip(),
            object=t["object"].strip(),
            rel_type=RelationType(typ),
        ))
    return result
