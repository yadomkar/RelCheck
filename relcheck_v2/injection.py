"""
RelCheck v2 — Hallucination Injection (Synthetic Test)
========================================================
Utilities for creating controlled synthetic hallucinations by injecting
false relational statements into captions. Used for recall evaluation:
inject known-false relations, then measure whether the verifier detects them.
"""

from __future__ import annotations

import json
import re

from .api import llm_call


# ── Question → Statement conversion ───────────────────────────────────


def question_to_statement(question: str) -> str | None:
    """Convert an R-Bench yes/no question to a false declarative statement.

    Handles two common patterns via regex, with LLM fallback:
      Pattern 1: "Is the/a/an SUBJECT REST" → "The SUBJECT is REST."
      Pattern 2: "Is there a/an SUBJECT REST" → "There is a SUBJECT REST."

    Args:
        question: R-Bench yes/no question string.

    Returns:
        Declarative statement string, or None if conversion failed.
    """
    q = question.strip().rstrip("?").strip()

    # Pattern 1: "Is the/a/an SUBJECT REST"
    m = re.match(r"^[Ii]s (the|a|an) (\w+(?:\s+\w+)?) (.+)", q)
    if m:
        det = m.group(1).capitalize()
        subj = m.group(2)
        rest = m.group(3)
        return f"{det} {subj} is {rest}."

    # Pattern 2: "Is there a/an SUBJECT REST"
    m = re.match(r"^[Ii]s there (a|an) (.+)", q)
    if m:
        return f"There is {m.group(1)} {m.group(2)}."

    # Fallback: LLM conversion
    prompt = (
        "Convert this yes/no question to a short declarative statement. "
        "Output ONLY the statement.\n"
        f'Question: "{question}"'
    )
    resp = llm_call([{"role": "user", "content": prompt}], max_tokens=60)
    return resp.strip().rstrip(".") + "." if resp else None


# ── Triple extraction from question ───────────────────────────────────


def parse_question(question: str) -> tuple[str, str, str]:
    """Extract (subject, relation, object) triple from an R-Bench yes/no question.

    Uses LLM to parse the question into a structured triple. Falls back to
    ('entity', 'unknown', 'entity') on parsing failure.

    Args:
        question: R-Bench yes/no question string.

    Returns:
        Tuple of (subject, relation, object) strings.
    """
    prompt = (
        "Extract the relational triple from this question as JSON.\n"
        f'Question: "{question}"\n'
        'Output: {{"subject": "...", "relation": "...", "object": "..."}}'
        "\nONLY valid JSON, one object."
    )
    raw = llm_call([{"role": "user", "content": prompt}], max_tokens=80)
    try:
        clean = re.sub(r"```[a-z]*\n?", "", raw or "").replace("```", "").strip()
        t = json.loads(clean)
        return (
            t.get("subject", "entity").strip(),
            t.get("relation", "unknown").strip(),
            t.get("object", "entity").strip(),
        )
    except Exception:
        return ("entity", "unknown", "entity")


# ── Relation type classification ──────────────────────────────────────

_SPATIAL_KEYWORDS: frozenset[str] = frozenset({
    "left", "right", "above", "below", "behind", "front", "under",
    "next to", "beside", "on top", "beneath", "near", "between",
})

_ACTION_KEYWORDS: frozenset[str] = frozenset({
    "riding", "holding", "eating", "carrying", "wearing", "sitting",
    "standing", "walking", "playing", "using", "looking", "touching",
})


def classify_rel_type(question: str) -> str:
    """Classify a question's relation as SPATIAL, ACTION, or ATTRIBUTE.

    Uses simple keyword matching against the question text to categorize
    relation types. Used for per-relation-type evaluation breakdown.

    Args:
        question: R-Bench yes/no question string.

    Returns:
        One of "SPATIAL", "ACTION", or "ATTRIBUTE".
    """
    q = question.lower()
    if any(w in q for w in _SPATIAL_KEYWORDS):
        return "SPATIAL"
    if any(w in q for w in _ACTION_KEYWORDS):
        return "ACTION"
    return "ATTRIBUTE"
