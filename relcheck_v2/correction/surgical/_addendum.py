"""
RelCheck v2 — Caption Addendum Builders
=========================================
Append missing spatial facts and visual description facts
to corrected captions without modifying existing content.
"""

from __future__ import annotations

from collections import Counter

from nltk.util import ngrams

from ..._logging import log
from ...api import llm_call
from ...config import ADDENDUM_MAX_WORDS_ADDED, ADDENDUM_SURVIVAL_RATIO
from ...prompts import MISSING_FACTS_PROMPT
from .._vqa import _parse_spatial_facts
from ._consensus import caption_name_for, relation_already_expressed


def build_spatial_addendum(
    corrected_caption: str,
    kb: dict,
    max_facts: int = 5,
) -> tuple[str, int]:
    """Append missing KB spatial facts to caption.

    Args:
        corrected_caption: Current caption text.
        kb: Visual KB dict.
        max_facts: Maximum number of spatial facts to add.

    Returns:
        Tuple of (new_caption, n_facts_added).
    """
    spatial_facts = kb.get("spatial_facts", [])
    if not spatial_facts:
        return corrected_caption, 0

    cap_lower = corrected_caption.lower()
    missing: list[tuple[str, str, str, str, str, str]] = []

    for fact in spatial_facts:
        triples = _parse_spatial_facts([fact])
        if not triples:
            continue
        subj, rel, obj = triples[0]
        if not subj or not obj:
            continue
        if relation_already_expressed(subj, rel, obj, cap_lower):
            continue
        cap_subj = caption_name_for(subj, cap_lower)
        cap_obj = caption_name_for(obj, cap_lower)
        missing.append((subj, rel, obj, fact, cap_subj, cap_obj))
        if len(missing) >= max_facts:
            break

    if not missing:
        return corrected_caption, 0

    fact_phrases = [f"the {cs} is {r} the {co}" for _, r, _, _, cs, co in missing]
    if len(fact_phrases) == 1:
        addendum = f"Spatially, {fact_phrases[0]}."
    elif len(fact_phrases) == 2:
        addendum = f"Spatially, {fact_phrases[0]}, and {fact_phrases[1]}."
    else:
        joined = ", ".join(fact_phrases[:-1]) + f", and {fact_phrases[-1]}"
        addendum = f"Spatially, {joined}."

    new_cap = corrected_caption.rstrip() + " " + addendum
    return new_cap, len(missing)


def add_missing_fact_addendum(
    corrected_caption: str,
    kb: dict,
) -> tuple[str, int]:
    """Insert facts from visual description that are absent from caption.

    Uses LLM to identify up to 2 missing facts and insert them
    at semantically appropriate positions.

    Args:
        corrected_caption: Current caption text.
        kb: Visual KB dict.

    Returns:
        Tuple of (new_caption, n_facts_added).
    """
    visual_desc = kb.get("visual_description", "")
    if not visual_desc or len(visual_desc.strip()) < 20:
        return corrected_caption, 0
    if len(corrected_caption.split()) > 300:
        return corrected_caption, 0

    prompt = MISSING_FACTS_PROMPT.format(
        caption=corrected_caption[:1500],
        visual_description=visual_desc[:1500],
    )
    raw = llm_call([{"role": "user", "content": prompt}], max_tokens=800)
    if not raw:
        return corrected_caption, 0

    result = raw.strip().strip('"').strip("'").strip()
    if result.upper() == "NONE" or not result:
        return corrected_caption, 0

    orig_words = len(corrected_caption.split())
    new_words = len(result.split())
    added = new_words - orig_words

    if added <= 0 or added > ADDENDUM_MAX_WORDS_ADDED:
        return corrected_caption, 0

    # Check token survival (LLM didn't rewrite the whole thing)
    orig_tokens = corrected_caption.lower().split()
    result_lower = result.lower()
    surviving = sum(1 for t in orig_tokens if t in result_lower)
    if surviving / max(len(orig_tokens), 1) < ADDENDUM_SURVIVAL_RATIO:
        return corrected_caption, 0

    # Detect n-gram repetition (bigrams and trigrams)
    result_toks = result.lower().split()
    for n in (2, 3):
        counts = Counter(ngrams(result_toks, n))
        if any(c >= 3 for c in counts.values()):
            return corrected_caption, 0

    # Detect new bigram repetition (not present in original)
    orig_bigrams = set(ngrams(orig_tokens, 2))
    result_bigram_counts = Counter(ngrams(result_toks, 2))
    for bg, count in result_bigram_counts.items():
        if bg not in orig_bigrams and count >= 2:
            return corrected_caption, 0

    return result, 1
