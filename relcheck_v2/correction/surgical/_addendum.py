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
from .._metrics import (
    REJECT_LOW_SURVIVAL, REJECT_NEW_BIGRAM_REPETITION,
    REJECT_NGRAM_REPETITION, REJECT_TOO_MANY_WORDS,
    MetricsCollector,
)
from .._vqa import _parse_spatial_facts
from ._consensus import caption_name_for, relation_already_expressed


def build_spatial_addendum(
    corrected_caption: str,
    kb: dict,
    max_facts: int = 5,
    img_id: str = "",
    metrics: MetricsCollector | None = None,
) -> tuple[str, int]:
    """Append missing KB spatial facts to caption.

    Args:
        corrected_caption: Current caption text.
        kb: Visual KB dict.
        max_facts: Maximum number of spatial facts to add.
        img_id: Image identifier (used only when metrics is not None).
        metrics: Optional metrics collector for path logging.

    Returns:
        Tuple of (new_caption, n_facts_added).
    """
    spatial_facts = kb.get("spatial_facts", [])
    kb_spatial_facts_available = len(spatial_facts)

    if not spatial_facts:
        if metrics is not None:
            metrics.record_spatial_addendum(
                img_id, n_facts_added=0,
                kb_spatial_facts_available=0,
                n_already_expressed=0, n_novel=0,
            )
        return corrected_caption, 0

    cap_lower = corrected_caption.lower()
    missing: list[tuple[str, str, str, str, str, str]] = []
    n_already_expressed = 0

    for fact in spatial_facts:
        triples = _parse_spatial_facts([fact])
        if not triples:
            continue
        subj, rel, obj = triples[0]
        if not subj or not obj:
            continue
        if relation_already_expressed(subj, rel, obj, cap_lower):
            n_already_expressed += 1
            continue
        cap_subj = caption_name_for(subj, cap_lower)
        cap_obj = caption_name_for(obj, cap_lower)
        missing.append((subj, rel, obj, fact, cap_subj, cap_obj))
        if len(missing) >= max_facts:
            break

    n_novel = len(missing)

    if not missing:
        if metrics is not None:
            metrics.record_spatial_addendum(
                img_id, n_facts_added=0,
                kb_spatial_facts_available=kb_spatial_facts_available,
                n_already_expressed=n_already_expressed, n_novel=0,
            )
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
    n_facts_added = len(missing)

    if metrics is not None:
        metrics.record_spatial_addendum(
            img_id, n_facts_added=n_facts_added,
            kb_spatial_facts_available=kb_spatial_facts_available,
            n_already_expressed=n_already_expressed, n_novel=n_novel,
        )

    return new_cap, n_facts_added


def add_missing_fact_addendum(
    corrected_caption: str,
    kb: dict,
    img_id: str = "",
    metrics: MetricsCollector | None = None,
) -> tuple[str, int]:
    """Insert facts from visual description that are absent from caption.

    Uses LLM to identify up to 2 missing facts and insert them
    at semantically appropriate positions.

    Args:
        corrected_caption: Current caption text.
        kb: Visual KB dict.
        img_id: Image identifier (used only when metrics is not None).
        metrics: Optional metrics collector for path logging.

    Returns:
        Tuple of (new_caption, n_facts_added).
    """
    visual_desc = kb.get("visual_description", "")
    visual_desc_too_short = not visual_desc or len(visual_desc.strip()) < 20

    if visual_desc_too_short:
        if metrics is not None:
            metrics.record_missing_fact_addendum(
                img_id,
                llm_returned_facts=False,
                n_words_added=0,
                accepted=False,
                rejection_reason=None,
                visual_description_input=visual_desc or "",
                visual_description_too_short=True,
            )
        return corrected_caption, 0

    if len(corrected_caption.split()) > 300:
        if metrics is not None:
            metrics.record_missing_fact_addendum(
                img_id,
                llm_returned_facts=False,
                n_words_added=0,
                accepted=False,
                rejection_reason=None,
                visual_description_input=visual_desc,
                visual_description_too_short=False,
            )
        return corrected_caption, 0

    prompt = MISSING_FACTS_PROMPT.format(
        caption=corrected_caption[:1500],
        visual_description=visual_desc[:1500],
    )
    raw = llm_call([{"role": "user", "content": prompt}], max_tokens=800)
    if not raw:
        if metrics is not None:
            metrics.record_missing_fact_addendum(
                img_id,
                llm_returned_facts=False,
                n_words_added=0,
                accepted=False,
                rejection_reason=None,
                visual_description_input=visual_desc,
                visual_description_too_short=False,
            )
        return corrected_caption, 0

    result = raw.strip().strip('"').strip("'").strip()
    if result.upper() == "NONE" or not result:
        if metrics is not None:
            metrics.record_missing_fact_addendum(
                img_id,
                llm_returned_facts=False,
                n_words_added=0,
                accepted=False,
                rejection_reason=None,
                visual_description_input=visual_desc,
                visual_description_too_short=False,
            )
        return corrected_caption, 0

    # LLM returned something — now evaluate quality
    llm_returned_facts = True
    orig_words = len(corrected_caption.split())
    new_words = len(result.split())
    added = new_words - orig_words
    rejection_reason: str | None = None

    if added <= 0 or added > ADDENDUM_MAX_WORDS_ADDED:
        rejection_reason = REJECT_TOO_MANY_WORDS
        if metrics is not None:
            metrics.record_missing_fact_addendum(
                img_id,
                llm_returned_facts=llm_returned_facts,
                n_words_added=max(added, 0),
                accepted=False,
                rejection_reason=rejection_reason,
                visual_description_input=visual_desc,
                visual_description_too_short=False,
            )
        return corrected_caption, 0

    # Check token survival (LLM didn't rewrite the whole thing)
    orig_tokens = corrected_caption.lower().split()
    result_lower = result.lower()
    surviving = sum(1 for t in orig_tokens if t in result_lower)
    if surviving / max(len(orig_tokens), 1) < ADDENDUM_SURVIVAL_RATIO:
        rejection_reason = REJECT_LOW_SURVIVAL
        if metrics is not None:
            metrics.record_missing_fact_addendum(
                img_id,
                llm_returned_facts=llm_returned_facts,
                n_words_added=added,
                accepted=False,
                rejection_reason=rejection_reason,
                visual_description_input=visual_desc,
                visual_description_too_short=False,
            )
        return corrected_caption, 0

    # Detect n-gram repetition (bigrams and trigrams)
    result_toks = result.lower().split()
    for n in (2, 3):
        counts = Counter(ngrams(result_toks, n))
        if any(c >= 3 for c in counts.values()):
            rejection_reason = REJECT_NGRAM_REPETITION
            if metrics is not None:
                metrics.record_missing_fact_addendum(
                    img_id,
                    llm_returned_facts=llm_returned_facts,
                    n_words_added=added,
                    accepted=False,
                    rejection_reason=rejection_reason,
                    visual_description_input=visual_desc,
                    visual_description_too_short=False,
                )
            return corrected_caption, 0

    # Detect new bigram repetition (not present in original)
    orig_bigrams = set(ngrams(orig_tokens, 2))
    result_bigram_counts = Counter(ngrams(result_toks, 2))
    for bg, count in result_bigram_counts.items():
        if bg not in orig_bigrams and count >= 2:
            rejection_reason = REJECT_NEW_BIGRAM_REPETITION
            if metrics is not None:
                metrics.record_missing_fact_addendum(
                    img_id,
                    llm_returned_facts=llm_returned_facts,
                    n_words_added=added,
                    accepted=False,
                    rejection_reason=rejection_reason,
                    visual_description_input=visual_desc,
                    visual_description_too_short=False,
                )
            return corrected_caption, 0

    # Accepted
    if metrics is not None:
        metrics.record_missing_fact_addendum(
            img_id,
            llm_returned_facts=llm_returned_facts,
            n_words_added=added,
            accepted=True,
            rejection_reason=None,
            visual_description_input=visual_desc,
            visual_description_too_short=False,
        )

    return result, 1
