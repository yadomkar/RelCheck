"""
RelCheck v2 — Correction Utilities
====================================
Text quality checks and helper functions shared by enrichment
and surgical correction pipelines.
"""

from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache

from nltk.corpus import wordnet as wn
from rapidfuzz import fuzz

from ..config import SPATIAL_OPPOSITES


# ── Entity normalization ─────────────────────────────────────────────────

_ENTITY_STOP: frozenset[str] = frozenset([
    "a", "an", "the", "some",
    "is", "are", "was", "were", "be", "been", "being",
    "sits", "sit", "standing", "stand", "positioned", "position",
    "placed", "place", "seen", "located", "appears", "appear",
    "lying", "lies", "lay", "resting", "rest",
])


def normalize_entity(text: str) -> str:
    """Lowercase and strip leading articles for fuzzy matching.

    Args:
        text: Raw entity string (e.g. "the large dog").

    Returns:
        Normalized string (e.g. "large dog").
    """
    if not text:
        return ""
    text = text.lower().strip()
    for art in ("a ", "an ", "the ", "some "):
        if text.startswith(art):
            text = text[len(art):]
    return text.strip()


def core_noun(text: str) -> str:
    """Extract core noun from entity span, stripping filler words.

    Removes leading/trailing stop words and truncates to 3 words max.

    Args:
        text: Entity string (e.g. "a man sitting on").

    Returns:
        Core noun phrase (e.g. "man").
    """
    words = normalize_entity(text).split()
    while words and words[0] in _ENTITY_STOP:
        words = words[1:]
    while words and words[-1] in _ENTITY_STOP:
        words = words[:-1]
    return " ".join(words[:3]).strip()


_FUZZY_MATCH_THRESHOLD: int = 80  # rapidfuzz partial_ratio threshold


def entity_matches(cap_entity: str, kb_entity: str) -> bool:
    """Fuzzy match using core noun extraction, substring containment, and rapidfuzz.

    Three-level cascade:
        1. Exact substring containment (fast, free)
        2. rapidfuzz partial_ratio (handles plurals, typos, word reordering)

    Args:
        cap_entity: Entity string from caption.
        kb_entity: Entity string from knowledge base.

    Returns:
        True if the entities match at any level.
    """
    core_cap = core_noun(cap_entity)
    core_kb = core_noun(kb_entity)
    if not core_cap or not core_kb:
        return False
    # Level 1: substring containment
    if (core_kb in core_cap) or (core_cap in core_kb):
        return True
    # Level 2: rapidfuzz (catches plurals like dog/dogs, bike/biking)
    return fuzz.partial_ratio(core_cap, core_kb) >= _FUZZY_MATCH_THRESHOLD


# ── Text quality checks ─────────────────────────────────────────────────


def has_garble(text: str) -> bool:
    """Detect artifacts from bad LLM correction output."""
    t = text.lower()
    if re.search(r"\w+\s+it\s+up", t):
        return True
    if "mat it" in t:
        return True
    if re.search(r"\S{30,}", t):
        return True
    spatial_garbles = [
        r"\bright the\b",
        r"\bleft a\b",
        r"\bbelow floor\b",
        r"\babove the left\b",
        r"\bbelow the left\b",
        r"\bin front of front\b",
    ]
    for pat in spatial_garbles:
        if re.search(pat, t):
            return True
    words = t.split()
    for i in range(len(words) - 3):
        bigram = f"{words[i]} {words[i+1]}"
        for j in range(i + 2, min(i + 6, len(words) - 1)):
            if f"{words[j]} {words[j+1]}" == bigram:
                return True
    action_garbles = [
        "holding hat on head",
        "holding jacket over shoulder",
        "controlling the dogs right",
        "to the right of right",
        "to the left of the left",
    ]
    for pat in action_garbles:
        if pat in t:
            return True
    return False


@lru_cache(maxsize=512)
def _wordnet_antonyms(word: str) -> frozenset[str]:
    """Look up antonyms via WordNet (cached for performance).

    Args:
        word: A single word to look up.

    Returns:
        Frozenset of known antonyms (may be empty).
    """
    antonyms: set[str] = set()
    for synset in wn.synsets(word):
        for lemma in synset.lemmas():
            for ant in lemma.antonyms():
                antonyms.add(ant.name().replace("_", " "))
    return frozenset(antonyms)


def _get_antonym(word: str) -> str | None:
    """Get the antonym of a word, checking domain overrides first, then WordNet.

    Args:
        word: A single lowercase word.

    Returns:
        The antonym string, or None if no antonym is known.
    """
    # Domain-specific spatial opposites take priority
    if word in SPATIAL_OPPOSITES:
        return SPATIAL_OPPOSITES[word]
    # WordNet lookup
    ants = _wordnet_antonyms(word)
    return next(iter(ants), None)


def has_self_contradiction(original_cap: str, corrected_cap: str) -> bool:
    """Detect contradictory replacements in corrected caption.

    Uses SPATIAL_OPPOSITES for domain-specific pairs (above/below, etc.)
    and falls back to nltk WordNet for general antonyms (open/close,
    push/pull, etc.) — no hand-maintained action verb dict needed.

    Args:
        original_cap: Original caption before correction.
        corrected_cap: Caption after correction.

    Returns:
        True if a self-contradiction is detected.
    """
    orig_words = original_cap.lower().split()
    orig_cnt = Counter(orig_words)
    corr_lower = corrected_cap.lower()
    corr_cnt = Counter(corr_lower.split())

    for word in set(orig_words):
        if len(word) < 4:
            continue
        opp = _get_antonym(word)
        if opp is None:
            continue
        o_c = orig_cnt[word]
        c_c = corr_cnt.get(word, 0)

        if 0 < c_c < o_c and opp in corr_lower:
            return True
        if word in corr_lower and opp in corr_lower:
            return True

    return False


def extract_correct_rel_from_reason(reason_str: str) -> str | None:
    """Parse a stored correct relation from verifier reason strings.

    Looks for patterns like ``geometry shows left`` or
    ``correct relation: 'next to'`` in the reason string.

    Args:
        reason_str: Reason string from a verification result.

    Returns:
        The extracted relation, or None if not found.
    """
    m = re.search(r"geometry shows (\S+)", reason_str or "")
    if m:
        return m.group(1).replace("_", " ")
    m = re.search(r"correct relation: '([^']+)'", reason_str or "")
    if m:
        return m.group(1)
    return None
