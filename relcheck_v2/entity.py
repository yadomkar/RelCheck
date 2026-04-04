"""
RelCheck v2 — Entity Matching & Text Utilities
================================================
Noun extraction (spaCy), fuzzy matching (rapidfuzz), edit distance,
and synonym resolution. Replaces hand-rolled implementations scattered
across the original notebook.
"""

from __future__ import annotations

from typing import Optional

import spacy
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from ._logging import log
from .config import ENTITY_SYNONYMS

# ── Lazy spaCy loading ───────────────────────────────────────────────────

_nlp: Optional[spacy.language.Language] = None


def _get_nlp() -> spacy.language.Language:
    """Load en_core_web_sm on first use."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


# ── Noun extraction ──────────────────────────────────────────────────────

def extract_nouns(text: str) -> list[str]:
    """Extract unique noun root lemmas via spaCy noun chunks.

    Args:
        text: Input text to analyze.

    Returns:
        Deduplicated, lowercased root lemmas. Handles multi-word nouns
        like 'baseball bat' better than stopword-filter heuristics.
    """
    doc = _get_nlp()(text)
    return list({chunk.root.lemma_.lower() for chunk in doc.noun_chunks})


# ── Core noun extraction ─────────────────────────────────────────────────

def core_noun(phrase: str) -> str:
    """Extract the head noun from a phrase via spaCy.

    Args:
        phrase: Input phrase to analyze.

    Returns:
        The head noun (lemmatized, lowercased).

    Examples:
        'a large red ball'  → 'ball'
        'the old man'       → 'man'
        'dog sits'          → 'dog'
    """
    if not phrase or not phrase.strip():
        return phrase.lower().strip() if phrase else ""
    doc = _get_nlp()(phrase.strip())
    # Take last non-stop, alphabetic token (typically the head noun)
    tokens = [t for t in doc if not t.is_stop and t.is_alpha]
    if tokens:
        return tokens[-1].lemma_.lower()
    # Fallback: just return last token
    return doc[-1].lemma_.lower() if doc else phrase.lower().strip()


# ── Normalization ────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    """Lowercase and strip leading articles (a, an, the, some).

    Args:
        text: Input text to normalize.

    Returns:
        Normalized text (lowercased, articles stripped).
    """
    if not text:
        return ""
    text = text.lower().strip()
    for article in ("a ", "an ", "the ", "some "):
        if text.startswith(article):
            text = text[len(article):]
    return text.strip()


def clean_label(label: str) -> str:
    """Clean a detection label: lowercase + strip articles.

    Args:
        label: Detection label (e.g., from GroundingDINO).

    Returns:
        Cleaned label.

    Note:
        Used to normalize GroundingDINO output labels.
    """
    return normalize(label)


# ── Synonym resolution ───────────────────────────────────────────────────

def candidate_synonyms(name: str) -> set[str]:
    """All known synonyms for a name, plus the name itself.

    Args:
        name: Entity name to look up.

    Returns:
        Set of synonyms, including the name itself.

    Note:
        Checks both the full name and each individual word against
        ENTITY_SYNONYMS. E.g. 'cell phone' → {'phone', 'cell phone',
        'smartphone', 'mobile'}.
    """
    syns: set[str] = {name}
    # Direct lookup
    syns.update(ENTITY_SYNONYMS.get(name, []))
    # Per-word lookup
    for word in name.split():
        syns.update(ENTITY_SYNONYMS.get(word, [word]))
    return syns


# ── Entity matching ──────────────────────────────────────────────────────

def entity_matches(a: str, b: str, threshold: int = 80) -> bool:
    """Fuzzy entity matching via core noun + synonyms + rapidfuzz fallback.

    Args:
        a: First entity name.
        b: Second entity name.
        threshold: Minimum token_sort_ratio for fuzzy match (0–100).

    Returns:
        True if entities match via any of the four levels.

    Four-level cascade (in order):
        1. Exact core noun match (fast path)
        2. Substring containment (handles 'dog' vs 'large dog')
        3. Synonym set intersection (maps 'man' ↔ 'person', etc.)
        4. Fuzzy token_sort_ratio >= threshold (robustness fallback)
    """
    if not a or not b:
        return False

    core_a = core_noun(a)
    core_b = core_noun(b)

    # Level 1: exact core match
    if core_a == core_b:
        return True

    # Level 2: substring containment (handles 'dog' vs 'large dog')
    if core_a in core_b or core_b in core_a:
        return True

    # Level 3: synonym intersection
    syns_a = candidate_synonyms(core_a)
    syns_b = candidate_synonyms(core_b)
    if syns_a & syns_b:
        return True

    # Level 4: fuzzy fallback
    return fuzz.token_sort_ratio(normalize(a), normalize(b)) >= threshold


# ── Edit distance ────────────────────────────────────────────────────────

def levenshtein_distance(s1: str, s2: str) -> int:
    """Character-level edit distance via rapidfuzz (C extension, ~100x faster).

    Args:
        s1: First string.
        s2: Second string.

    Returns:
        Edit distance (number of single-character edits).
    """
    return Levenshtein.distance(s1, s2)


def edit_rate(before: str, after: str) -> float:
    """Normalized Levenshtein distance between two strings.

    Args:
        before: Original string.
        after: Modified string.

    Returns:
        Edit rate as a float in [0, 1]: distance / max(len(before), len(after)).
    """
    max_len = max(len(before), len(after), 1)
    return levenshtein_distance(before, after) / max_len
