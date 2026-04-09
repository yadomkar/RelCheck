"""Property-based and unit tests for KeyConceptExtractor (Stage 1).

Validates: Requirements 1.2, 1.3, 1.4
"""

from unittest.mock import MagicMock

from hypothesis import given, settings
from hypothesis import strategies as st

from relcheck_v3.claim_generation.concept_extractor import KeyConceptExtractor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_client(response: str) -> MagicMock:
    """Create a mock OpenAIClient whose chat() returns *response*."""
    client = MagicMock()
    client.chat.return_value = response
    return client


# Hypothesis strategy: non-empty token (no periods, no leading/trailing spaces)
_token = st.text(
    min_size=1,
    max_size=30,
    alphabet=st.characters(categories=("L", "N"), whitelist_characters=" "),
).map(str.strip).filter(lambda t: len(t) > 0 and "." not in t and t != "None")


# ---------------------------------------------------------------------------
# Property 1: Concept parsing from period-separated string
# ---------------------------------------------------------------------------
# Feature: woodpecker-correction, Property 1: Concept parsing from period-separated string
# Validates: Requirements 1.2


@given(tokens=st.lists(_token, min_size=1, max_size=10))
@settings(max_examples=100)
def test_concept_parsing_from_period_separated_string(tokens: list[str]) -> None:
    """Splitting a period-separated string on periods and stripping whitespace
    produces a list whose length equals the number of segments, and each
    element equals the corresponding segment stripped."""
    # Build a period-separated response with optional surrounding whitespace
    response = ". ".join(tokens)

    client = _make_mock_client(response)
    extractor = KeyConceptExtractor(client=client)
    concepts = extractor.extract("any caption")

    # Deduplicate tokens preserving order (extractor deduplicates)
    seen: set[str] = set()
    expected: list[str] = []
    for t in tokens:
        stripped = t.strip()
        if stripped and stripped not in seen:
            seen.add(stripped)
            expected.append(stripped)

    assert concepts == expected


# ---------------------------------------------------------------------------
# Property 2: No duplicate concepts in extraction output
# ---------------------------------------------------------------------------
# Feature: woodpecker-correction, Property 2: No duplicate concepts in extraction output
# Validates: Requirements 1.3


@given(tokens=st.lists(_token, min_size=1, max_size=10))
@settings(max_examples=100)
def test_no_duplicate_concepts_in_extraction_output(tokens: list[str]) -> None:
    """For any list returned by extract(), len(concepts) == len(set(concepts))."""
    # Intentionally allow duplicate tokens in the generated list
    response = ". ".join(tokens)

    client = _make_mock_client(response)
    extractor = KeyConceptExtractor(client=client)
    concepts = extractor.extract("any caption")

    assert len(concepts) == len(set(concepts))


# ---------------------------------------------------------------------------
# Unit test: "None" response returns empty list
# ---------------------------------------------------------------------------
# Validates: Requirements 1.4


def test_none_response_returns_empty_list() -> None:
    """When the GPT response is 'None', extract() returns an empty list."""
    client = _make_mock_client("None")
    extractor = KeyConceptExtractor(client=client)
    result = extractor.extract("A photo of a sunset over the ocean.")
    assert result == []
