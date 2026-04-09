"""Property-based tests for prompt template placeholder substitution.

Validates: Requirements 10.5
"""

import re

from hypothesis import given, settings
from hypothesis import strategies as st

from relcheck_v3.claim_generation.prompts import (
    OBJECT_QUESTION_TEMPLATE,
    STAGE1_USER_TEMPLATE,
    STAGE2_USER_TEMPLATE,
    STAGE4_QA_TO_CLAIM_TEMPLATE,
)

# Strategy: non-empty strings that avoid braces so they can't be confused with placeholders
_nonempty_text = st.text(
    min_size=1,
    max_size=80,
    alphabet=st.characters(categories=("L", "N", "P", "Z"), exclude_characters="{}"),
)

# Regex matching any remaining {placeholder} token
_PLACEHOLDER_RE = re.compile(r"\{[^}]+\}")


# ---------------------------------------------------------------------------
# Property 17: Prompt template placeholder substitution
# ---------------------------------------------------------------------------
# Feature: woodpecker-correction, Property 17: Prompt template placeholder substitution
# Validates: Requirements 10.5


@given(examples=_nonempty_text, sentence=_nonempty_text)
@settings(max_examples=100)
def test_stage1_user_template_substitution(examples: str, sentence: str) -> None:
    """STAGE1_USER_TEMPLATE: formatted prompt contains each value and no remaining placeholders."""
    result = STAGE1_USER_TEMPLATE.format(examples=examples, sentence=sentence)
    assert examples in result
    assert sentence in result
    assert not _PLACEHOLDER_RE.search(result)


@given(examples=_nonempty_text, sentence=_nonempty_text, entities=_nonempty_text)
@settings(max_examples=100)
def test_stage2_user_template_substitution(
    examples: str, sentence: str, entities: str
) -> None:
    """STAGE2_USER_TEMPLATE: formatted prompt contains each value and no remaining placeholders."""
    result = STAGE2_USER_TEMPLATE.format(
        examples=examples, sentence=sentence, entities=entities
    )
    assert examples in result
    assert sentence in result
    assert entities in result
    assert not _PLACEHOLDER_RE.search(result)


@given(question=_nonempty_text, answer=_nonempty_text)
@settings(max_examples=100)
def test_stage4_qa_to_claim_template_substitution(
    question: str, answer: str
) -> None:
    """STAGE4_QA_TO_CLAIM_TEMPLATE: formatted prompt contains each value and no remaining placeholders."""
    result = STAGE4_QA_TO_CLAIM_TEMPLATE.format(question=question, answer=answer)
    assert question in result
    assert answer in result
    assert not _PLACEHOLDER_RE.search(result)


@given(object_name=_nonempty_text)
@settings(max_examples=100)
def test_object_question_template_substitution(object_name: str) -> None:
    """OBJECT_QUESTION_TEMPLATE: formatted prompt contains the value and no remaining placeholders."""
    result = OBJECT_QUESTION_TEMPLATE.format(object=object_name)
    assert object_name in result
    assert not _PLACEHOLDER_RE.search(result)
