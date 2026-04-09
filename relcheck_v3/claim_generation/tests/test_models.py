"""Property-based tests for claim generation data models.

Validates: Requirements 11.4, 13.4
"""

import math

from hypothesis import given, settings
from hypothesis import strategies as st

from relcheck_v3.claim_generation.models import (
    AttributeQA,
    AttributeQuestion,
    CountClaim,
    ObjectAnswer,
    OverallClaim,
    SampleResult,
    SpecificClaim,
    StageTimings,
    VisualKnowledgeBase,
)

# ---------------------------------------------------------------------------
# Hypothesis strategies for nested Pydantic models
# ---------------------------------------------------------------------------

# Reusable building blocks
_text = st.text(min_size=1, max_size=50, alphabet=st.characters(categories=("L", "N", "P", "Z")))
_short_text = st.text(min_size=1, max_size=20, alphabet=st.characters(categories=("L", "N")))
_bbox_coord = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
_bbox = st.lists(_bbox_coord, min_size=4, max_size=4)
_non_neg_float = st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False)

st_attribute_question = st.builds(
    AttributeQuestion,
    question=_text,
    entities=st.lists(_short_text, min_size=1, max_size=5),
)

st_object_answer = st.builds(
    ObjectAnswer,
    object_name=_short_text,
    count=st.integers(min_value=0, max_value=20),
    bboxes=st.lists(_bbox, min_size=0, max_size=10),
)

st_attribute_qa = st.builds(
    AttributeQA,
    question=_text,
    entities=st.lists(_short_text, min_size=1, max_size=5),
    answer=_text,
)

st_count_claim = st.builds(
    CountClaim,
    object_name=_short_text,
    count=st.integers(min_value=0, max_value=20),
    claim_text=_text,
    bboxes=st.lists(_bbox, min_size=0, max_size=10),
)

st_specific_claim = st.builds(
    SpecificClaim,
    object_name=_short_text,
    instance_index=st.one_of(st.none(), st.integers(min_value=0, max_value=20)),
    bbox=st.one_of(st.none(), _bbox),
    claim_text=_text,
)

st_overall_claim = st.builds(
    OverallClaim,
    claim_text=_text,
)

st_vkb = st.builds(
    VisualKnowledgeBase,
    count_claims=st.lists(st_count_claim, min_size=0, max_size=5),
    specific_claims=st.lists(st_specific_claim, min_size=0, max_size=5),
    overall_claims=st.lists(st_overall_claim, min_size=0, max_size=5),
)

@st.composite
def _draw_stage_timings(draw: st.DrawFn) -> StageTimings:
    """Generate StageTimings where total_seconds >= sum of individual stages."""
    s1 = draw(_non_neg_float)
    s2 = draw(_non_neg_float)
    s3 = draw(_non_neg_float)
    s4 = draw(_non_neg_float)
    stage_sum = s1 + s2 + s3 + s4
    # total includes overhead, so add a non-negative extra amount
    overhead = draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False))
    return StageTimings(
        stage1_seconds=s1,
        stage2_seconds=s2,
        stage3_seconds=s3,
        stage4_seconds=s4,
        total_seconds=stage_sum + overhead,
    )


st_stage_timings = _draw_stage_timings()

st_sample_result = st.builds(
    SampleResult,
    image_id=_short_text,
    ref_cap=_text,
    key_concepts=st.lists(_short_text, min_size=0, max_size=5),
    object_questions=st.lists(_text, min_size=0, max_size=5),
    attribute_questions=st.lists(st_attribute_question, min_size=0, max_size=3),
    object_answers=st.dictionaries(
        keys=_short_text,
        values=st_object_answer,
        min_size=0,
        max_size=3,
    ),
    attribute_answers=st.lists(st_attribute_qa, min_size=0, max_size=3),
    visual_knowledge_base=st_vkb,
    vkb_text=_text,
    timings=st_stage_timings,
    success=st.booleans(),
    error_message=st.one_of(st.none(), _text),
)


# ---------------------------------------------------------------------------
# Property 20: Checkpoint data serialization round-trip
# ---------------------------------------------------------------------------
# Feature: woodpecker-correction, Property 20: Checkpoint data serialization round-trip
# Validates: Requirements 11.4


@given(result=st_sample_result)
@settings(max_examples=100)
def test_sample_result_serialization_roundtrip(result: SampleResult) -> None:
    """SampleResult serializes to JSON and deserializes back to an equivalent object."""
    json_str = result.model_dump_json()
    restored = SampleResult.model_validate_json(json_str)
    assert restored == result


# ---------------------------------------------------------------------------
# Property 23: Stage timings are non-negative and consistent
# ---------------------------------------------------------------------------
# Feature: woodpecker-correction, Property 23: Stage timings are non-negative and consistent
# Validates: Requirements 13.4


@given(timings=st_stage_timings)
@settings(max_examples=100)
def test_stage_timings_non_negative_and_consistent(timings: StageTimings) -> None:
    """All individual stage timings are >= 0 and total_seconds >= sum of stages."""
    assert timings.stage1_seconds >= 0
    assert timings.stage2_seconds >= 0
    assert timings.stage3_seconds >= 0
    assert timings.stage4_seconds >= 0
    assert timings.total_seconds >= 0

    stage_sum = (
        timings.stage1_seconds
        + timings.stage2_seconds
        + timings.stage3_seconds
        + timings.stage4_seconds
    )
    # total_seconds should be >= sum of individual stages
    # Use a small epsilon for floating-point comparison
    assert timings.total_seconds >= stage_sum or math.isclose(
        timings.total_seconds, stage_sum, rel_tol=1e-9
    )
