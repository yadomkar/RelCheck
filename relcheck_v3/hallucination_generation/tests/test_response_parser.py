"""Tests for ResponseParser."""

import json

import pytest

from relcheck_v3.hallucination_generation.models import AnnotatedRecord, HallucinationType, ParsedRecord
from relcheck_v3.hallucination_generation.response_parser import ResponseParser


@pytest.fixture
def parser() -> ResponseParser:
    return ResponseParser()


@pytest.fixture
def sample_record() -> AnnotatedRecord:
    return AnnotatedRecord(
        image_id="12345",
        caption="A dog sitting on a bench",
        image_path="/tmp/img.jpg",
        hallucination_type=HallucinationType.OBJECT_EXISTENCE,
        index=0,
    )


# --- JSON parsing tests ---


class TestJsonParsing:
    def test_parse_json_standard_keys(self, parser: ResponseParser, sample_record: AnnotatedRecord):
        raw = json.dumps({
            "image_id": "12345",
            "gt_cap": "A dog sitting on a bench",
            "ref_cap": "A cat sitting on a bench",
            "type": "Object Existence",
            "reason": "Replaced dog with cat",
        })
        result = parser.parse(raw, sample_record)
        assert result.parse_success is True
        assert result.image_id == "12345"
        assert result.gt_cap == "A dog sitting on a bench"
        assert result.ref_cap == "A cat sitting on a bench"
        assert result.hallucination_type == "Object Existence"
        assert result.reason == "Replaced dog with cat"
        assert result.raw_text is None

    def test_parse_json_paper_keys(self, parser: ResponseParser, sample_record: AnnotatedRecord):
        raw = json.dumps({
            "image id": "12345",
            "GT-Cap": "A dog sitting on a bench",
            "Ref-Cap": "A cat sitting on a bench",
            "Type": "Object Existence",
            "Reason": "Replaced dog with cat",
        })
        result = parser.parse(raw, sample_record)
        assert result.parse_success is True
        assert result.ref_cap == "A cat sitting on a bench"

    def test_parse_json_embedded_in_text(self, parser: ResponseParser, sample_record: AnnotatedRecord):
        raw = 'Here is the result: {"image_id": "12345", "gt_cap": "A dog", "ref_cap": "A cat", "type": "Attribute", "reason": "Changed animal"}'
        result = parser.parse(raw, sample_record)
        assert result.parse_success is True
        assert result.ref_cap == "A cat"

    def test_parse_json_missing_field_falls_to_regex(self, parser: ResponseParser, sample_record: AnnotatedRecord):
        """JSON missing a field should fall through to regex, and if regex also fails, return failure."""
        raw = json.dumps({
            "image_id": "12345",
            "gt_cap": "A dog",
            # missing ref_cap, type, reason
        })
        result = parser.parse(raw, sample_record)
        assert result.parse_success is False


# --- Key-value regex parsing tests ---


class TestRegexParsing:
    def test_parse_kv_paper_format(self, parser: ResponseParser, sample_record: AnnotatedRecord):
        raw = (
            "'image id': 12345, "
            "'GT-Cap': A dog sitting on a bench, "
            "'Ref-Cap': A cat sitting on a bench, "
            "'Type': Object Existence, "
            "'Reason': Replaced dog with cat"
        )
        result = parser.parse(raw, sample_record)
        assert result.parse_success is True
        assert result.image_id == "12345"
        assert result.gt_cap == "A dog sitting on a bench"
        assert result.ref_cap == "A cat sitting on a bench"
        assert result.hallucination_type == "Object Existence"
        assert result.reason == "Replaced dog with cat"

    def test_parse_kv_quoted_values(self, parser: ResponseParser, sample_record: AnnotatedRecord):
        raw = (
            "'image id': '12345', "
            "'GT-Cap': 'A dog sitting on a bench', "
            "'Ref-Cap': 'A cat sitting on a bench', "
            "'Type': 'Object Existence', "
            "'Reason': 'Replaced dog with cat'"
        )
        result = parser.parse(raw, sample_record)
        assert result.parse_success is True
        assert result.image_id == "12345"
        assert result.ref_cap == "A cat sitting on a bench"

    def test_parse_kv_multiline(self, parser: ResponseParser, sample_record: AnnotatedRecord):
        raw = (
            "'image id': 12345\n"
            "'GT-Cap': A dog sitting on a bench\n"
            "'Ref-Cap': A cat sitting on a bench\n"
            "'Type': Object Existence\n"
            "'Reason': Replaced dog with cat"
        )
        result = parser.parse(raw, sample_record)
        assert result.parse_success is True
        assert result.ref_cap == "A cat sitting on a bench"


# --- Failure cases ---


class TestParseFailure:
    def test_empty_text(self, parser: ResponseParser, sample_record: AnnotatedRecord):
        result = parser.parse("", sample_record)
        assert result.parse_success is False
        assert result.raw_text == ""
        # Fallback values from original_record
        assert result.image_id == "12345"
        assert result.gt_cap == "A dog sitting on a bench"
        assert result.hallucination_type == "Object Existence"
        assert result.ref_cap == ""
        assert result.reason == ""

    def test_garbage_text(self, parser: ResponseParser, sample_record: AnnotatedRecord):
        result = parser.parse("This is just random text with no structure", sample_record)
        assert result.parse_success is False
        assert result.raw_text == "This is just random text with no structure"

    def test_partial_fields(self, parser: ResponseParser, sample_record: AnnotatedRecord):
        raw = "'image id': 12345, 'GT-Cap': A dog"
        result = parser.parse(raw, sample_record)
        assert result.parse_success is False


# --- Serialize tests ---


class TestSerialize:
    def test_serialize_format(self, parser: ResponseParser):
        record = ParsedRecord(
            image_id="12345",
            gt_cap="A dog sitting on a bench",
            ref_cap="A cat sitting on a bench",
            hallucination_type="Object Existence",
            reason="Replaced dog with cat",
            parse_success=True,
        )
        text = parser.serialize(record)
        assert "'image id': 12345" in text
        assert "'GT-Cap': A dog sitting on a bench" in text
        assert "'Ref-Cap': A cat sitting on a bench" in text
        assert "'Type': Object Existence" in text
        assert "'Reason': Replaced dog with cat" in text

    def test_round_trip(self, parser: ResponseParser, sample_record: AnnotatedRecord):
        """Serialize then re-parse should produce equivalent fields."""
        original = ParsedRecord(
            image_id="99999",
            gt_cap="Two birds on a wire",
            ref_cap="Three birds on a wire",
            hallucination_type="Count",
            reason="Changed count from two to three",
            parse_success=True,
        )
        serialized = parser.serialize(original)
        reparsed = parser.parse(serialized, sample_record)
        assert reparsed.parse_success is True
        assert reparsed.image_id == original.image_id
        assert reparsed.gt_cap == original.gt_cap
        assert reparsed.ref_cap == original.ref_cap
        assert reparsed.hallucination_type == original.hallucination_type
        assert reparsed.reason == original.reason
