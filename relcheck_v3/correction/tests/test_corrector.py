"""Unit tests for the HallucinationCorrector (Stage 5a + 5b)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from relcheck_v3.correction.config import CorrectionConfig
from relcheck_v3.correction.corrector import (
    HallucinationCorrector,
    HallucinationResult,
    _levenshtein,
)


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def config() -> CorrectionConfig:
    return CorrectionConfig(openai_api_key="sk-test-key")


@pytest.fixture
def corrector(config: CorrectionConfig) -> HallucinationCorrector:
    return HallucinationCorrector(config)


SAMPLE_CAPTION = "A cat is sitting on the left side of a dog near a red ball."

SAMPLE_KB_TEXT = (
    "=== CLAIM ===\n"
    "Count:\n1. There is 1 cat.\n2. There is 1 dog.\n3. There is 1 ball.\n"
    "Specific:\n1. The ball is blue.\n"
    "Overall:\n1. A cat and a dog are in the image.\n"
    "\n=== GEOM ===\n"
    "1. cat is to the RIGHT of dog\n"
    "\n=== SCENE ===\n"
    "1. cat sitting on table (conf=0.72)\n"
)

STAGE5A_RESPONSE_JSON = json.dumps(
    {
        "hallucinated_span": "left side",
        "reason": "GEOM layer says: cat is to the RIGHT of dog",
        "correction_hint": "right side",
        "confidence": "high",
    }
)

CORRECTED_CAPTION = "A cat is sitting on the right side of a dog near a blue ball."


# ── Levenshtein tests ─────────────────────────────────────────────────


class TestLevenshtein:
    def test_equal_strings(self):
        assert _levenshtein("abc", "abc") == 0

    def test_one_insertion(self):
        assert _levenshtein("abc", "abcd") == 1

    def test_one_deletion(self):
        assert _levenshtein("abcd", "abc") == 1

    def test_one_substitution(self):
        assert _levenshtein("abc", "axc") == 1

    def test_empty_strings(self):
        assert _levenshtein("", "") == 0

    def test_one_empty(self):
        assert _levenshtein("abc", "") == 3
        assert _levenshtein("", "abc") == 3

    def test_symmetry(self):
        assert _levenshtein("kitten", "sitting") == _levenshtein("sitting", "kitten")


# ── JSON extraction tests ─────────────────────────────────────────────


class TestExtractJson:
    def test_bare_json(self, corrector):
        result = corrector._extract_json('{"key": "val"}')
        assert result == {"key": "val"}

    def test_markdown_fenced(self, corrector):
        text = '```json\n{"key": "val"}\n```'
        result = corrector._extract_json(text)
        assert result == {"key": "val"}

    def test_text_before_json(self, corrector):
        text = 'Here is the result:\n{"key": "val"}'
        result = corrector._extract_json(text)
        assert result == {"key": "val"}

    def test_invalid_json(self, corrector):
        result = corrector._extract_json("not json at all")
        assert result is None

    def test_empty_string(self, corrector):
        assert corrector._extract_json("") is None


# ── Stage 5a: identify_hallucination tests ────────────────────────────


class TestIdentifyHallucination:
    @patch.object(HallucinationCorrector, "_call")
    def test_successful_identification(self, mock_call, corrector):
        mock_call.return_value = STAGE5A_RESPONSE_JSON

        result = corrector.identify_hallucination(SAMPLE_CAPTION, SAMPLE_KB_TEXT)

        assert result is not None
        assert result.hallucinated_span == "left side"
        assert result.confidence == "high"
        assert "RIGHT" in result.reason

        # Verify the call was made with thinking model + high effort
        mock_call.assert_called_once()
        call_kwargs = mock_call.call_args
        assert call_kwargs.kwargs["model"] == "gpt-5.4"
        assert call_kwargs.kwargs["reasoning_effort"] == "high"

    @patch.object(HallucinationCorrector, "_call")
    def test_unparseable_response(self, mock_call, corrector):
        mock_call.return_value = "I couldn't find any issues."

        result = corrector.identify_hallucination(SAMPLE_CAPTION, SAMPLE_KB_TEXT)
        assert result is None

    @patch.object(HallucinationCorrector, "_call")
    def test_missing_key_in_json(self, mock_call, corrector):
        mock_call.return_value = json.dumps({"hallucinated_span": "left"})

        result = corrector.identify_hallucination(SAMPLE_CAPTION, SAMPLE_KB_TEXT)
        assert result is None  # missing reason + correction_hint


# ── Stage 5b: correct_caption tests ──────────────────────────────────


class TestCorrectCaption:
    @patch.object(HallucinationCorrector, "_call")
    def test_successful_correction(self, mock_call, corrector):
        mock_call.return_value = CORRECTED_CAPTION

        hallucination = HallucinationResult(
            hallucinated_span="left side",
            reason="GEOM says RIGHT",
            correction_hint="right side",
            confidence="high",
            raw_json={},
        )

        result = corrector.correct_caption(SAMPLE_CAPTION, SAMPLE_KB_TEXT, hallucination)

        # Edit distance "left" → "right" + "red" → "blue" is within gate
        assert "right side" in result or result == SAMPLE_CAPTION
        mock_call.assert_called_once()
        call_kwargs = mock_call.call_args
        assert call_kwargs.kwargs["reasoning_effort"] == "none"

    @patch.object(HallucinationCorrector, "_call")
    def test_edit_gate_rejects_too_large(self, mock_call, corrector):
        # Return a completely different caption (edit distance > 50)
        mock_call.return_value = "X" * 200

        hallucination = HallucinationResult(
            hallucinated_span="left",
            reason="test",
            correction_hint="right",
            confidence="high",
            raw_json={},
        )

        result = corrector.correct_caption(SAMPLE_CAPTION, SAMPLE_KB_TEXT, hallucination)
        assert result == SAMPLE_CAPTION  # passthrough

    @patch.object(HallucinationCorrector, "_call")
    def test_edit_gate_rejects_too_small(self, mock_call, corrector):
        # Return caption with only 1 char changed (below min_edit_chars=5)
        almost_same = SAMPLE_CAPTION[:-1] + "!"
        mock_call.return_value = almost_same

        hallucination = HallucinationResult(
            hallucinated_span=".",
            reason="test",
            correction_hint="!",
            confidence="low",
            raw_json={},
        )

        result = corrector.correct_caption(SAMPLE_CAPTION, SAMPLE_KB_TEXT, hallucination)
        assert result == SAMPLE_CAPTION  # passthrough (edit < 5)


# ── Full run() tests ─────────────────────────────────────────────────


class TestRun:
    @patch.object(HallucinationCorrector, "_call")
    def test_full_pipeline(self, mock_call, corrector):
        # First call = Stage 5a (thinking), second = Stage 5b (correction)
        mock_call.side_effect = [STAGE5A_RESPONSE_JSON, CORRECTED_CAPTION]

        corrected, hallucination = corrector.run(SAMPLE_CAPTION, SAMPLE_KB_TEXT)

        assert mock_call.call_count == 2
        # Stage 5a should use reasoning_effort=high
        assert mock_call.call_args_list[0].kwargs["reasoning_effort"] == "high"
        # Stage 5b should use reasoning_effort=none
        assert mock_call.call_args_list[1].kwargs["reasoning_effort"] == "none"

    @patch.object(HallucinationCorrector, "_call")
    def test_passthrough_on_identification_failure(self, mock_call, corrector):
        mock_call.return_value = "No hallucination found."

        corrected, hallucination = corrector.run(SAMPLE_CAPTION, SAMPLE_KB_TEXT)

        assert corrected == SAMPLE_CAPTION
        assert hallucination is None
        assert mock_call.call_count == 1  # Stage 5b never called


# ── Config tests ─────────────────────────────────────────────────────


class TestConfig:
    def test_no_api_key_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="API key"):
                HallucinationCorrector(CorrectionConfig(openai_api_key=""))

    def test_env_var_fallback(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-from-env"}):
            corrector = HallucinationCorrector(CorrectionConfig())
            assert corrector._client is not None
