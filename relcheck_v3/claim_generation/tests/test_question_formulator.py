"""Unit tests for Stage 2: QuestionFormulator."""

from unittest.mock import MagicMock, patch

import pytest

from relcheck_v3.claim_generation.models import (
    AttributeQuestion,
    FormulatedQuestions,
    ObjectQuestion,
)
from relcheck_v3.claim_generation.question_formulator import QuestionFormulator


@pytest.fixture()
def mock_client() -> MagicMock:
    """Create a mock OpenAIClient."""
    return MagicMock()


class TestObjectQuestionGeneration:
    """Tests for hardcoded object-level question generation."""

    def test_one_concept(self, mock_client: MagicMock) -> None:
        mock_client.chat.return_value = "None"
        qf = QuestionFormulator(mock_client)
        result = qf.formulate("A dog in the park.", ["dog"])

        assert len(result.object_questions) == 1
        assert result.object_questions[0].object_name == "dog"
        assert result.object_questions[0].question == (
            "Is there any dog in the image? How many are there?"
        )

    def test_multiple_concepts(self, mock_client: MagicMock) -> None:
        mock_client.chat.return_value = "None"
        qf = QuestionFormulator(mock_client)
        result = qf.formulate("A dog and a cat on a bed.", ["dog", "cat", "bed"])

        assert len(result.object_questions) == 3
        for oq, name in zip(result.object_questions, ["dog", "cat", "bed"]):
            assert oq.object_name == name
            assert oq.question == f"Is there any {name} in the image? How many are there?"

    def test_empty_concepts(self, mock_client: MagicMock) -> None:
        mock_client.chat.return_value = "None"
        qf = QuestionFormulator(mock_client)
        result = qf.formulate("An empty scene.", [])

        assert result.object_questions == []
        assert result.attribute_questions == []


class TestAttributeQuestionParsing:
    """Tests for attribute-level question parsing from LLM response."""

    def test_single_entity_question(self, mock_client: MagicMock) -> None:
        mock_client.chat.return_value = "What color is the dog?&dog"
        qf = QuestionFormulator(mock_client)
        result = qf.formulate("A brown dog.", ["dog"])

        assert len(result.attribute_questions) == 1
        aq = result.attribute_questions[0]
        assert aq.question == "What color is the dog?"
        assert aq.entities == ["dog"]

    def test_multi_entity_question(self, mock_client: MagicMock) -> None:
        mock_client.chat.return_value = (
            "Is the man standing in the kitchen?&man.kitchen"
        )
        qf = QuestionFormulator(mock_client)
        result = qf.formulate("A man in a kitchen.", ["man", "kitchen"])

        assert len(result.attribute_questions) == 1
        aq = result.attribute_questions[0]
        assert aq.question == "Is the man standing in the kitchen?"
        assert aq.entities == ["man", "kitchen"]

    def test_multiple_lines(self, mock_client: MagicMock) -> None:
        mock_client.chat.return_value = (
            "What color is the cat?&cat\n"
            "What color is the dog?&dog\n"
        )
        qf = QuestionFormulator(mock_client)
        result = qf.formulate("A dog and a cat.", ["dog", "cat"])

        assert len(result.attribute_questions) == 2
        assert result.attribute_questions[0].question == "What color is the cat?"
        assert result.attribute_questions[1].question == "What color is the dog?"

    def test_skips_lines_without_ampersand(self, mock_client: MagicMock) -> None:
        mock_client.chat.return_value = (
            "What color is the dog?&dog\n"
            "This line has no delimiter\n"
            "What is the cat doing?&cat\n"
        )
        qf = QuestionFormulator(mock_client)
        result = qf.formulate("A dog and a cat.", ["dog", "cat"])

        assert len(result.attribute_questions) == 2

    def test_skips_empty_lines(self, mock_client: MagicMock) -> None:
        mock_client.chat.return_value = (
            "What color is the dog?&dog\n"
            "\n"
            "What is the cat doing?&cat\n"
        )
        qf = QuestionFormulator(mock_client)
        result = qf.formulate("A dog and a cat.", ["dog", "cat"])

        assert len(result.attribute_questions) == 2


class TestNoneFallback:
    """Tests for 'None' response handling (Req 2.4)."""

    def test_none_response_returns_only_object_questions(
        self, mock_client: MagicMock
    ) -> None:
        mock_client.chat.return_value = "None"
        qf = QuestionFormulator(mock_client)
        result = qf.formulate("A dog in the park.", ["dog"])

        assert len(result.object_questions) == 1
        assert result.attribute_questions == []


class TestAPIFailureFallback:
    """Tests for graceful fallback on API failure (Req 2.5)."""

    def test_api_exception_returns_only_object_questions(
        self, mock_client: MagicMock
    ) -> None:
        mock_client.chat.side_effect = Exception("API error after retries")
        qf = QuestionFormulator(mock_client)
        result = qf.formulate("A dog in the park.", ["dog"])

        assert len(result.object_questions) == 1
        assert result.object_questions[0].object_name == "dog"
        assert result.attribute_questions == []

    def test_api_failure_with_multiple_concepts(
        self, mock_client: MagicMock
    ) -> None:
        mock_client.chat.side_effect = RuntimeError("Connection failed")
        qf = QuestionFormulator(mock_client)
        result = qf.formulate("A dog and a cat.", ["dog", "cat"])

        assert len(result.object_questions) == 2
        assert result.attribute_questions == []


class TestPromptConstruction:
    """Tests that the correct prompts are sent to the LLM."""

    def test_entities_joined_with_periods(self, mock_client: MagicMock) -> None:
        mock_client.chat.return_value = "None"
        qf = QuestionFormulator(mock_client)
        qf.formulate("A dog and a cat on a bicycle.", ["dog", "cat", "bicycle"])

        # Verify the user message contains entities joined with periods
        call_args = mock_client.chat.call_args
        user_message = call_args[0][1]  # second positional arg
        assert "dog.cat.bicycle" in user_message

    def test_uses_stage2_system_message(self, mock_client: MagicMock) -> None:
        mock_client.chat.return_value = "None"
        qf = QuestionFormulator(mock_client)
        qf.formulate("A dog.", ["dog"])

        call_args = mock_client.chat.call_args
        system_message = call_args[0][0]  # first positional arg
        assert system_message == (
            "You are a language assistant that helps to ask questions "
            "about a sentence."
        )
