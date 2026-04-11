"""Correction stage configuration."""

from pydantic import BaseModel


class CorrectionConfig(BaseModel):
    """Configuration for the hallucination correction stage.

    Attributes:
        openai_api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
        thinking_model: Model ID for the correction call.
            GPT-5.4 with reasoning_effort=high for deep analysis.
        reasoning_effort: Reasoning effort level for the correction call.
            One of: none, low, medium, high, xhigh.
        max_edit_chars: Maximum character-level edit distance allowed.
            Corrections exceeding this are rejected (passthrough).
        min_edit_chars: Minimum character-level edit distance required.
            Corrections smaller than this are rejected (passthrough).
    """

    openai_api_key: str = ""
    thinking_model: str = "gpt-5.4"
    reasoning_effort: str = "high"
    max_edit_chars: int = 50
    min_edit_chars: int = 5
