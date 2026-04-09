"""Stage 1: Key concept extraction from captions."""

import logging

from relcheck_v3.claim_generation.openai_client import OpenAIClient
from relcheck_v3.claim_generation.prompts import (
    STAGE1_EXAMPLES,
    STAGE1_SYSTEM_MESSAGE,
    STAGE1_USER_TEMPLATE,
)

logger = logging.getLogger(__name__)


class KeyConceptExtractor:
    """Extracts concrete objects from a caption using GPT-5.4-mini.

    Uses the exact Woodpecker paper prompt (Table 4, Appendix A.1) to
    identify key concepts in singular form with no duplicates.
    """

    def __init__(self, client: OpenAIClient) -> None:
        """Store the OpenAI client instance."""
        self._client = client

    def extract(self, caption: str) -> list[str]:
        """Extract key concepts from a caption using GPT-5.4-mini.

        Sends the caption to GPT using the Stage 1 prompt template,
        parses the period-separated response, and returns deduplicated
        concepts in singular form.

        Args:
            caption: The caption string to extract concepts from.

        Returns:
            List of unique concept strings in singular form.
            Empty list if the model responds with "None".
        """
        user_message = STAGE1_USER_TEMPLATE.format(
            examples=STAGE1_EXAMPLES,
            sentence=caption,
        )

        response = self._client.chat(STAGE1_SYSTEM_MESSAGE, user_message)
        response = response.strip()

        if response == "None":
            return []

        # Split on periods, strip whitespace, filter empty strings
        raw_concepts = [c.strip() for c in response.split(".") if c.strip()]

        # Deduplicate while preserving order
        seen: set[str] = set()
        concepts: list[str] = []
        for concept in raw_concepts:
            if concept not in seen:
                seen.add(concept)
                concepts.append(concept)

        return concepts
