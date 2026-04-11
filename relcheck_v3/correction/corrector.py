"""Hallucination corrector — single-stage GPT-5.4 correction.

One GPT-5.4 call with reasoning_effort=high cross-references the caption
against the 3-layer KB and produces a corrected caption directly.
"""

from __future__ import annotations

import logging
import os
import time

import openai

from relcheck_v3.correction.config import CorrectionConfig
from relcheck_v3.correction.prompts import (
    RELCHECK_SYSTEM_MESSAGE,
    RELCHECK_USER_TEMPLATE,
)

logger = logging.getLogger(__name__)

_RETRYABLE = (openai.APIError, openai.RateLimitError, openai.APITimeoutError)
_MAX_RETRIES = 3
_BASE_DELAY = 1


class HallucinationCorrector:
    """Single-stage GPT-5.4 hallucination correction.

    One call with reasoning_effort=high to identify and fix
    hallucinations in a single pass.
    """

    def __init__(self, config: CorrectionConfig | None = None) -> None:
        self._config = config or CorrectionConfig()
        resolved_key = self._config.openai_api_key or os.environ.get(
            "OPENAI_API_KEY", ""
        )
        if not resolved_key:
            raise ValueError(
                "OpenAI API key not configured. "
                "Pass openai_api_key in CorrectionConfig or set "
                "OPENAI_API_KEY environment variable."
            )
        self._client = openai.OpenAI(api_key=resolved_key)

    def run(self, caption: str, kb_text: str) -> tuple[str, None]:
        """Cross-reference caption against KB and return corrected version.

        Single GPT-5.4 call with reasoning_effort=high.

        Args:
            caption: The MLLM-generated caption.
            kb_text: Formatted VKB text from ``KnowledgeBase.format()``.

        Returns:
            A tuple of (corrected_caption, None).
            Second element is None for backward compatibility.
        """
        t0 = time.monotonic()

        user_msg = RELCHECK_USER_TEMPLATE.format(caption=caption, vkb=kb_text)

        corrected = self._call(
            model=self._config.thinking_model,
            system=RELCHECK_SYSTEM_MESSAGE,
            user=user_msg,
            reasoning_effort=self._config.reasoning_effort,
        )

        corrected = corrected.strip()

        # Edit-distance guard
        if not self._passes_edit_gate(caption, corrected):
            logger.info("Correction rejected by edit-distance gate — passthrough")
            return caption, None

        elapsed = time.monotonic() - t0
        changed = corrected != caption
        logger.info(
            "Correction done in %.2fs — %s",
            elapsed,
            "corrected" if changed else "passthrough (no change)",
        )

        return corrected, None

    def _call(
        self,
        model: str,
        system: str,
        user: str,
        reasoning_effort: str = "high",
    ) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    reasoning_effort=reasoning_effort,
                )
                return response.choices[0].message.content or ""
            except _RETRYABLE as exc:
                last_error = exc
                if attempt < _MAX_RETRIES:
                    delay = 2**attempt * _BASE_DELAY
                    logger.warning(
                        "OpenAI API error (attempt %d/%d): %s. Retrying in %.1fs...",
                        attempt + 1,
                        _MAX_RETRIES + 1,
                        exc,
                        delay,
                    )
                    time.sleep(delay)

        raise last_error  # type: ignore[misc]

    def _passes_edit_gate(self, original: str, corrected: str) -> bool:
        from Levenshtein import distance
        dist = distance(original, corrected)
        min_ok = dist >= self._config.min_edit_chars
        max_ok = dist <= self._config.max_edit_chars
        return min_ok and max_ok


