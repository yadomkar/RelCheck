"""LLM judge shim for extracting yes/no answers from corrected descriptions.

Uses a cheap LLM (default ``gpt-5.4-mini``) to derive binary answers for
discriminative benchmark questions.  The same judge instance is shared across
all correction systems to ensure fair comparison.

Results are cached to disk via :class:`~relcheck_v3.mllm.cache.InferenceCache`
keyed by ``hashlib.sha256(description + question)``.

Requirements: 7.1, 7.2, 7.3, 7.4
"""

from __future__ import annotations

import hashlib
import logging

import openai
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from relcheck_v3.mllm.cache import InferenceCache

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = (
    "Given this description of an image: {description}\n"
    "Answer this question with only yes or no: {question}"
)


class AnswerExtractor:
    """Derive yes/no answers from corrected descriptions via an LLM judge.

    Args:
        openai_api_key: API key for the OpenAI SDK.
        model: Model identifier for the judge LLM.
        cache: Optional :class:`InferenceCache` for persisting judge outputs.
    """

    def __init__(
        self,
        openai_api_key: str,
        model: str = "gpt-5.4-mini",
        cache: InferenceCache | None = None,
    ) -> None:
        self._client = openai.OpenAI(api_key=openai_api_key)
        self._model = model
        self._cache = cache

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_yesno(self, description: str, question: str) -> str:
        """Prompt the LLM judge and return ``"yes"`` or ``"no"``.

        The prompt sent to the judge is::

            Given this description of an image: {description}
            Answer this question with only yes or no: {question}

        Results are cached by ``hashlib.sha256(description + question)``.
        Non-yes/no responses are normalised to ``"no"`` with a warning.

        Args:
            description: The (possibly corrected) image description.
            question: The discriminative yes/no question.

        Returns:
            ``"yes"`` or ``"no"``.
        """
        cache_key = self._make_cache_key(description, question)

        # Check cache first.
        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        raw_answer = self._call_judge(description, question)
        normalised = self._normalise(raw_answer)

        # Persist to cache.
        if self._cache is not None:
            self._cache.put(
                cache_key,
                normalised,
                model_id=self._model,
            )

        return normalised

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _make_cache_key(description: str, question: str) -> str:
        """Build a deterministic cache key from description and question."""
        raw = description + question
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @retry(
        retry=retry_if_exception_type(
            (openai.APITimeoutError, openai.APIConnectionError, openai.RateLimitError),
        ),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _call_judge(self, description: str, question: str) -> str:
        """Send the prompt to the LLM judge with retry on transient errors."""
        prompt = _PROMPT_TEMPLATE.format(
            description=description,
            question=question,
        )
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_completion_tokens=5,
        )
        content = response.choices[0].message.content or ""
        return content.strip()

    @staticmethod
    def _normalise(raw: str) -> str:
        """Normalise the judge response to ``"yes"`` or ``"no"``.

        Any response that is not clearly ``"yes"`` is mapped to ``"no"``
        with a warning logged.
        """
        lowered = raw.strip().lower().rstrip(".")
        if lowered == "yes":
            return "yes"
        if lowered == "no":
            return "no"

        logger.warning(
            "LLM judge returned non-yes/no response %r — normalising to 'no'",
            raw,
        )
        return "no"
