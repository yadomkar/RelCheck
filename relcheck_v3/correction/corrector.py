"""Hallucination corrector — two-stage GPT-5.4 correction pipeline.

Stage 5a (Thinking): GPT-5.4 with reasoning_effort=high cross-references
    the caption against the 3-layer KB to identify the specific hallucination.
Stage 5b (Correction): GPT-5.4 with reasoning_effort=none applies
    Woodpecker-style minimal surgical edits guided by KB evidence.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass

import openai

from relcheck_v3.correction.config import CorrectionConfig
from relcheck_v3.correction.prompts import (
    CORRECTION_SYSTEM_MESSAGE,
    CORRECTION_USER_TEMPLATE,
    THINKING_SYSTEM_MESSAGE,
    THINKING_USER_TEMPLATE,
)

logger = logging.getLogger(__name__)

# Transient errors that should trigger retry
_RETRYABLE = (openai.APIError, openai.RateLimitError, openai.APITimeoutError)
_MAX_RETRIES = 3
_BASE_DELAY = 1  # seconds


@dataclass
class HallucinationResult:
    """Output of Stage 5a — hallucination identification.

    Attributes:
        hallucinated_span: Exact substring from the caption that is wrong.
        reason: What the VKB says instead (cites specific KB layer).
        correction_hint: What the span should be replaced with.
        confidence: "high", "medium", or "low".
        raw_json: The full parsed JSON dict from the model response.
    """

    hallucinated_span: str
    reason: str
    correction_hint: str
    confidence: str
    raw_json: dict


class HallucinationCorrector:
    """Two-stage GPT-5.4 hallucination correction.

    Stage 5a uses GPT-5.4 with reasoning_effort=high to *think* about
    which claim in the caption contradicts the Visual Knowledge Base.

    Stage 5b uses GPT-5.4 with reasoning_effort=none to apply minimal,
    surgical edits following the Woodpecker correction prompt (Table 6).
    """

    def __init__(self, config: CorrectionConfig | None = None) -> None:
        """Initialize the corrector from configuration.

        Args:
            config: Optional CorrectionConfig.  Falls back to defaults
                (reads OPENAI_API_KEY from environment).
        """
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

    # ------------------------------------------------------------------
    # Stage 5a — Hallucination Identification (Thinking)
    # ------------------------------------------------------------------

    def identify_hallucination(
        self, caption: str, kb_text: str
    ) -> HallucinationResult | None:
        """Identify the single hallucinated claim in a caption.

        Calls GPT-5.4 with ``reasoning_effort=high`` so the model reasons
        deeply about the caption-vs-KB mismatch before emitting its answer.

        Args:
            caption: The MLLM-generated caption to analyse.
            kb_text: Formatted 3-layer VKB (from ``KnowledgeBase.format()``).

        Returns:
            A ``HallucinationResult`` if identification succeeds, or
            ``None`` when the model response cannot be parsed.
        """
        user_msg = THINKING_USER_TEMPLATE.format(caption=caption, vkb=kb_text)

        raw = self._call(
            model=self._config.thinking_model,
            system=THINKING_SYSTEM_MESSAGE,
            user=user_msg,
            reasoning_effort=self._config.reasoning_effort,
        )

        # Parse the JSON response — tolerate markdown fences
        parsed = self._extract_json(raw)
        if parsed is None:
            logger.warning(
                "Stage 5a: could not parse JSON from model response:\n%s",
                raw[:500],
            )
            return None

        try:
            return HallucinationResult(
                hallucinated_span=parsed["hallucinated_span"],
                reason=parsed["reason"],
                correction_hint=parsed["correction_hint"],
                confidence=parsed.get("confidence", "medium"),
                raw_json=parsed,
            )
        except KeyError as exc:
            logger.warning("Stage 5a: missing key %s in response: %s", exc, parsed)
            return None

    # ------------------------------------------------------------------
    # Stage 5b — Surgical Caption Correction
    # ------------------------------------------------------------------

    def correct_caption(
        self,
        caption: str,
        kb_text: str,
        hallucination: HallucinationResult,
    ) -> str:
        """Apply minimal edits to correct the hallucinated span.

        Uses the verbatim Woodpecker correction prompt (Table 6, Appendix
        A.3) extended with the hallucination identification output from
        Stage 5a.  Runs GPT-5.4 with ``reasoning_effort=none`` (no chain-
        of-thought overhead) for fast, deterministic correction.

        Args:
            caption: The original (hallucinated) caption.
            kb_text: Formatted 3-layer VKB text.
            hallucination: The Stage 5a identification result.

        Returns:
            The corrected caption string.  On failure, returns the
            original caption unchanged (passthrough).
        """
        # Format hallucination info for the prompt
        hallucination_info = (
            f'Hallucinated span: "{hallucination.hallucinated_span}"\n'
            f"Reason: {hallucination.reason}\n"
            f"Suggested correction: {hallucination.correction_hint}"
        )

        user_msg = CORRECTION_USER_TEMPLATE.format(
            kb_info=kb_text,
            passage=caption,
            hallucination_info=hallucination_info,
        )

        corrected = self._call(
            model=self._config.correction_model,
            system=CORRECTION_SYSTEM_MESSAGE,
            user=user_msg,
            reasoning_effort="none",  # fast, no CoT needed
        )

        corrected = corrected.strip()

        # Validate edit-distance guard
        if not self._passes_edit_gate(caption, corrected):
            logger.info(
                "Stage 5b: correction rejected by edit-distance gate "
                "(returning original caption)"
            )
            return caption

        return corrected

    # ------------------------------------------------------------------
    # Combined convenience method
    # ------------------------------------------------------------------

    def run(self, caption: str, kb_text: str) -> tuple[str, HallucinationResult | None]:
        """Run full Stage 5: identify hallucination then correct.

        Args:
            caption: The MLLM-generated caption.
            kb_text: Formatted VKB text from ``KnowledgeBase.format()``.

        Returns:
            A tuple of (corrected_caption, hallucination_result).
            If identification fails or the correction is rejected,
            returns (original_caption, None).
        """
        t0 = time.monotonic()

        hallucination = self.identify_hallucination(caption, kb_text)
        if hallucination is None:
            logger.info("Stage 5: no hallucination identified; passthrough")
            return caption, None

        logger.info(
            "Stage 5a done in %.2fs — hallucinated: %r (confidence=%s)",
            time.monotonic() - t0,
            hallucination.hallucinated_span,
            hallucination.confidence,
        )

        corrected = self.correct_caption(caption, kb_text, hallucination)

        elapsed = time.monotonic() - t0
        changed = corrected != caption
        logger.info(
            "Stage 5 done in %.2fs — %s",
            elapsed,
            "corrected" if changed else "passthrough (no change)",
        )

        return corrected, hallucination if changed else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call(
        self,
        model: str,
        system: str,
        user: str,
        reasoning_effort: str = "none",
    ) -> str:
        """Call OpenAI chat completions with retry and reasoning_effort.

        GPT-5.4's reasoning_effort parameter controls how much internal
        chain-of-thought the model uses before answering:
            - "high"/"xhigh" — deep reasoning (Stage 5a thinking)
            - "none"/"low"   — fast, direct answers (Stage 5b correction)

        Args:
            model: Model ID (e.g. "gpt-5.4").
            system: System message content.
            user: User message content.
            reasoning_effort: One of none/low/medium/high/xhigh.

        Returns:
            The model's response text.

        Raises:
            The last OpenAI error after exhausting retries.
        """
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
        """Check whether the correction falls within the allowed edit window.

        Rejects corrections that are either too large (wholesale rewrite)
        or too small (cosmetic whitespace changes).  Uses character-level
        Levenshtein distance.

        Returns:
            True if the edit distance is in [min_edit_chars, max_edit_chars].
        """
        dist = _levenshtein(original, corrected)
        min_ok = dist >= self._config.min_edit_chars
        max_ok = dist <= self._config.max_edit_chars
        if not (min_ok and max_ok):
            logger.debug(
                "Edit gate: distance=%d (min=%d, max=%d) → %s",
                dist,
                self._config.min_edit_chars,
                self._config.max_edit_chars,
                "PASS" if (min_ok and max_ok) else "REJECT",
            )
        return min_ok and max_ok

    @staticmethod
    def _extract_json(text: str) -> dict | None:
        """Extract the first JSON object from model output.

        Handles common wrappers: markdown fences (```json ... ```),
        leading/trailing whitespace, and bare JSON.
        """
        text = text.strip()

        # Strip markdown code fences
        if text.startswith("```"):
            # Remove opening fence (possibly ```json)
            first_nl = text.index("\n") if "\n" in text else 3
            text = text[first_nl + 1 :]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        # Find first { and last }
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None

        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None


def _levenshtein(s: str, t: str) -> int:
    """Compute character-level Levenshtein distance between two strings.

    Uses the standard O(min(m,n)) space dynamic programming approach.
    """
    if len(s) < len(t):
        return _levenshtein(t, s)

    if len(t) == 0:
        return len(s)

    prev_row = list(range(len(t) + 1))
    for i, sc in enumerate(s):
        curr_row = [i + 1]
        for j, tc in enumerate(t):
            cost = 0 if sc == tc else 1
            curr_row.append(
                min(
                    curr_row[j] + 1,       # insertion
                    prev_row[j + 1] + 1,   # deletion
                    prev_row[j] + cost,     # substitution
                )
            )
        prev_row = curr_row

    return prev_row[-1]
