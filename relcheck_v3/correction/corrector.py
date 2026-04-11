"""Hallucination corrector — single-stage GPT-5.4 correction.

One GPT-5.4 chat completion call with reasoning_effort=high cross-references
the caption against the 3-layer KB and produces both a corrected caption and
structured edit metadata in a single pass.

Returns a CorrectionResult with diagnostic fields used downstream for
per-KB-layer ablation analysis.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Literal

import openai
from Levenshtein import distance as lev_distance
from pydantic import BaseModel, Field, ValidationError

from relcheck_v3.correction.config import CorrectionConfig
from relcheck_v3.correction.prompts import (
    RELCHECK_SYSTEM_MESSAGE,
    RELCHECK_USER_TEMPLATE,
)

logger = logging.getLogger(__name__)

_RETRYABLE = (openai.APIError, openai.RateLimitError, openai.APITimeoutError)
_MAX_RETRIES = 3
_BASE_DELAY = 1


# -----------------------------------------------------------------------------
# Result types
# -----------------------------------------------------------------------------

ContradictedBy = Literal[
    "CLAIM-Count", "CLAIM-Specific", "CLAIM-Overall", "SCENE", "GEOM"
]
Confidence = Literal["high", "medium"]
PassthroughReason = Literal[
    "no_edits_needed", "edit_gate", "json_parse_error", "api_error"
]


class Edit(BaseModel):
    """A single span-level correction applied to the caption."""

    original_span: str
    replacement: str
    contradicted_by: ContradictedBy
    evidence: str
    confidence: Confidence


class CorrectionResult(BaseModel):
    """Output of a single corrector run.

    Attributes:
        corrected_caption: The (possibly corrected) caption to use downstream.
        edits: Edits the model claims to have applied. Empty list means the
            model judged the caption fully consistent with the VKB. May be
            non-empty even when was_corrected=False if the edit gate rejected
            an attempted correction (the metadata is preserved for analysis).
        was_corrected: True iff corrected_caption differs from the original.
        passthrough_reason: If non-None, explains why the corrector returned
            the original caption unchanged.
    """

    corrected_caption: str
    edits: list[Edit] = Field(default_factory=list)
    was_corrected: bool
    passthrough_reason: PassthroughReason | None = None


# -----------------------------------------------------------------------------
# Internal model-output schema (what GPT-5.4 returns inside the JSON object)
# -----------------------------------------------------------------------------

class _ModelOutput(BaseModel):
    corrected_caption: str
    edits: list[Edit] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Corrector
# -----------------------------------------------------------------------------

class HallucinationCorrector:
    """Single-stage GPT-5.4 hallucination correction with structured output."""

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

    def run(self, caption: str, kb_text: str) -> CorrectionResult:
        """Cross-reference caption against KB and return a structured result.

        Args:
            caption: The MLLM-generated caption.
            kb_text: Formatted VKB text from ``KnowledgeBase.format()``.

        Returns:
            A ``CorrectionResult`` with the corrected caption, the list of
            edits the model applied, and a passthrough_reason if no
            correction was applied.
        """
        t0 = time.monotonic()
        user_msg = RELCHECK_USER_TEMPLATE.format(caption=caption, vkb=kb_text)

        try:
            raw = self._call(
                model=self._config.thinking_model,
                system=RELCHECK_SYSTEM_MESSAGE,
                user=user_msg,
                reasoning_effort=self._config.reasoning_effort,
            )
        except Exception as exc:
            logger.error("Correction API call failed after retries: %s", exc)
            return CorrectionResult(
                corrected_caption=caption,
                edits=[],
                was_corrected=False,
                passthrough_reason="api_error",
            )

        # Parse JSON output and validate against schema.
        try:
            payload = json.loads(raw)
            parsed = _ModelOutput.model_validate(payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.warning(
                "Correction output was not valid JSON or schema. "
                "Passing through original caption. Error: %s. Raw: %r",
                exc,
                raw[:200],
            )
            return CorrectionResult(
                corrected_caption=caption,
                edits=[],
                was_corrected=False,
                passthrough_reason="json_parse_error",
            )

        corrected = parsed.corrected_caption.strip()
        edits = parsed.edits

        # Model claimed no edits, or returned the same caption — passthrough.
        if not edits or corrected == caption:
            elapsed = time.monotonic() - t0
            logger.info(
                "Correction done in %.2fs — no edits needed", elapsed
            )
            return CorrectionResult(
                corrected_caption=caption,
                edits=[],
                was_corrected=False,
                passthrough_reason="no_edits_needed",
            )

        # Edit-distance gate (final guardrail in case the model ignored
        # the constraint stated in the prompt).
        if not self._passes_edit_gate(caption, corrected):
            dist = lev_distance(caption, corrected)
            logger.info(
                "Correction rejected by edit-distance gate "
                "(distance=%d, allowed=[%d,%d]) — passthrough",
                dist,
                self._config.min_edit_chars,
                self._config.max_edit_chars,
            )
            return CorrectionResult(
                corrected_caption=caption,
                edits=edits,  # preserve metadata for analysis
                was_corrected=False,
                passthrough_reason="edit_gate",
            )

        elapsed = time.monotonic() - t0
        logger.info(
            "Correction done in %.2fs — applied %d edit(s)",
            elapsed,
            len(edits),
        )
        return CorrectionResult(
            corrected_caption=corrected,
            edits=edits,
            was_corrected=True,
            passthrough_reason=None,
        )

    # -------------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------------

    def _call(
        self,
        model: str,
        system: str,
        user: str,
        reasoning_effort: str,
    ) -> str:
        """Make a single chat completion call with retries.

        Always passes reasoning_effort and verbosity explicitly. GPT-5.4
        defaults reasoning_effort to "none" — silently getting no reasoning
        is a foot-gun we don't want.
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = self._client.chat.completions.create(  # type: ignore[call-overload]
                    model=model,
                    messages=messages,
                    reasoning_effort=reasoning_effort,
                    verbosity="low",
                    response_format={"type": "json_object"},
                )
                return response.choices[0].message.content or ""
            except _RETRYABLE as exc:
                last_error = exc
                if attempt < _MAX_RETRIES:
                    delay = 2**attempt * _BASE_DELAY
                    logger.warning(
                        "OpenAI API error (attempt %d/%d): %s. "
                        "Retrying in %.1fs...",
                        attempt + 1,
                        _MAX_RETRIES + 1,
                        exc,
                        delay,
                    )
                    time.sleep(delay)

        assert last_error is not None
        raise last_error

    def _passes_edit_gate(self, original: str, corrected: str) -> bool:
        dist = lev_distance(original, corrected)
        min_ok = dist >= self._config.min_edit_chars
        max_ok = dist <= self._config.max_edit_chars
        return min_ok and max_ok
