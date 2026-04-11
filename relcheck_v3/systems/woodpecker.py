"""WoodpeckerBaseline — faithful reproduction of Yin et al.'s 5-stage pipeline.

Uses GPT-3.5 for Stages 1, 2, 5; GroundingDINO for object detection (Stage 3a);
BLIP-2-FlanT5-XXL for attribute VQA (Stage 3b).  No GEOM layer, no SCENE layer.

The existing ``ClaimGenerationPipeline`` handles Stages 1–4.  This module adds
Stage 5 correction via the OpenAI SDK with ``tenacity`` retry.

Requirements: 6.3, 6.7
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

import openai
from tenacity import retry, stop_after_attempt, wait_exponential

from relcheck_v3.claim_generation.config import ClaimGenConfig
from relcheck_v3.claim_generation.pipeline import ClaimGenerationPipeline
from relcheck_v3.mllm.cache import InferenceCache

logger = logging.getLogger(__name__)

# Woodpecker Stage 5 prompt (paper appendix Table 6)
_STAGE5_PROMPT = (
    "Based on the given claims, please correct the description of the image.\n"
    "Description: {mllm_output}\n"
    "Claims: {vkb_text}\n"
    "Corrected description:"
)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def _call_openai(
    client: openai.OpenAI,
    model: str,
    prompt: str,
) -> str:
    """Call OpenAI chat completion with tenacity retry on transient errors.

    Args:
        client: Configured ``openai.OpenAI`` client instance.
        model: Model identifier (e.g. ``"gpt-3.5-turbo"``).
        prompt: The user prompt to send.

    Returns:
        The assistant's response text.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_completion_tokens=512,
    )
    return response.choices[0].message.content or ""


class WoodpeckerBaseline:
    """Faithful reproduction of the Woodpecker 5-stage correction pipeline.

    Stages 1–4 are delegated to the existing ``ClaimGenerationPipeline``.
    Stage 5 (correction) uses the OpenAI SDK to call the corrector model
    with the Woodpecker paper's Table 6 prompt.

    Attributes:
        system_id: Identifier for this system variant (``"woodpecker"``).
    """

    system_id: str = "woodpecker"

    def __init__(
        self,
        openai_api_key: str,
        corrector_model: str = "gpt-3.5-turbo",
        gdino_config: str = "",
        gdino_checkpoint: str = "",
        cache_dir: str = "cache/systems/woodpecker/",
    ) -> None:
        """Initialise the Woodpecker baseline.

        Args:
            openai_api_key: OpenAI API key for GPT calls.
            corrector_model: Model used for Stage 5 correction.
            gdino_config: Path to GroundingDINO config file.
            gdino_checkpoint: Path to GroundingDINO checkpoint.
            cache_dir: Directory for caching intermediate and final results.

        Raises:
            ValueError: If *openai_api_key* is empty and the
                ``OPENAI_API_KEY`` environment variable is not set.
        """
        resolved_key = openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "OpenAI API key is required for WoodpeckerBaseline. "
                "Pass openai_api_key or set the OPENAI_API_KEY env var."
            )

        self._corrector_model = corrector_model
        self._client = openai.OpenAI(api_key=resolved_key)
        self._cache = InferenceCache(Path(cache_dir))

        # Build claim-generation pipeline config (GPT-3.5 for Stages 1–2)
        self._pipeline = ClaimGenerationPipeline(
            ClaimGenConfig(
                openai_api_key=resolved_key,
                gpt_model_id="gpt-5.4-mini",
                detector_config=gdino_config,
                detector_model_path=gdino_checkpoint,
            )
        )

    # ------------------------------------------------------------------
    # CorrectionSystem interface
    # ------------------------------------------------------------------

    def correct(self, image_path: str, mllm_output: str) -> str:
        """Run Stages 1–5 and return the corrected description.

        Args:
            image_path: Absolute path to the source image file.
            mllm_output: Raw text produced by the MLLM.

        Returns:
            The corrected description string.
        """
        # Check final-result cache first
        cache_key = self._make_cache_key(image_path, mllm_output)
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.debug("Woodpecker cache hit for %s", image_path)
            return cached

        # Stages 1–4: claim generation
        image_id = Path(image_path).stem
        result = self._pipeline.process_single(
            image=image_path,
            ref_cap=mllm_output,
            image_id=image_id,
        )

        if not result.success:
            logger.warning(
                "Woodpecker claim generation failed for %s: %s — returning raw output",
                image_path,
                result.error_message,
            )
            return mllm_output

        # Stage 5: correction via GPT
        vkb_text = result.visual_knowledge_base.format()
        prompt = _STAGE5_PROMPT.format(
            mllm_output=mllm_output,
            vkb_text=vkb_text,
        )

        corrected = _call_openai(self._client, self._corrector_model, prompt)
        logger.info("Woodpecker Stage 5 corrected %s", image_path)

        # Persist to cache
        self._cache.put(
            cache_key,
            corrected,
            model_id=self._corrector_model,
            image_id=image_id,
        )
        return corrected

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _make_cache_key(self, image_path: str, mllm_output: str) -> str:
        """Build a deterministic cache key for the correction result.

        Key: ``sha256(system_id + "|" + image_path + "|" + sha256(mllm_output))[:16]``
        """
        output_hash = hashlib.sha256(mllm_output.encode("utf-8")).hexdigest()
        raw = f"{self.system_id}|{image_path}|{output_hash}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
