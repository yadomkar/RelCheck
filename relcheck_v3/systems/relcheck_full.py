"""RelCheckFull — RelCheck with all three KB layers (CLAIM + GEOM + SCENE).

Passes the PIL image to ``build_kb()`` so RelTR runs for the SCENE layer.
Requires ``reltr_cfg.ENABLE_RELTR = True`` and a valid RelTR checkpoint.

Uses the two-stage GPT-5.4 correction pipeline from ``relcheck_v3.correction``.

Requirements: 6.6, 6.7
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

from PIL import Image

from relcheck_v3.claim_generation.config import ClaimGenConfig
from relcheck_v3.claim_generation.pipeline import ClaimGenerationPipeline
from relcheck_v3.correction.config import CorrectionConfig
from relcheck_v3.correction.corrector import HallucinationCorrector
from relcheck_v3.kb.builder import build_kb
from relcheck_v3.mllm.cache import InferenceCache
from relcheck_v3.reltr import config as reltr_cfg

logger = logging.getLogger(__name__)


class RelCheckFull:
    """RelCheck correction using all three KB layers: CLAIM + GEOM + SCENE.

    Uses the two-stage GPT-5.4 correction pipeline:
      - Stage 5a: reasoning_effort=high to identify the hallucination
      - Stage 5b: reasoning_effort=none for surgical correction

    Attributes:
        system_id: Identifier for this system variant (``"full"``).
        last_kb_text: The KB text from the most recent ``correct()`` call.
            Useful for inspection/debugging. Also cached to disk.
    """

    system_id: str = "full"

    def __init__(
        self,
        openai_api_key: str,
        corrector_model: str = "gpt-5.4",
        gdino_config: str = "",
        gdino_checkpoint: str = "",
        reltr_checkpoint: str = "",
        cache_dir: str = "cache/systems/full/",
    ) -> None:
        resolved_key = openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "OpenAI API key is required for RelCheckFull. "
                "Pass openai_api_key or set the OPENAI_API_KEY env var."
            )

        self._cache = InferenceCache(Path(cache_dir))
        self._kb_cache = InferenceCache(Path(cache_dir) / "kb")
        self._corrector = HallucinationCorrector(
            CorrectionConfig(
                openai_api_key=resolved_key,
                thinking_model=corrector_model,
                correction_model=corrector_model,
            )
        )

        reltr_cfg.ENABLE_RELTR = True
        if reltr_checkpoint:
            reltr_cfg.RELTR_CHECKPOINT_PATH = reltr_checkpoint

        self._pipeline = ClaimGenerationPipeline(
            ClaimGenConfig(
                openai_api_key=resolved_key,
                detector_config=gdino_config,
                detector_model_path=gdino_checkpoint,
            )
        )

        self.last_kb_text: str = ""

    def correct(self, image_path: str, mllm_output: str) -> str:
        cache_key = self._make_cache_key(image_path, mllm_output)
        cached = self._cache.get(cache_key)
        if cached is not None:
            # Also restore KB from cache if available
            kb_cached = self._kb_cache.get(cache_key)
            if kb_cached is not None:
                self.last_kb_text = kb_cached
            return cached

        image_id = Path(image_path).stem
        result = self._pipeline.process_single(
            image=image_path, ref_cap=mllm_output, image_id=image_id,
        )
        if not result.success:
            logger.warning("Claim gen failed for %s — passthrough", image_path)
            return mllm_output

        pil_image = Image.open(image_path).convert("RGB")
        kb = build_kb(
            vkb=result.visual_knowledge_base,
            object_answers=result.object_answers,
            image=pil_image,
        )

        self.last_kb_text = kb.format()

        # Cache the KB text
        self._kb_cache.put(cache_key, self.last_kb_text, model_id="kb", image_id=image_id)

        corrected, _ = self._corrector.run(mllm_output, self.last_kb_text)
        self._cache.put(cache_key, corrected, model_id="gpt-5.4", image_id=image_id)
        return corrected

    def get_cached_kb(self, image_path: str, mllm_output: str) -> str | None:
        """Retrieve the cached KB text for a given image/output pair."""
        cache_key = self._make_cache_key(image_path, mllm_output)
        return self._kb_cache.get(cache_key)

    def _make_cache_key(self, image_path: str, mllm_output: str) -> str:
        output_hash = hashlib.sha256(mllm_output.encode("utf-8")).hexdigest()
        raw = f"{self.system_id}|{image_path}|{output_hash}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
