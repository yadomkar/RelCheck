"""RelCheckClaimGeom — RelCheck with CLAIM + GEOM layers (SCENE disabled).

Same as ``RelCheckClaimOnly`` but passes actual ``object_answers`` to
``build_kb()`` so the GEOM layer computes pairwise spatial facts from
bounding-box geometry.  SCENE layer remains disabled (``image=None``).

Requirements: 6.5, 6.7
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

from relcheck_v3.claim_generation.config import ClaimGenConfig
from relcheck_v3.claim_generation.pipeline import ClaimGenerationPipeline
from relcheck_v3.correction.config import CorrectionConfig
from relcheck_v3.correction.corrector import HallucinationCorrector
from relcheck_v3.kb.builder import build_kb
from relcheck_v3.mllm.cache import InferenceCache

logger = logging.getLogger(__name__)


class RelCheckClaimGeom:
    """RelCheck correction using CLAIM + GEOM layers of the Knowledge Base.

    Uses the two-stage GPT-5.4 correction pipeline:
      - Stage 5a: reasoning_effort=high to identify the hallucination
      - Stage 5b: reasoning_effort=none for surgical correction

    Attributes:
        system_id: Identifier for this system variant (``"claim+geom"``).
    """

    system_id: str = "claim+geom"

    def __init__(
        self,
        openai_api_key: str,
        corrector_model: str = "gpt-5.4",
        gdino_config: str = "",
        gdino_checkpoint: str = "",
        cache_dir: str = "cache/systems/claim_geom/",
    ) -> None:
        resolved_key = openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "OpenAI API key is required for RelCheckClaimGeom. "
                "Pass openai_api_key or set the OPENAI_API_KEY env var."
            )

        self._cache = InferenceCache(Path(cache_dir))
        self._corrector = HallucinationCorrector(
            CorrectionConfig(
                openai_api_key=resolved_key,
                thinking_model=corrector_model,
                correction_model=corrector_model,
            )
        )
        self._pipeline = ClaimGenerationPipeline(
            ClaimGenConfig(
                openai_api_key=resolved_key,
                detector_config=gdino_config,
                detector_model_path=gdino_checkpoint,
            )
        )

    def correct(self, image_path: str, mllm_output: str) -> str:
        cache_key = self._make_cache_key(image_path, mllm_output)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        image_id = Path(image_path).stem
        result = self._pipeline.process_single(
            image=image_path, ref_cap=mllm_output, image_id=image_id,
        )
        if not result.success:
            logger.warning("Claim gen failed for %s — passthrough", image_path)
            return mllm_output

        kb = build_kb(
            vkb=result.visual_knowledge_base,
            object_answers=result.object_answers,
            image=None,
        )

        corrected, _ = self._corrector.run(mllm_output, kb.format())
        self._cache.put(cache_key, corrected, model_id="gpt-5.4", image_id=image_id)
        return corrected

    def _make_cache_key(self, image_path: str, mllm_output: str) -> str:
        output_hash = hashlib.sha256(mllm_output.encode("utf-8")).hexdigest()
        raw = f"{self.system_id}|{image_path}|{output_hash}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
