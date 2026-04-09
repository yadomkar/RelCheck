"""RelCheckCaptionEditor — top-level Caption_Editor for the correction stage.

Orchestrates the full RelCheck correction pipeline:
    1. Claim generation (Woodpecker Stages 1-4) → Visual Knowledge Base
    2. KB building (CLAIM + GEOM + SCENE)
    3. Hallucination identification (GPT-5.4 thinking)
    4. Surgical correction (GPT-5.4 fast)

Implements the ``Caption_Editor`` protocol so it can be dropped into the
evaluation harness alongside baselines (Passthrough, LLaVA, etc.).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from relcheck_v3.correction.config import CorrectionConfig
from relcheck_v3.correction.corrector import HallucinationCorrector, HallucinationResult

if TYPE_CHECKING:
    from PIL.Image import Image
    from relcheck_v3.claim_generation.config import ClaimGenConfig
    from relcheck_v3.claim_generation.pipeline import ClaimGenerationPipeline
    from relcheck_v3.kb.builder import KnowledgeBase

logger = logging.getLogger(__name__)


class RelCheckCaptionEditor:
    """Full RelCheck correction pipeline as a pluggable ``Caption_Editor``.

    Usage::

        editor = RelCheckCaptionEditor(
            claim_gen_config=ClaimGenConfig(openai_api_key="sk-..."),
            correction_config=CorrectionConfig(openai_api_key="sk-..."),
        )
        corrected = editor.edit_caption("path/to/image.jpg", "A cat sitting on a dog.")

    The editor runs claim generation (Stages 1-4) to build a Visual
    Knowledge Base, then correction (Stage 5) to identify and fix the
    hallucination.  If any stage fails gracefully, the original caption
    is returned unchanged (safe passthrough).
    """

    def __init__(
        self,
        claim_gen_config: ClaimGenConfig | None = None,
        correction_config: CorrectionConfig | None = None,
    ) -> None:
        """Initialize the editor with claim generation and correction configs.

        Args:
            claim_gen_config: Configuration for Stages 1-4.  Defaults
                to ``ClaimGenConfig()`` (reads OPENAI_API_KEY from env).
            correction_config: Configuration for Stage 5.  Defaults to
                ``CorrectionConfig()`` (reads OPENAI_API_KEY from env).
        """
        # Lazy imports — claim_generation.pipeline pulls in torch/groundingdino
        # which are GPU-only deps not available in test / CI environments.
        from relcheck_v3.claim_generation.config import ClaimGenConfig
        from relcheck_v3.claim_generation.pipeline import ClaimGenerationPipeline

        self._claim_gen_config = claim_gen_config or ClaimGenConfig()
        self._correction_config = correction_config or CorrectionConfig()

        self._pipeline = ClaimGenerationPipeline(self._claim_gen_config)
        self._corrector = HallucinationCorrector(self._correction_config)

    def edit_caption(self, image: str | Image, ref_cap: str) -> str:
        """Run the full RelCheck pipeline on one image/caption pair.

        Implements the ``Caption_Editor`` protocol.

        Args:
            image: File path string or PIL Image object.
            ref_cap: The reference caption (potentially hallucinated).

        Returns:
            The corrected caption.  Returns ``ref_cap`` unchanged on any
            failure (claim generation error, KB build error, correction
            rejected by edit gate, etc.).
        """
        # ── Stages 1-4: Claim Generation → Visual Knowledge Base ──
        result = self._pipeline.process_single(
            image=image,
            ref_cap=ref_cap,
            image_id="correction",
        )

        if not result.success:
            logger.warning(
                "Claim generation failed: %s — returning original caption",
                result.error_message,
            )
            return ref_cap

        # ── KB Building (CLAIM + GEOM + SCENE) ──
        from relcheck_v3.kb.builder import build_kb

        try:
            kb = build_kb(
                vkb=result.visual_knowledge_base,
                object_answers=result.object_answers,
                image=self._resolve_image(image),
            )
        except Exception as exc:
            logger.warning("KB build failed: %s — returning original", exc)
            return ref_cap

        kb_text = kb.format()
        logger.info(
            "KB built: %d claims, %d spatial facts, %d scene triples",
            len(result.visual_knowledge_base.count_claims)
            + len(result.visual_knowledge_base.specific_claims)
            + len(result.visual_knowledge_base.overall_claims),
            len(kb.spatial_facts),
            len(kb.scene_graph),
        )

        # ── Stage 5: Correction ──
        corrected, hallucination = self._corrector.run(ref_cap, kb_text)

        if hallucination is not None:
            logger.info(
                "Caption corrected: %r → %r (span=%r, confidence=%s)",
                ref_cap[:60],
                corrected[:60],
                hallucination.hallucinated_span,
                hallucination.confidence,
            )
        else:
            logger.info("No correction applied (passthrough)")

        return corrected

    @property
    def corrector(self) -> HallucinationCorrector:
        """Access the underlying corrector for advanced use."""
        return self._corrector

    @property
    def pipeline(self) -> ClaimGenerationPipeline:
        """Access the claim generation pipeline for advanced use."""
        return self._pipeline

    @staticmethod
    def _resolve_image(image: str | Image) -> Image | None:
        """Convert image path to PIL Image if needed (for RelTR)."""
        from PIL import Image as PILImage

        if isinstance(image, PILImage.Image):
            return image
        try:
            return PILImage.open(image).convert("RGB")
        except Exception:
            return None
