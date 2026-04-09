"""ClaimGenerationPipeline: synchronous end-to-end orchestration."""

import logging
import os
import time

from PIL import Image
from tqdm import tqdm

from relcheck_v3.claim_generation.claim_generator import ClaimGenerator, QA2ClaimModel
from relcheck_v3.claim_generation.concept_extractor import KeyConceptExtractor
from relcheck_v3.claim_generation.config import ClaimGenConfig
from relcheck_v3.claim_generation.grounding_dino import GroundingDINODetector
from relcheck_v3.claim_generation.models import (
    AttributeQA,
    AttributeQuestion,
    InputSample,
    ObjectAnswer,
    SampleResult,
    StageTimings,
    VisualKnowledgeBase,
)
from relcheck_v3.claim_generation.openai_client import OpenAIClient
from relcheck_v3.claim_generation.question_formulator import QuestionFormulator
from relcheck_v3.claim_generation.result_store import ResultStore
from relcheck_v3.claim_generation.visual_validator import VisualValidator

logger = logging.getLogger(__name__)


class ClaimGenerationPipeline:
    """Orchestrates Stages 1–4 of the Woodpecker claim generation pipeline.

    Chains KeyConceptExtractor → QuestionFormulator → VisualValidator →
    ClaimGenerator to build a Visual Knowledge Base from an image and caption.
    Supports single-sample and batch processing with checkpointing.
    """

    def __init__(self, config: ClaimGenConfig) -> None:
        """Initialize pipeline components from configuration.

        Args:
            config: Pipeline configuration with API key, model IDs,
                thresholds, output directory, and checkpoint settings.

        Raises:
            ValueError: If no OpenAI API key is configured (neither in
                config nor in the OPENAI_API_KEY environment variable).
        """
        # Validate API key before initializing anything
        resolved_key = config.openai_api_key or os.environ.get(
            "OPENAI_API_KEY", ""
        )
        if not resolved_key:
            raise ValueError(
                "OpenAI API key not configured. "
                "Set openai_api_key in ClaimGenConfig or the "
                "OPENAI_API_KEY environment variable."
            )

        self._config = config

        # Shared clients
        self._openai = OpenAIClient(
            api_key=resolved_key, model=config.gpt_model_id
        )
        self._detector = GroundingDINODetector(
            model_id=config.grounding_dino_model_id
        )
        self._qa2claim = QA2ClaimModel(
            model_id=config.qa2claim_model_id
        )

        # Stage components
        self._extractor = KeyConceptExtractor(self._openai)
        self._formulator = QuestionFormulator(self._openai)
        self._validator = VisualValidator(self._openai, self._detector)
        self._generator = ClaimGenerator(self._qa2claim)

        # Result persistence
        self._store = ResultStore(
            output_dir=config.output_dir,
            checkpoint_interval=config.checkpoint_interval,
        )

    # ------------------------------------------------------------------
    # Single-sample processing
    # ------------------------------------------------------------------

    def process_single(
        self,
        image: str | Image.Image,
        ref_cap: str,
        image_id: str = "unknown",
    ) -> SampleResult:
        """Run Stages 1–4 on a single image/caption pair.

        Args:
            image: File path string or PIL Image object.
            ref_cap: The reference caption to verify.
            image_id: Identifier for the sample (used in results).

        Returns:
            SampleResult with all intermediates, VKB, timings, and
            success status. On stage failure, returns a partial result
            with success=False and error_message set.
        """
        total_start = time.monotonic()
        timings = StageTimings()

        # Defaults for partial results on failure
        key_concepts: list[str] = []
        object_questions: list[str] = []
        attribute_questions: list[AttributeQuestion] = []
        object_answers: dict[str, ObjectAnswer] = {}
        attribute_answers: list[AttributeQA] = []
        vkb = VisualKnowledgeBase(
            count_claims=[], specific_claims=[], overall_claims=[]
        )
        vkb_text = ""

        try:
            # Stage 1: Key Concept Extraction
            t0 = time.monotonic()
            key_concepts = self._extractor.extract(ref_cap)
            timings.stage1_seconds = time.monotonic() - t0
            logger.info(
                "[%s] Stage 1 done in %.2fs — %d concepts",
                image_id,
                timings.stage1_seconds,
                len(key_concepts),
            )

            # Stage 2: Question Formulation
            t0 = time.monotonic()
            questions = self._formulator.formulate(ref_cap, key_concepts)
            object_questions = [q.question for q in questions.object_questions]
            attribute_questions = questions.attribute_questions
            timings.stage2_seconds = time.monotonic() - t0
            logger.info(
                "[%s] Stage 2 done in %.2fs — %d object Qs, %d attribute Qs",
                image_id,
                timings.stage2_seconds,
                len(object_questions),
                len(attribute_questions),
            )

            # Stage 3: Visual Validation
            t0 = time.monotonic()

            # 3a: Object-level validation
            for concept in key_concepts:
                answer = self._validator.validate_object(image, concept)
                object_answers[concept] = answer

            # 3b: Attribute-level validation
            for aq in attribute_questions:
                answer_text = self._validator.validate_attribute(
                    image, aq.question
                )
                attribute_answers.append(
                    AttributeQA(
                        question=aq.question,
                        entities=aq.entities,
                        answer=answer_text,
                    )
                )

            timings.stage3_seconds = time.monotonic() - t0
            logger.info(
                "[%s] Stage 3 done in %.2fs — %d objects, %d attributes",
                image_id,
                timings.stage3_seconds,
                len(object_answers),
                len(attribute_answers),
            )

            # Stage 4: Claim Generation
            t0 = time.monotonic()
            vkb = self._generator.generate(object_answers, attribute_answers)
            vkb_text = vkb.format()
            timings.stage4_seconds = time.monotonic() - t0
            logger.info(
                "[%s] Stage 4 done in %.2fs — %d count, %d specific, %d overall claims",
                image_id,
                timings.stage4_seconds,
                len(vkb.count_claims),
                len(vkb.specific_claims),
                len(vkb.overall_claims),
            )

            timings.total_seconds = time.monotonic() - total_start

            return SampleResult(
                image_id=image_id,
                ref_cap=ref_cap,
                key_concepts=key_concepts,
                object_questions=object_questions,
                attribute_questions=attribute_questions,
                object_answers=object_answers,
                attribute_answers=attribute_answers,
                visual_knowledge_base=vkb,
                vkb_text=vkb_text,
                timings=timings,
                success=True,
            )

        except Exception as exc:
            timings.total_seconds = time.monotonic() - total_start
            error_msg = f"{type(exc).__name__}: {exc}"
            logger.error(
                "[%s] Pipeline failed: %s", image_id, error_msg, exc_info=True
            )

            return SampleResult(
                image_id=image_id,
                ref_cap=ref_cap,
                key_concepts=key_concepts,
                object_questions=object_questions,
                attribute_questions=attribute_questions,
                object_answers=object_answers,
                attribute_answers=attribute_answers,
                visual_knowledge_base=vkb,
                vkb_text=vkb_text,
                timings=timings,
                success=False,
                error_message=error_msg,
            )

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def process_batch(
        self, samples: list[InputSample]
    ) -> list[SampleResult]:
        """Process a batch of samples with checkpointing and progress bar.

        Loads any existing checkpoint and resumes from the first
        unprocessed sample. Appends each result to the ResultStore
        immediately. Exports CSV and JSONL on completion.

        Args:
            samples: List of InputSample objects to process.

        Returns:
            List of SampleResult objects for all samples (including
            previously checkpointed results).
        """
        # Respect max_samples config
        if self._config.max_samples is not None:
            samples = samples[: self._config.max_samples]

        # Load checkpoint to skip already-processed samples
        checkpoint = self._store.load_checkpoint()
        logger.info(
            "Loaded checkpoint with %d completed samples", len(checkpoint)
        )

        results: list[SampleResult] = []
        processed_count = 0

        for sample in tqdm(samples, desc="Claim generation"):
            # Skip already-processed samples
            if sample.image_id in checkpoint:
                results.append(checkpoint[sample.image_id])
                continue

            result = self.process_single(
                image=sample.image_path,
                ref_cap=sample.ref_cap,
                image_id=sample.image_id,
            )
            results.append(result)
            self._store.append(result)
            processed_count += 1

        # Export final outputs
        self._store.export_csv()
        self._store.export_jsonl()
        logger.info(
            "Batch complete: %d processed, %d from checkpoint, %d total",
            processed_count,
            len(checkpoint),
            len(results),
        )

        return results
