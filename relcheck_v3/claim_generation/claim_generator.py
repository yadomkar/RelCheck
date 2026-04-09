"""Stage 4: Visual claim generation (Q&A to Visual Knowledge Base).

Uses the local khhuang/zerofec-qa2claim-t5-base T5 model for QA-to-claim
conversion, matching the original Woodpecker codebase exactly.
"""

import logging

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from relcheck_v3.claim_generation.models import (
    AttributeQA,
    CountClaim,
    ObjectAnswer,
    OverallClaim,
    SpecificClaim,
    VisualKnowledgeBase,
)

logger = logging.getLogger(__name__)


class QA2ClaimModel:
    """Local T5-base Seq2Seq model for QA-to-claim conversion.

    Loads khhuang/zerofec-qa2claim-t5-base (222M params) onto CUDA GPU.
    Input format: "{answer} \\n {question}" — matching the Woodpecker code.
    """

    def __init__(
        self,
        model_id: str = "khhuang/zerofec-qa2claim-t5-base",
        device: str = "cuda",
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "GPU is required for QA2Claim model inference. "
                "CUDA is not available on this machine."
            )
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)

        gpu_memory_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)
        logger.info(
            "QA2Claim model loaded on %s — GPU memory: %.1f MB",
            device,
            gpu_memory_mb,
        )

    def convert(self, question: str, answer: str) -> str:
        """Convert a QA pair to a declarative claim sentence.

        Input format matches the Woodpecker codebase exactly:
        "{answer} \\n {question}"

        Args:
            question: The verification question.
            answer: The answer to the question.

        Returns:
            Declarative claim sentence.
        """
        input_text = f"{answer} \\n {question}"
        input_ids = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=512,
        ).input_ids.to(self.model.device)

        generated_ids = self.model.generate(
            input_ids,
            max_length=64,
            num_beams=4,
            early_stopping=True,
        )
        result = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        return result[0] if result else ""


class ClaimGenerator:
    """Converts Q&A results into a structured Visual Knowledge Base.

    Builds three sections:
    - Count claims: object existence and counts with bounding boxes
    - Specific claims: per-instance attribute claims (single-entity QAs)
    - Overall claims: inter-object relationship claims (multi-entity QAs)

    All QA-to-claim conversions use the local T5 model.
    """

    def __init__(self, qa2claim: QA2ClaimModel) -> None:
        """Store the QA2Claim model instance."""
        self._qa2claim = qa2claim

    def generate(
        self,
        object_answers: dict[str, ObjectAnswer],
        attribute_qas: list[AttributeQA],
    ) -> VisualKnowledgeBase:
        """Build a Visual Knowledge Base from object answers and attribute QAs.

        Args:
            object_answers: Mapping of object_name → ObjectAnswer with
                count and bounding boxes from Stage 3a.
            attribute_qas: List of AttributeQA from Stage 3b, each with
                question, entities, and answer.

        Returns:
            VisualKnowledgeBase with count_claims, specific_claims,
            and overall_claims sections.
        """
        count_claims = self._build_count_claims(object_answers)
        specific_claims, overall_claims = self._build_attribute_claims(
            attribute_qas, object_answers
        )

        return VisualKnowledgeBase(
            count_claims=count_claims,
            specific_claims=specific_claims,
            overall_claims=overall_claims,
        )

    @staticmethod
    def _build_count_claims(
        object_answers: dict[str, ObjectAnswer],
    ) -> list[CountClaim]:
        """Generate count claims from object detection results."""
        claims: list[CountClaim] = []
        for name, answer in object_answers.items():
            if answer.count == 0:
                text = f"There is no {name}."
            elif answer.count == 1:
                text = f"There is 1 {name}."
            else:
                text = f"There are {answer.count} {name}."

            claims.append(
                CountClaim(
                    object_name=name,
                    count=answer.count,
                    claim_text=text,
                    bboxes=answer.bboxes,
                )
            )
        return claims

    def _build_attribute_claims(
        self,
        attribute_qas: list[AttributeQA],
        object_answers: dict[str, ObjectAnswer],
    ) -> tuple[list[SpecificClaim], list[OverallClaim]]:
        """Route attribute QAs to Specific or Overall claims."""
        specific_claims: list[SpecificClaim] = []
        overall_claims: list[OverallClaim] = []

        for qa in attribute_qas:
            claim_text = self._qa_to_claim(qa.question, qa.answer)
            if claim_text is None:
                continue

            if len(qa.entities) == 1:
                entity_name = qa.entities[0]
                bbox, instance_index = self._resolve_bbox(
                    entity_name, object_answers
                )
                specific_claims.append(
                    SpecificClaim(
                        object_name=entity_name,
                        instance_index=instance_index,
                        bbox=bbox,
                        claim_text=claim_text,
                    )
                )
            else:
                overall_claims.append(OverallClaim(claim_text=claim_text))

        return specific_claims, overall_claims

    def _qa_to_claim(self, question: str, answer: str) -> str | None:
        """Convert a QA pair to a declarative claim via the local T5 model.

        Returns None on failure (logs a warning).
        """
        try:
            claim = self._qa2claim.convert(question, answer)
            if not claim:
                logger.warning(
                    "Empty claim from QA2Claim for Q: '%s' A: '%s'. Skipping.",
                    question,
                    answer,
                )
                return None
            return claim
        except Exception:
            logger.warning(
                "QA-to-claim conversion failed for Q: '%s' A: '%s'. Skipping.",
                question,
                answer,
            )
            return None

    @staticmethod
    def _resolve_bbox(
        entity_name: str,
        object_answers: dict[str, ObjectAnswer],
    ) -> tuple[list[float] | None, int | None]:
        """Look up the first bounding box for an entity from object answers."""
        answer = object_answers.get(entity_name)
        if answer is None or not answer.bboxes:
            return None, None
        return answer.bboxes[0], 1
