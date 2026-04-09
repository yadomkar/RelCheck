"""Stage 3: Visual knowledge validation via Grounding DINO and GPT-5.4-mini VQA."""

import logging

from PIL import Image

from relcheck_v3.claim_generation.grounding_dino import GroundingDINODetector
from relcheck_v3.claim_generation.models import ObjectAnswer
from relcheck_v3.claim_generation.openai_client import OpenAIClient

logger = logging.getLogger(__name__)

_ATTRIBUTE_VQA_SYSTEM = (
    "You are a visual question answering assistant. "
    "Answer the question about the image concisely."
)


class VisualValidator:
    """Answers verification questions by querying the image.

    Object-level questions are answered via Grounding DINO (local GPU).
    Attribute-level questions are answered via GPT-5.4-mini multimodal VQA.
    """

    def __init__(
        self, client: OpenAIClient, detector: GroundingDINODetector
    ) -> None:
        self._client = client
        self._detector = detector

    def validate_objects(
        self,
        image_path: str,
        concepts: list[str],
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> dict[str, ObjectAnswer]:
        """Detect all concepts in the image in a single pass.

        Matches Woodpecker's Detector.detect_objects() — passes all
        entities as a period-separated prompt for joint detection.

        Args:
            image_path: Path to the image file.
            concepts: List of entity names to detect.
            box_threshold: Confidence threshold for boxes.
            text_threshold: Confidence threshold for text.

        Returns:
            Dict mapping entity name → ObjectAnswer with count and bboxes.
            On detector exception, returns count=0 for all concepts.
        """
        if not concepts:
            return {}

        entity_str = ".".join(concepts)

        try:
            return self._detector.detect_objects(
                image_path=image_path,
                entity_str=entity_str,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )
        except Exception:
            logger.error(
                "Grounding DINO failed for entities '%s'",
                entity_str,
                exc_info=True,
            )
            return {
                c: ObjectAnswer(object_name=c, count=0, bboxes=[])
                for c in concepts
            }

    def validate_attribute(
        self, image: str | Image.Image, question: str
    ) -> str:
        """Answer an attribute question about the image via GPT-5.4-mini VQA.

        Returns "unknown" if the API call fails after retry exhaustion.
        """
        try:
            return self._client.chat_with_image(
                _ATTRIBUTE_VQA_SYSTEM, question, image
            )
        except Exception:
            logger.error(
                "Attribute VQA failed for question '%s'",
                question,
                exc_info=True,
            )
            return "unknown"
