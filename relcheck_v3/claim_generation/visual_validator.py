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
        """Store the OpenAI client and Grounding DINO detector instances."""
        self._client = client
        self._detector = detector

    def validate_object(
        self, image: str | Image.Image, object_name: str
    ) -> ObjectAnswer:
        """Detect object instances in the image using Grounding DINO.

        Args:
            image: File path string or PIL Image object.
            object_name: Name of the object to detect.

        Returns:
            ObjectAnswer with count and bounding boxes.
            On detector exception, returns count=0 and empty bboxes.
        """
        try:
            detections = self._detector.detect(image, object_name)
        except Exception:
            logger.error(
                "Grounding DINO failed for object '%s'", object_name, exc_info=True
            )
            return ObjectAnswer(object_name=object_name, count=0, bboxes=[])

        bboxes = [d.bbox for d in detections]
        return ObjectAnswer(
            object_name=object_name,
            count=len(bboxes),
            bboxes=bboxes,
        )

    def validate_attribute(
        self, image: str | Image.Image, question: str
    ) -> str:
        """Answer an attribute question about the image via GPT-5.4-mini VQA.

        Args:
            image: File path string or PIL Image object.
            question: The attribute-level verification question.

        Returns:
            Short text answer from the model, or "unknown" if the API
            call fails after retry exhaustion.
        """
        try:
            return self._client.chat_with_image(
                _ATTRIBUTE_VQA_SYSTEM, question, image
            )
        except Exception:
            logger.error(
                "Attribute VQA failed for question '%s'", question, exc_info=True
            )
            return "unknown"
