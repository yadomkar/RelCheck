"""Grounding DINO object detector wrapper.

Uses the official GroundingDINO package (not the HuggingFace port) to match
the original Woodpecker codebase exactly. Includes IOU-based deduplication,
area filtering, and spaCy similarity matching for phrase-to-entity mapping.
"""

import logging

import numpy as np
import torch
from torchvision.ops import box_convert

from relcheck_v3.claim_generation.models import Detection, ObjectAnswer

logger = logging.getLogger(__name__)

# Thresholds matching Woodpecker exactly
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
AREA_THRESHOLD = 0.001   # filter out tiny detections
IOU_THRESHOLD = 0.95     # deduplicate overlapping boxes


def _compute_iou(box1: list[float], box2: list[float]) -> float:
    """Compute IoU between two [x1, y1, x2, y2] normalized boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def _in_dict(entity_dict: dict, norm_box: list[float]) -> bool:
    """Check if a box overlaps with any existing box in the entity dict."""
    for ent_info in entity_dict.values():
        if "bbox" in ent_info and any(
            _compute_iou(norm_box, box) > IOU_THRESHOLD
            for box in ent_info["bbox"]
        ):
            return True
    return False


def _find_most_similar_strings(
    nlp, source_strings: list[str], target_strings: list[str]
) -> list[str]:
    """Map detected phrases to the closest entity name using spaCy similarity.

    Exactly matches Woodpecker's find_most_similar_strings function.
    """
    target_docs = [nlp(text) for text in target_strings]

    def find_most_similar(source_str: str) -> str:
        source_doc = nlp(source_str)
        similarities = [
            (target_doc, target_doc.similarity(source_doc))
            for target_doc in target_docs
        ]
        most_similar_doc = max(similarities, key=lambda item: item[1])[0]
        return most_similar_doc.text

    return [find_most_similar(s) for s in source_strings]


class GroundingDINODetector:
    """Wraps the official GroundingDINO package, matching Woodpecker exactly.

    Uses groundingdino.util.inference.predict() with:
    - All entities passed as a single period-separated prompt
    - IOU-based deduplication (threshold 0.95)
    - Area filtering (threshold 0.001)
    - spaCy similarity matching for phrase-to-entity mapping
    """

    def __init__(
        self,
        detector_config: str,
        detector_model_path: str,
        device: str = "cuda:0",
    ) -> None:
        """Load GroundingDINO model from official checkpoint.

        Args:
            detector_config: Path to GroundingDINO config file
                (e.g. 'GroundingDINO_SwinT_OGC.py').
            detector_model_path: Path to model checkpoint
                (e.g. 'groundingdino_swint_ogc.pth').
            device: CUDA device string.

        Raises:
            RuntimeError: If CUDA is not available.
        """
        if not torch.cuda.is_available():
            raise RuntimeError(
                "GPU is required for Grounding DINO inference. "
                "CUDA is not available on this machine."
            )

        self.device = device
        from groundingdino.util.inference import load_model
        self.model = load_model(detector_config, detector_model_path, device=device)
        import spacy
        self.nlp = spacy.load("en_core_web_md")

        gpu_memory_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)
        logger.info(
            "Grounding DINO loaded on %s — GPU memory: %.1f MB",
            device,
            gpu_memory_mb,
        )

    def detect_objects(
        self,
        image_path: str,
        entity_str: str,
        box_threshold: float = BOX_THRESHOLD,
        text_threshold: float = TEXT_THRESHOLD,
        area_threshold: float = AREA_THRESHOLD,
    ) -> dict[str, ObjectAnswer]:
        """Detect all entities in an image in a single pass.

        Matches Woodpecker's Detector.detect_objects() exactly:
        - Passes all entities as a period-separated caption
        - Uses spaCy to map detected phrases back to entity names
        - Filters by area threshold and IOU deduplication

        Args:
            image_path: Path to the image file.
            entity_str: Period-separated entity string (e.g. "man.elephant").
            box_threshold: Confidence threshold for box predictions.
            text_threshold: Confidence threshold for text predictions.
            area_threshold: Minimum normalized area to keep a detection.

        Returns:
            Dict mapping entity name → ObjectAnswer with count and bboxes.
        """
        entity_list = [e.strip() for e in entity_str.split(".") if e.strip()]
        if not entity_list:
            return {}

        # Initialize entity dict (matches Woodpecker's defaultdict pattern)
        global_entity_dict: dict[str, dict] = {}
        for ent in entity_list:
            global_entity_dict.setdefault(ent, {
                "total_count": 0,
                "bbox": [],
            })

        # Load image using GroundingDINO's load_image (returns numpy + tensor)
        from groundingdino.util.inference import load_image, predict
        image_source, image_tensor = load_image(image_path)
        h, w, _ = image_source.shape

        # Run detection — single pass with all entities
        boxes, logits, phrases = predict(
            model=self.model,
            image=image_tensor,
            caption=entity_str,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device,
        )

        # Convert cxcywh → xyxy in pixel coords, then normalize
        boxes_pixel = boxes * torch.Tensor([w, h, w, h])
        xyxy = box_convert(boxes=boxes_pixel, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        normed_xyxy = np.around(
            np.clip(xyxy / np.array([w, h, w, h]), 0.0, 1.0), 3
        ).tolist()

        # Map detected phrases to entity names using spaCy similarity
        phrases = _find_most_similar_strings(self.nlp, phrases, entity_list)

        # Extract detections with filtering (matches Woodpecker exactly)
        for entity, norm_box in zip(phrases, normed_xyxy):
            # Filter out too-small objects
            box_area = (norm_box[2] - norm_box[0]) * (norm_box[3] - norm_box[1])
            if box_area < area_threshold:
                continue
            # Filter out duplicates by IOU
            if _in_dict(global_entity_dict, norm_box):
                continue

            global_entity_dict[entity]["total_count"] += 1
            global_entity_dict[entity]["bbox"].append(norm_box)

        # Convert to ObjectAnswer models
        result: dict[str, ObjectAnswer] = {}
        for ent in entity_list:
            info = global_entity_dict[ent]
            result[ent] = ObjectAnswer(
                object_name=ent,
                count=info["total_count"],
                bboxes=info["bbox"],
            )

        return result
