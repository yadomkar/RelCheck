"""Grounding DINO object detector wrapper."""

import logging
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from relcheck_v3.claim_generation.models import Detection

logger = logging.getLogger(__name__)


class GroundingDINODetector:
    """Wraps Grounding DINO loaded from HuggingFace onto CUDA GPU."""

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-base",
        device: str = "cuda",
    ) -> None:
        """Load Grounding DINO model from HuggingFace onto CUDA GPU.

        Args:
            model_id: HuggingFace model identifier.
            device: Target device (must be CUDA).

        Raises:
            RuntimeError: If CUDA is not available.
        """
        if not torch.cuda.is_available():
            raise RuntimeError(
                "GPU is required for Grounding DINO inference. "
                "CUDA is not available on this machine."
            )

        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id
        ).to(device)

        gpu_memory_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)
        logger.info(
            "Grounding DINO loaded on %s — GPU memory: %.1f MB",
            device,
            gpu_memory_mb,
        )

    def detect(
        self,
        image: str | Image.Image,
        text_prompt: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> list[Detection]:
        """Detect objects matching text_prompt in the image.

        Args:
            image: File path string or PIL Image object.
            text_prompt: Text description of objects to detect.
            box_threshold: Confidence threshold for box predictions.
            text_threshold: Confidence threshold for text predictions.

        Returns:
            List of Detection objects with normalized bbox coordinates.
        """
        # Convert file path to PIL Image if needed
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise TypeError(f"Expected str or PIL Image, got {type(image)}")

        width, height = image.size

        # Process inputs
        inputs = self.processor(
            images=image, text=text_prompt, return_tensors="pt"
        ).to(self.device)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process: pass target_sizes so boxes come back in pixel coords
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs["input_ids"],
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=torch.tensor([[height, width]], device=self.device),
        )

        detections: list[Detection] = []
        if results:
            result = results[0]  # single image batch
            boxes = result["boxes"]  # tensor of shape (N, 4) in pixel coords
            scores = result["scores"]  # tensor of shape (N,)

            for box, score in zip(boxes, scores):
                # Normalize to [0, 1] relative to image dimensions
                x_min = box[0].item() / width
                y_min = box[1].item() / height
                x_max = box[2].item() / width
                y_max = box[3].item() / height

                # Clamp to [0.0, 1.0]
                x_min = max(0.0, min(1.0, x_min))
                y_min = max(0.0, min(1.0, y_min))
                x_max = max(0.0, min(1.0, x_max))
                y_max = max(0.0, min(1.0, y_max))

                detections.append(
                    Detection(
                        bbox=[x_min, y_min, x_max, y_max],
                        confidence=score.item(),
                    )
                )

        return detections
