"""DataLoader: COCO-EE and Flickr30K-EE annotation parsing."""

import json
import logging
import os

from relcheck_v3.hallucination_generation.models import CaptionRecord

logger = logging.getLogger(__name__)


class DataLoader:
    """Loads image-caption pairs from COCO-EE or Flickr30K-EE annotation files."""

    SUPPORTED_DATASETS = ("coco-ee", "flickr30k-ee")

    def load(
        self, dataset_name: str, annotation_path: str, image_dir: str
    ) -> list[CaptionRecord]:
        """Parse annotation file and resolve image paths.

        Args:
            dataset_name: "coco-ee" or "flickr30k-ee"
            annotation_path: Path to annotation JSON file
            image_dir: Directory containing source images

        Returns:
            List of CaptionRecord with image_id, caption, image_path

        Raises:
            FileNotFoundError: if annotation_path does not exist
            ValueError: if dataset_name is not supported
        """
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Unsupported dataset '{dataset_name}'. "
                f"Supported: {', '.join(self.SUPPORTED_DATASETS)}"
            )

        if not os.path.isfile(annotation_path):
            raise FileNotFoundError(
                f"Annotation file not found: {annotation_path}"
            )

        with open(annotation_path, "r", encoding="utf-8") as f:
            annotations = json.load(f)

        records: list[CaptionRecord] = []
        for entry in annotations:
            image_id = str(entry["image_id"])
            caption = entry["caption"]

            if dataset_name == "coco-ee":
                image_path = os.path.join(
                    image_dir, f"COCO_val2014_{int(image_id):012d}.jpg"
                )
            else:  # flickr30k-ee
                image_path = os.path.join(image_dir, f"{image_id}.jpg")

            if not os.path.isfile(image_path):
                logger.warning(
                    "Image file not found, skipping record: %s", image_path
                )
                continue

            records.append(
                CaptionRecord(
                    image_id=image_id, caption=caption, image_path=image_path
                )
            )

        return records
