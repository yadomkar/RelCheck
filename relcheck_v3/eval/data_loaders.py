"""Data loaders for CE test sets and POPE benchmark."""

import json
import logging
import os

from relcheck_v3.eval.models import CESample, POPEDomain, POPEQuestion, POPESetting

logger = logging.getLogger(__name__)


class CEDataLoader:
    """Loads COCO-CE and Flickr30K-CE test sets for caption editing evaluation.

    Each test set is a JSON file containing records with image_id, gt_cap,
    and ref_cap fields. Image paths are resolved by checking for explicit
    path fields in the record, then falling back to common naming patterns.
    """

    def load(self, test_set_path: str, image_dir: str) -> list[CESample]:
        """Parse CE test set file and resolve image paths.

        Args:
            test_set_path: Path to the CE test set JSON file.
            image_dir: Directory containing source images.

        Returns:
            List of CESample with validated fields and resolved image paths.

        Raises:
            FileNotFoundError: If test_set_path does not exist.
        """
        if not os.path.isfile(test_set_path):
            raise FileNotFoundError(
                f"CE test set file not found: {test_set_path}"
            )

        with open(test_set_path, "r", encoding="utf-8") as f:
            entries = json.load(f)

        samples: list[CESample] = []
        for entry in entries:
            image_id = str(entry.get("image_id", ""))
            gt_cap = entry.get("gt_cap", "")
            ref_cap = entry.get("ref_cap", "")

            # Validate non-empty captions
            if not gt_cap or not gt_cap.strip():
                logger.warning(
                    "Empty GT-Cap for image_id=%s, skipping record", image_id
                )
                continue
            if not ref_cap or not ref_cap.strip():
                logger.warning(
                    "Empty Ref-Cap for image_id=%s, skipping record", image_id
                )
                continue

            # Resolve image path
            image_path = self._resolve_image_path(entry, image_id, image_dir)

            if not os.path.isfile(image_path):
                logger.warning(
                    "Image file not found, skipping record: %s", image_path
                )
                continue

            samples.append(
                CESample(
                    image_id=image_id,
                    gt_cap=gt_cap,
                    ref_cap=ref_cap,
                    image_path=image_path,
                )
            )

        return samples

    @staticmethod
    def _resolve_image_path(
        entry: dict, image_id: str, image_dir: str
    ) -> str:
        """Resolve the image file path for a test set entry.

        Checks in order:
        1. Explicit 'image_path' field in the entry
        2. Explicit 'file_name' field in the entry
        3. COCO naming pattern: COCO_val2014_{image_id:012d}.jpg
        4. Simple pattern: {image_id}.jpg
        """
        # 1. Explicit image_path
        if "image_path" in entry and entry["image_path"]:
            path = entry["image_path"]
            if os.path.isabs(path):
                return path
            return os.path.join(image_dir, path)

        # 2. Explicit file_name
        if "file_name" in entry and entry["file_name"]:
            return os.path.join(image_dir, entry["file_name"])

        # 3. COCO naming pattern
        try:
            coco_name = f"COCO_val2014_{int(image_id):012d}.jpg"
            coco_path = os.path.join(image_dir, coco_name)
            if os.path.isfile(coco_path):
                return coco_path
        except (ValueError, TypeError):
            pass

        # 4. Simple pattern
        return os.path.join(image_dir, f"{image_id}.jpg")


class POPEDataLoader:
    """Loads POPE benchmark question files for all 9 domain×setting combinations.

    Question files are expected at ``{pope_data_dir}/{domain}_pope_{setting}.json``
    where each line is a JSON object with ``image``, ``text``, and ``label`` fields.
    Image paths are resolved by joining the per-domain image directory with the
    image filename from each record.
    """

    def load(
        self,
        pope_data_dir: str,
        image_dirs: dict[POPEDomain, str],
    ) -> dict[tuple[POPEDomain, POPESetting], list[POPEQuestion]]:
        """Load POPE question files for all 9 domain×setting combinations.

        Args:
            pope_data_dir: Directory containing POPE question JSON files.
            image_dirs: Mapping from each POPEDomain to the directory
                containing that domain's images.

        Returns:
            Dictionary keyed by (domain, setting) tuples, each mapping to
            a list of POPEQuestion objects parsed from the corresponding file.

        Raises:
            FileNotFoundError: If any of the 9 expected question files is missing.
        """
        result: dict[tuple[POPEDomain, POPESetting], list[POPEQuestion]] = {}

        for domain in POPEDomain:
            for setting in POPESetting:
                filename = f"{domain.value}_pope_{setting.value}.json"
                filepath = os.path.join(pope_data_dir, filename)

                if not os.path.isfile(filepath):
                    raise FileNotFoundError(
                        f"POPE question file not found: {filepath}"
                    )

                image_dir = image_dirs.get(domain, "")
                questions = self._load_question_file(
                    filepath, image_dir, domain, setting
                )
                result[(domain, setting)] = questions

        return result

    @staticmethod
    def _load_question_file(
        filepath: str,
        image_dir: str,
        domain: POPEDomain,
        setting: POPESetting,
    ) -> list[POPEQuestion]:
        """Parse a single POPE question file.

        Each line is a JSON object with at minimum ``image``, ``text``,
        and ``label`` fields.
        """
        questions: list[POPEQuestion] = []

        with open(filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(
                        "Skipping malformed JSON at %s:%d", filepath, line_num
                    )
                    continue

                image_filename = record.get("image", "")
                question_text = record.get("text", "")
                label = record.get("label", "").strip().lower()

                if not image_filename or not question_text or label not in ("yes", "no"):
                    logger.warning(
                        "Skipping invalid POPE record at %s:%d", filepath, line_num
                    )
                    continue

                image_path = os.path.join(image_dir, image_filename)
                image_id = os.path.splitext(os.path.basename(image_filename))[0]

                questions.append(
                    POPEQuestion(
                        image_id=image_id,
                        question=question_text,
                        ground_truth=label,
                        image_path=image_path,
                        domain=domain,
                        setting=setting,
                    )
                )

        return questions
