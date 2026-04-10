"""AMBER benchmark loader for the multi-benchmark evaluation harness.

Loads all four AMBER evaluation tasks — three discriminative (existence,
attribute, relation) and one generative — and yields uniform
``BenchmarkSample`` objects.

Discriminative JSONs contain entries with ``id``, ``image``, ``query``, and
``answer`` fields.  The generative JSON contains entries with ``id`` and
``image`` only (no question — the MLLM describes the image freely).

AMBER uses its own images (not COCO), so ``reltr_tag`` is always ``None``.

Reference: https://github.com/junyangwang0410/AMBER (Wang et al. 2023)
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

from relcheck_v3.benchmarks.models import BenchmarkSample

logger = logging.getLogger(__name__)

# Mapping from JSON filename suffix to short split code used in sample_id
# and the ``split`` field.
_AMBER_FILES: dict[str, str] = {
    "query_discriminative-existence.json": "de",
    "query_discriminative-attribute.json": "da",
    "query_discriminative-relation.json": "dr",
    "query_generative.json": "g",
}


class AMBERLoader:
    """Loads AMBER benchmark samples for all four evaluation tasks.

    The expected layout is::

        amber_data_dir/
        ├── query_discriminative-existence.json
        ├── query_discriminative-attribute.json
        ├── query_discriminative-relation.json
        └── query_generative.json

        amber_image_dir/
        ├── image1.jpg
        ├── image2.jpg
        └── ...
    """

    def iter_samples(
        self,
        amber_data_dir: str,
        amber_image_dir: str,
    ) -> Iterator[BenchmarkSample]:
        """Yield ``BenchmarkSample`` for all AMBER tasks.

        Args:
            amber_data_dir: Directory containing the four AMBER query
                JSON files.
            amber_image_dir: Directory containing AMBER images.

        Yields:
            :class:`BenchmarkSample` for each entry in the JSON files.

        Raises:
            FileNotFoundError: If any of the four AMBER data files is
                missing.
        """
        data_root = Path(amber_data_dir)
        image_root = Path(amber_image_dir)

        for filename, split in _AMBER_FILES.items():
            filepath = data_root / filename
            if not filepath.is_file():
                raise FileNotFoundError(
                    f"AMBER data file not found: {filepath}"
                )

            with open(filepath, "r", encoding="utf-8") as fh:
                entries: list[dict] = json.load(fh)

            is_generative = split == "g"

            for entry in entries:
                entry_id = entry["id"]
                image_rel = entry["image"]
                image_path = str(image_root / image_rel)

                if not (image_root / image_rel).is_file():
                    logger.warning(
                        "AMBER image not found, skipping: %s", image_path
                    )
                    continue

                if is_generative:
                    question = ""
                    label = ""
                else:
                    question = entry.get("query", "")
                    raw_answer = entry.get("answer", "")
                    label = str(raw_answer).strip().lower()

                yield BenchmarkSample(
                    sample_id=f"amber_{split}_{entry_id}",
                    image_path=image_path,
                    question=question,
                    label=label,
                    split=split,
                    benchmark="amber",
                    reltr_tag=None,
                    metadata={
                        "amber_id": entry_id,
                        "image_rel": image_rel,
                    },
                )
