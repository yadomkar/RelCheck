"""MME benchmark loader for the multi-benchmark evaluation harness.

Loads the four hallucination-relevant MME subtasks (existence, count, position,
color) and yields uniform ``BenchmarkSample`` objects.  Each subtask directory
contains a tab-separated text file with ``{image_name}\\t{question}\\t{answer}``
lines and an ``images/`` subdirectory with the corresponding image files.

MME uses its own images (not COCO), so ``reltr_tag`` is always ``None``.

Reference: https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models
           (MME benchmark, Fu et al. 2023)
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

from relcheck_v3.benchmarks.models import BenchmarkSample

logger = logging.getLogger(__name__)

_SUBTASKS: list[str] = ["existence", "count", "position", "color"]


class MMELoader:
    """Loads MME hallucination subtask samples.

    The expected directory layout under *mme_data_dir* is::

        mme_data_dir/
        ├── existence/
        │   ├── existence.txt
        │   └── images/
        │       ├── img1.jpg
        │       └── ...
        ├── count/
        │   ├── count.txt
        │   └── images/
        ├── position/
        │   ├── position.txt
        │   └── images/
        └── color/
            ├── color.txt
            └── images/
    """

    def iter_samples(self, mme_data_dir: str) -> Iterator[BenchmarkSample]:
        """Yield ``BenchmarkSample`` for all four MME hallucination subtasks.

        Args:
            mme_data_dir: Root directory containing the MME subtask
                directories (existence, count, position, color).

        Yields:
            :class:`BenchmarkSample` for each valid line in the subtask
            text files.

        Raises:
            FileNotFoundError: If a subtask directory or its text file is
                missing.
        """
        root = Path(mme_data_dir)

        for subtask in _SUBTASKS:
            subtask_dir = root / subtask
            if not subtask_dir.is_dir():
                raise FileNotFoundError(
                    f"MME subtask directory not found: {subtask_dir}"
                )

            txt_path = subtask_dir / f"{subtask}.txt"
            if not txt_path.is_file():
                raise FileNotFoundError(
                    f"MME subtask text file not found: {txt_path}"
                )

            images_dir = subtask_dir / "images"

            lines = txt_path.read_text(encoding="utf-8").splitlines()
            for idx, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue

                parts = line.split("\t")
                if len(parts) != 3:
                    logger.warning(
                        "Skipping malformed MME line %d in %s: %r",
                        idx,
                        txt_path,
                        line,
                    )
                    continue

                image_name, question, answer = parts
                image_path = str(images_dir / image_name)

                if not (images_dir / image_name).is_file():
                    logger.warning(
                        "MME image not found, skipping: %s", image_path
                    )
                    continue

                yield BenchmarkSample(
                    sample_id=f"mme_{subtask}_{image_name}_{idx}",
                    image_path=image_path,
                    question=question,
                    label=answer,
                    split=subtask,
                    benchmark="mme",
                    reltr_tag=None,
                    metadata={
                        "subtask": subtask,
                        "image_name": image_name,
                        "line_idx": idx,
                    },
                )
