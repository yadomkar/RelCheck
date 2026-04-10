"""POPE benchmark loader for the multi-benchmark evaluation harness.

Wraps the existing ``eval.data_loaders.POPEDataLoader`` to yield uniform
``BenchmarkSample`` objects.  Only the COCO domain is loaded (3 settings:
random, popular, adversarial) since the harness evaluates on COCO val2014.

Optionally computes a ``reltr_tag`` per sample from COCO instance annotations,
indicating whether the image contains at least one object category that maps
to the RelTR Visual Genome vocabulary.  Images are **never** dropped based on
RelTR coverage — the tag is metadata for stratified reporting only.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

from relcheck_v3.benchmarks.models import BenchmarkSample
from relcheck_v3.eval.data_loaders import POPEDataLoader
from relcheck_v3.eval.models import POPEDomain, POPESetting

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# COCO → RelTR vocabulary mapping
# ---------------------------------------------------------------------------
# 38 exact matches plus clear synonyms, consistent with the mapping used in
# ``RelCheck_Full_Pipeline_cells.py``.  The target values are entries from
# ``relcheck_v3.reltr.reltr.RELTR_OBJECT_CLASSES``.

COCO_TO_RELTR: dict[str, str] = {
    # Exact matches (38)
    "person": "person",
    "car": "car",
    "motorcycle": "motorcycle",
    "airplane": "airplane",
    "bus": "bus",
    "train": "train",
    "truck": "truck",
    "boat": "boat",
    "bench": "bench",
    "bird": "bird",
    "cat": "cat",
    "dog": "dog",
    "horse": "horse",
    "sheep": "sheep",
    "cow": "cow",
    "elephant": "elephant",
    "bear": "bear",
    "zebra": "zebra",
    "giraffe": "giraffe",
    "umbrella": "umbrella",
    "tie": "tie",
    "kite": "kite",
    "skateboard": "skateboard",
    "surfboard": "surfboard",
    "bottle": "bottle",
    "cup": "cup",
    "fork": "fork",
    "bowl": "bowl",
    "banana": "banana",
    "orange": "orange",
    "pizza": "pizza",
    "chair": "chair",
    "bed": "bed",
    "toilet": "toilet",
    "laptop": "laptop",
    "sink": "sink",
    "book": "book",
    "clock": "clock",
    "vase": "vase",
    # Clear synonyms
    "bicycle": "bike",
    "skis": "ski",
    "tennis racket": "racket",
    "wine glass": "glass",
    "baseball glove": "glove",
    "dining table": "table",
    "potted plant": "plant",
    "tv": "screen",
    "couch": "seat",
    "backpack": "bag",
    "handbag": "bag",
    "suitcase": "bag",
    "cell phone": "phone",
}

_SETTING_TO_SPLIT: dict[POPESetting, str] = {
    POPESetting.RANDOM: "random",
    POPESetting.POPULAR: "popular",
    POPESetting.ADVERSARIAL: "adversarial",
}


def _build_reltr_tag_map(
    coco_instances_path: str,
) -> dict[int, bool]:
    """Build a mapping from COCO image_id to RelTR vocabulary overlap flag.

    Args:
        coco_instances_path: Path to the COCO instances JSON file
            (e.g. ``instances_val2014.json``).

    Returns:
        Dictionary mapping each annotated image_id to ``True`` if at least
        one of its annotated categories maps to a RelTR object class via
        :data:`COCO_TO_RELTR`, ``False`` otherwise.

    Raises:
        FileNotFoundError: If *coco_instances_path* does not exist.
    """
    path = Path(coco_instances_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"COCO instances annotation not found: {coco_instances_path}"
        )

    with open(path, "r", encoding="utf-8") as fh:
        instances = json.load(fh)

    cat_id_to_name: dict[int, str] = {
        c["id"]: c["name"] for c in instances["categories"]
    }

    # image_id → set of COCO category names
    image_categories: dict[int, set[str]] = {}
    for ann in instances["annotations"]:
        img_id: int = ann["image_id"]
        cat_name = cat_id_to_name.get(ann["category_id"], "")
        if cat_name:
            image_categories.setdefault(img_id, set()).add(cat_name)

    reltr_set = set(COCO_TO_RELTR.values())
    tag_map: dict[int, bool] = {}
    for img_id, cats in image_categories.items():
        mapped = {COCO_TO_RELTR.get(c) for c in cats} - {None}
        tag_map[img_id] = bool(mapped & reltr_set)

    logger.info(
        "RelTR tag map: %d images, %d with RelTR overlap",
        len(tag_map),
        sum(tag_map.values()),
    )
    return tag_map


class POPELoader:
    """Loads POPE benchmark samples for the COCO domain (3 settings).

    Wraps :class:`~relcheck_v3.eval.data_loaders.POPEDataLoader` and converts
    its ``POPEQuestion`` objects into the uniform ``BenchmarkSample`` schema.
    """

    def iter_samples(
        self,
        pope_data_dir: str,
        coco_image_dir: str,
        coco_instances_path: str | None = None,
    ) -> Iterator[BenchmarkSample]:
        """Yield ``BenchmarkSample`` for all three POPE COCO splits.

        Internally delegates to
        :meth:`~relcheck_v3.eval.data_loaders.POPEDataLoader.load` which
        parses ``coco_pope_random.json``, ``coco_pope_popular.json``, and
        ``coco_pope_adversarial.json`` and resolves image paths.

        Args:
            pope_data_dir: Directory containing POPE question JSON files.
            coco_image_dir: Directory containing COCO val2014 images.
            coco_instances_path: Optional path to COCO instances JSON for
                RelTR vocabulary tagging.  When ``None``, ``reltr_tag`` is
                set to ``None`` on every sample.

        Yields:
            :class:`BenchmarkSample` for each valid POPE question.

        Raises:
            FileNotFoundError: If any of the three POPE JSON files is
                missing (propagated from the underlying loader).
        """
        # Build RelTR tag map if annotations are available.
        reltr_tag_map: dict[int, bool] | None = None
        if coco_instances_path is not None:
            reltr_tag_map = _build_reltr_tag_map(coco_instances_path)

        # Delegate to the existing loader — only COCO domain.
        loader = POPEDataLoader()
        image_dirs = {POPEDomain.COCO: coco_image_dir}

        # The underlying loader iterates all domains × settings.  We pass
        # only COCO in image_dirs, but the loader will still try to open
        # files for all 9 combinations.  Instead, we call the internal
        # _load_question_file directly for just the 3 COCO settings, or
        # we can catch the FileNotFoundError for non-COCO domains.
        #
        # Simpler approach: call load() but only pass COCO image_dirs.
        # The loader raises FileNotFoundError for missing files — since we
        # only have COCO files, we load them one setting at a time.
        for setting in POPESetting:
            split = _SETTING_TO_SPLIT[setting]
            filename = f"coco_pope_{setting.value}.json"
            filepath = Path(pope_data_dir) / filename

            if not filepath.is_file():
                raise FileNotFoundError(
                    f"POPE question file not found: {filepath}"
                )

            questions = POPEDataLoader._load_question_file(
                str(filepath),
                coco_image_dir,
                POPEDomain.COCO,
                setting,
            )

            for idx, q in enumerate(questions):
                # Compute reltr_tag from the image_id.
                reltr_tag: bool | None = None
                if reltr_tag_map is not None:
                    # POPEQuestion.image_id is the filename stem, e.g.
                    # "COCO_val2014_000000123456".  Extract the numeric id.
                    numeric_id = _extract_numeric_image_id(q.image_id)
                    if numeric_id is not None:
                        reltr_tag = reltr_tag_map.get(numeric_id, False)
                    else:
                        reltr_tag = False

                yield BenchmarkSample(
                    sample_id=f"pope_{split}_{q.image_id}_{idx}",
                    image_path=q.image_path,
                    question=q.question,
                    label=q.ground_truth,
                    split=split,
                    benchmark="pope",
                    reltr_tag=reltr_tag,
                    metadata={
                        "image_id": q.image_id,
                        "domain": POPEDomain.COCO.value,
                        "setting": setting.value,
                    },
                )


def _extract_numeric_image_id(image_id: str) -> int | None:
    """Extract the numeric COCO image ID from a filename stem.

    Handles both ``"COCO_val2014_000000123456"`` and plain numeric strings
    like ``"123456"``.

    Args:
        image_id: The image ID string (filename without extension).

    Returns:
        Integer image ID, or ``None`` if parsing fails.
    """
    # Try extracting from COCO naming pattern first.
    parts = image_id.split("_")
    for part in reversed(parts):
        try:
            return int(part)
        except ValueError:
            continue

    # Fallback: try the whole string.
    try:
        return int(image_id)
    except ValueError:
        logger.warning("Could not parse numeric image ID from: %s", image_id)
        return None
