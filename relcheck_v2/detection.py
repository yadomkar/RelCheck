"""
RelCheck v2 — Object Detection
================================
Batched GroundingDINO detection with proper list-of-lists format,
torchvision IoU-based deduplication, and bbox utilities.

Uses Detection dataclass for type-safe handling with backward-compatibility
for tuple inputs via _ensure_detection() helper.
"""

from __future__ import annotations

import torch
from PIL import Image
from torchvision.ops import box_iou

from .config import GDINO_BOX_THRESHOLD, GDINO_TEXT_THRESHOLD
from .entity import clean_label, core_noun, candidate_synonyms
from .models import get_gdino, DEVICE
from .types import Detection, BBox
from ._logging import log


# ── Backward compatibility ───────────────────────────────────────────────

def _ensure_detection(item: Detection | tuple[str, float, list[float]]) -> Detection:
    """Convert tuple or Detection to Detection dataclass.

    Ensures functions remain backward-compatible with raw tuple input
    while maintaining type safety internally.

    Args:
        item: Either a Detection object or a (label, score, bbox) tuple.

    Returns:
        Detection dataclass instance.
    """
    if isinstance(item, Detection):
        return item
    if isinstance(item, (list, tuple)) and len(item) == 3:
        return Detection(label=item[0], score=item[1], bbox=item[2])
    raise TypeError(f"Expected Detection or 3-tuple, got {type(item)}")


# ── Core detection ───────────────────────────────────────────────────────

def detect_objects(
    image: Image.Image,
    queries: list[str],
    batch_size: int = 4,
) -> list[Detection]:
    """Run GroundingDINO on an image with batched queries.

    Uses the official HuggingFace API:
      - text as list-of-lists: [["a cat", "a dog"]]
      - Queries batched (default 4) to avoid attention dilution
      - Articles prefixed for better text encoder performance

    Args:
        image: PIL Image to detect objects in.
        queries: List of entity labels to search for.
        batch_size: Number of queries per forward pass (default 4).

    Returns:
        List of Detection objects with normalized bounding box coordinates.
    """
    model, processor = get_gdino()

    # Add articles for better text encoder performance
    article_queries: list[str] = []
    article_to_orig: dict[str, str] = {}
    for q in queries:
        q_clean = q.strip().lower()
        if q_clean.split()[0] in ("a", "an", "the"):
            aq = q_clean
        else:
            aq = f"a {q_clean}"
        article_queries.append(aq)
        article_to_orig[aq] = q_clean

    W, H = image.size
    all_dets: list[Detection] = []

    for i in range(0, len(article_queries), batch_size):
        batch = article_queries[i : i + batch_size]
        inputs = processor(
            images=image,
            text=[batch],  # Official list-of-lists format
            return_tensors="pt",
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=GDINO_BOX_THRESHOLD,
            text_threshold=GDINO_TEXT_THRESHOLD,
            target_sizes=[(image.height, image.width)],
        )[0]

        label_key = "text_labels" if "text_labels" in results else "labels"
        for score, label, box in zip(
            results["scores"], results[label_key], results["boxes"]
        ):
            x1, y1, x2, y2 = box.tolist()
            det = Detection(
                label=clean_label(label),
                score=score.item(),
                bbox=[x1 / W, y1 / H, x2 / W, y2 / H],
            )
            all_dets.append(det)

    log.debug("Detected %d objects from %d queries", len(all_dets), len(queries))
    return all_dets


# ── Deduplication ────────────────────────────────────────────────────────

def dedup_detections(
    dets: list[Detection | tuple[str, float, list[float]]],
    iou_threshold: float = 0.5,
) -> list[Detection]:
    """Remove duplicate detections of the same class using torchvision IoU.

    Keeps the highest-confidence detection for each overlapping pair of
    same-label boxes. Accepts both Detection objects and legacy tuples.

    Args:
        dets: List of Detection objects or (label, score, bbox) tuples.
        iou_threshold: IoU threshold for considering boxes as duplicates.

    Returns:
        List of deduplicated Detection objects sorted by label then score.
    """
    if not dets:
        return []

    # Convert all inputs to Detection objects for uniform handling
    dets_normalized = [_ensure_detection(d) for d in dets]

    # Sort by descending score for consistent priority
    dets_sorted = sorted(dets_normalized, key=lambda x: -x.score)

    # Group by label
    by_label: dict[str, list[Detection]] = {}
    for det in dets_sorted:
        by_label.setdefault(det.label, []).append(det)

    result: list[Detection] = []

    for label, entries in by_label.items():
        if len(entries) == 1:
            result.append(entries[0])
            continue

        # Build tensor of boxes for IoU computation
        boxes_tensor = torch.tensor([det.bbox for det in entries])
        ious = box_iou(boxes_tensor, boxes_tensor)

        keep: list[bool] = [True] * len(entries)
        for i in range(len(entries)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(entries)):
                if not keep[j]:
                    continue
                if ious[i, j] > iou_threshold:
                    # Keep higher-confidence one (entries sorted by -score)
                    keep[j] = False

        for idx, det in enumerate(entries):
            if keep[idx]:
                result.append(det)

    log.debug("Deduped %d → %d detections (IoU > %.2f)", len(dets_normalized), len(result), iou_threshold)
    return result


# ── Bbox utilities ───────────────────────────────────────────────────────

def find_best_bbox(
    entity: str,
    detections: list[Detection | tuple[str, float, list[float]]],
) -> BBox | None:
    """Find the highest-confidence bbox whose label matches entity.

    Uses core noun extraction + synonym matching for robust lookup.
    E.g. entity='large dog' matches detection label='a dog'.
    Accepts both Detection objects and legacy tuples for backward compatibility.

    Args:
        entity: Entity label to search for (may include adjectives/articles).
        detections: List of Detection objects or (label, score, bbox) tuples.

    Returns:
        Normalized bbox [x1, y1, x2, y2] of best matching detection, or None.
    """
    if not detections:
        return None

    # Convert all inputs to Detection objects
    dets_normalized = [_ensure_detection(d) for d in detections]

    target_core = core_noun(entity)
    target_syns = candidate_synonyms(target_core)

    best_score: float = -1.0
    best_box: BBox | None = None

    for det in dets_normalized:
        label_core = core_noun(det.label)
        label_syns = candidate_synonyms(label_core)

        # Match if synonym sets overlap or substring containment
        if (target_syns & label_syns
                or target_core in label_core
                or label_core in target_core):
            if det.score > best_score:
                best_score = det.score
                best_box = det.bbox

    return best_box


def find_best_bbox_from_kb(
    entity_name: str,
    kb: dict,
) -> BBox | None:
    """Find bbox from a Visual KB's detections list.

    KB detections may be stored as Detection objects, tuples, or a mix.
    The function handles any format via _ensure_detection().

    Args:
        entity_name: Entity to find in the KB detections.
        kb: Visual KB dict with "detections" key.

    Returns:
        Normalized bbox of best matching detection, or None if not found.
    """
    detections = kb.get("detections", [])
    return find_best_bbox(entity_name, detections)


def crop_to_bboxes(
    image: Image.Image,
    box1: BBox,
    box2: BBox,
    padding: float = 0.15,
) -> Image.Image:
    """Crop image to the region containing both bounding boxes.

    Useful for focused VQA on two entities (e.g., subject and object of
    a relational claim). Computes the union of both boxes with symmetric
    padding, then returns the original image if the crop would be too small.

    Args:
        image: PIL Image to crop.
        box1: First bbox as [x1, y1, x2, y2] in normalized (0–1) coords.
        box2: Second bbox as [x1, y1, x2, y2] in normalized (0–1) coords.
        padding: Fraction of the union to add as padding (default 0.15 = 15%).

    Returns:
        Cropped PIL Image, or original image if crop would be < 32×32 pixels.
    """
    W, H = image.size
    xs = [box1[0], box1[2], box2[0], box2[2]]
    ys = [box1[1], box1[3], box2[1], box2[3]]

    x1 = max(0.0, min(xs) - padding)
    y1 = max(0.0, min(ys) - padding)
    x2 = min(1.0, max(xs) + padding)
    y2 = min(1.0, max(ys) + padding)

    left, top = int(x1 * W), int(y1 * H)
    right, bottom = int(x2 * W), int(y2 * H)

    crop_width, crop_height = right - left, bottom - top
    if crop_width < 32 or crop_height < 32:
        log.debug("Crop too small (%d×%d), returning original image", crop_width, crop_height)
        return image

    log.debug("Cropped to %.1f%% of image area", 100 * (crop_width * crop_height) / (W * H))
    return image.crop((left, top, right, bottom))
