"""
RelCheck v2 — Object Detection
================================
Batched GroundingDINO detection with proper list-of-lists format,
torchvision IoU-based deduplication, and bbox utilities.
"""

from __future__ import annotations

from collections import Counter

import torch
from PIL import Image
from torchvision.ops import box_iou

from .config import GDINO_BOX_THRESHOLD, GDINO_TEXT_THRESHOLD
from .entity import clean_label, core_noun, candidate_synonyms
from .models import get_gdino, DEVICE

# Type alias for a single detection: (label, score, [x1, y1, x2, y2] normalized)
Detection = tuple[str, float, list[float]]


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

    Returns list of (label, score, [x1, y1, x2, y2]) with normalized coords.
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
            all_dets.append((
                clean_label(label),
                score.item(),
                [x1 / W, y1 / H, x2 / W, y2 / H],
            ))

    return all_dets


# ── Deduplication ────────────────────────────────────────────────────────

def dedup_detections(
    dets: list[Detection],
    iou_threshold: float = 0.5,
) -> list[Detection]:
    """Remove duplicate detections of the same class using torchvision IoU.

    Keeps the highest-confidence detection for each overlapping pair of
    same-label boxes.
    """
    if not dets:
        return []

    # Group by label
    by_label: dict[str, list[tuple[float, list[float]]]] = {}
    for label, score, bbox in sorted(dets, key=lambda x: -x[1]):
        by_label.setdefault(label, []).append((score, bbox))

    result: list[Detection] = []

    for label, entries in by_label.items():
        if len(entries) == 1:
            result.append((label, entries[0][0], entries[0][1]))
            continue

        # Build tensor of boxes for IoU computation
        boxes_tensor = torch.tensor([bbox for _, bbox in entries])
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

        for idx, (score, bbox) in enumerate(entries):
            if keep[idx]:
                result.append((label, score, bbox))

    return result


# ── Bbox utilities ───────────────────────────────────────────────────────

def find_best_bbox(
    entity: str,
    detections: list[Detection],
) -> list[float] | None:
    """Find the highest-confidence bbox whose label matches entity.

    Uses core noun extraction + synonym matching for robust lookup.
    E.g. entity='large dog' matches detection label='a dog'.
    """
    if not detections:
        return None

    target_core = core_noun(entity)
    target_syns = candidate_synonyms(target_core)

    best_score: float = -1.0
    best_box: list[float] | None = None

    for label, score, box in detections:
        label_core = core_noun(label)
        label_syns = candidate_synonyms(label_core)

        # Match if synonym sets overlap or substring containment
        if (target_syns & label_syns
                or target_core in label_core
                or label_core in target_core):
            if score > best_score:
                best_score = score
                best_box = box

    return best_box


def find_best_bbox_from_kb(
    entity_name: str,
    kb: dict,
) -> list[float] | None:
    """Find bbox from a Visual KB's detections list.

    KB detections are stored as list of (label, score, bbox) tuples,
    same format as detect_objects output.
    """
    detections = kb.get("detections", [])
    return find_best_bbox(entity_name, detections)


def crop_to_bboxes(
    image: Image.Image,
    box1: list[float],
    box2: list[float],
    padding: float = 0.15,
) -> Image.Image:
    """Crop image to the region containing both bounding boxes.

    Boxes are in normalized coords [x1, y1, x2, y2].
    Padding is added around the union of both boxes.
    Returns the original image if the crop would be too small.
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

    if right - left < 32 or bottom - top < 32:
        return image

    return image.crop((left, top, right, bottom))
