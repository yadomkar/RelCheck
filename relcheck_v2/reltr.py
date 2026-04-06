"""
RelCheck v2 — RelTR Scene Graph Integration
=============================================
Vocabulary constants, inference wrapper, and COCO vocab filter helpers
for the RelTR scene graph generation model.

All inference code is gated behind ``ENABLE_RELTR``. When disabled,
``extract_scene_graph`` returns ``[]`` without importing torch or
loading any model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._logging import log
from .config import ENABLE_RELTR, RELTR_CONF_THRESHOLD

if TYPE_CHECKING:
    import torch
    from PIL import Image


# ════════════════════════════════════════════════════════════════════════════
# VOCABULARY CONSTANTS (Visual Genome, minus 'N/A' sentinel)
# ════════════════════════════════════════════════════════════════════════════

RELTR_OBJECT_CLASSES: list[str] = [
    "airplane", "animal", "arm", "bag", "banana", "basket", "beach",
    "bear", "bed", "bench", "bike", "bird", "board", "boat", "book",
    "boot", "bottle", "bowl", "box", "boy", "branch", "building", "bus",
    "cabinet", "cap", "car", "cat", "chair", "child", "clock", "coat",
    "counter", "cow", "cup", "curtain", "desk", "dog", "door", "drawer",
    "ear", "elephant", "engine", "eye", "face", "fence", "finger", "flag",
    "flower", "food", "fork", "fruit", "giraffe", "girl", "glass", "glove",
    "guy", "hair", "hand", "handle", "hat", "head", "helmet", "hill",
    "horse", "house", "jacket", "jean", "kid", "kite", "lady", "lamp",
    "laptop", "leaf", "leg", "letter", "light", "logo", "man", "men",
    "motorcycle", "mountain", "mouth", "neck", "nose", "number", "orange",
    "pant", "paper", "paw", "people", "person", "phone", "pillow", "pizza",
    "plane", "plant", "plate", "player", "pole", "post", "pot", "racket",
    "railing", "rock", "roof", "room", "screen", "seat", "sheep", "shelf",
    "shirt", "shoe", "short", "sidewalk", "sign", "sink", "skateboard",
    "ski", "skier", "sleeve", "snow", "sock", "stand", "street",
    "surfboard", "table", "tail", "tie", "tile", "tire", "toilet", "towel",
    "tower", "track", "train", "tree", "truck", "trunk", "umbrella",
    "vase", "vegetable", "vehicle", "wave", "wheel", "window",
    "windshield", "wing", "wire", "woman", "wood",
]
"""150 Visual Genome object classes recognized by RelTR."""

RELTR_PREDICATE_CLASSES: list[str] = [
    "above", "across", "against", "along", "and", "at", "attached to",
    "behind", "belonging to", "between", "carrying", "covered in",
    "covering", "eating", "flying in", "for", "from", "growing on",
    "hanging from", "has", "holding", "in", "in front of", "laying on",
    "looking at", "lying on", "made of", "mounted on", "near", "of", "on",
    "on back of", "over", "painted on", "parked on", "part of", "playing",
    "riding", "says", "sitting on", "standing on", "to", "under", "using",
    "walking in", "walking on", "watching", "wearing", "wears", "with",
]
"""50 Visual Genome predicate classes recognized by RelTR."""


# ════════════════════════════════════════════════════════════════════════════
# BBOX CONVERSION HELPERS
# ════════════════════════════════════════════════════════════════════════════


def _box_cxcywh_to_xyxy(boxes: "torch.Tensor") -> "torch.Tensor":
    """Convert (cx, cy, w, h) boxes to (x1, y1, x2, y2) format."""
    import torch as _torch  # noqa: F811 — deferred import

    x_c, y_c, w, h = boxes.unbind(1)
    return _torch.stack(
        [x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h],
        dim=1,
    )


def _rescale_bboxes(
    out_bbox: "torch.Tensor",
    img_size: tuple[int, int],
) -> "torch.Tensor":
    """Scale model-output bboxes to pixel coordinates.

    Args:
        out_bbox: Tensor of shape (N, 4) in normalized cxcywh format.
        img_size: (width, height) of the source image.

    Returns:
        Tensor of shape (N, 4) in xyxy pixel coordinates.
    """
    import torch as _torch  # noqa: F811

    img_w, img_h = img_size
    corners = _box_cxcywh_to_xyxy(out_bbox)
    scale = _torch.tensor(
        [img_w, img_h, img_w, img_h],
        dtype=_torch.float32,
        device=out_bbox.device,
    )
    return corners * scale


def _normalize_bbox(
    pixel_bbox: list[float],
    img_w: int,
    img_h: int,
) -> list[float]:
    """Normalize a pixel-space [x1, y1, x2, y2] bbox to [0, 1] range."""
    dims = [img_w, img_h, img_w, img_h]
    return [max(0.0, min(1.0, v / d)) for v, d in zip(pixel_bbox, dims)]


# ════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ════════════════════════════════════════════════════════════════════════════


def extract_scene_graph(image: "Image.Image") -> list[dict]:
    """Run RelTR inference on a PIL image and return scene graph triples.

    Returns ``[]`` immediately when ``ENABLE_RELTR`` is ``False`` — no
    model load, no torch import, no inference.

    Each returned dict contains:
        subject, predicate, object (str labels),
        subject_conf, predicate_conf, object_conf (float 0–1),
        bbox_sub, bbox_obj (list[float] normalized to [0, 1]).

    Triples are filtered by the triple-gate confidence threshold and
    sorted by ``predicate_conf`` descending.
    """
    if not ENABLE_RELTR:
        return []

    import torch as _torch  # noqa: F811
    from .models import get_reltr, DEVICE

    model, transform = get_reltr()
    if model is None or transform is None:
        log.warning("RelTR model unavailable — returning empty scene graph")
        return []

    try:
        img_tensor = transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)

        with _torch.no_grad():
            outputs = model(img_tensor)

        return _parse_outputs(outputs, image.size)

    except Exception as exc:
        log.error("RelTR inference failed: %s", exc)
        return []


def _parse_outputs(
    outputs: dict[str, Any],
    img_size: tuple[int, int],
) -> list[dict]:
    """Parse raw RelTR model outputs into filtered, sorted triple dicts."""
    import torch as _torch  # noqa: F811

    probas = outputs["rel_logits"].softmax(-1)[0, :, :-1]
    probas_sub = outputs["sub_logits"].softmax(-1)[0, :, :-1]
    probas_obj = outputs["obj_logits"].softmax(-1)[0, :, :-1]

    # Triple-gate: all three confidences must exceed threshold
    keep = _torch.nonzero(
        (probas.max(-1)[0] > RELTR_CONF_THRESHOLD)
        & (probas_sub.max(-1)[0] > RELTR_CONF_THRESHOLD)
        & (probas_obj.max(-1)[0] > RELTR_CONF_THRESHOLD)
    ).flatten()

    if len(keep) == 0:
        return []

    img_w, img_h = img_size
    bboxes_sub = _rescale_bboxes(outputs["sub_boxes"][0, keep], img_size)
    bboxes_obj = _rescale_bboxes(outputs["obj_boxes"][0, keep], img_size)

    triples: list[dict] = []
    for i, idx in enumerate(keep):
        triples.append({
            "subject": RELTR_OBJECT_CLASSES[probas_sub[idx].argmax()],
            "predicate": RELTR_PREDICATE_CLASSES[probas[idx].argmax()],
            "object": RELTR_OBJECT_CLASSES[probas_obj[idx].argmax()],
            "subject_conf": probas_sub[idx].max().item(),
            "predicate_conf": probas[idx].max().item(),
            "object_conf": probas_obj[idx].max().item(),
            "bbox_sub": _normalize_bbox(bboxes_sub[i].tolist(), img_w, img_h),
            "bbox_obj": _normalize_bbox(bboxes_obj[i].tolist(), img_w, img_h),
        })

    triples.sort(key=lambda t: -t["predicate_conf"])
    return triples


# ════════════════════════════════════════════════════════════════════════════
# COCO VOCABULARY FILTER HELPERS
# ════════════════════════════════════════════════════════════════════════════


def coco_categories_covered(coco_categories: list[str]) -> bool:
    """Check if every COCO category maps to at least one RelTR object class.

    Uses ``entity_matches`` for synonym-aware fuzzy matching.

    Args:
        coco_categories: List of COCO category name strings.

    Returns:
        True if all categories have a RelTR counterpart.
    """
    from .correction._utils import entity_matches

    return all(
        any(entity_matches(cat, rc) for rc in RELTR_OBJECT_CLASSES)
        for cat in coco_categories
    )


def coco_has_reltr_predicate_coverage(
    spatial_relations: list[tuple[str, str, str]],
) -> bool:
    """Check if at least one spatial relation maps to a RelTR predicate.

    Args:
        spatial_relations: List of ``(subject, predicate, object)`` tuples
            derived from COCO bbox geometry.

    Returns:
        True if any predicate has a RelTR counterpart.
    """
    from .correction._utils import entity_matches

    return any(
        any(entity_matches(pred, rp) for rp in RELTR_PREDICATE_CLASSES)
        for _, pred, _ in spatial_relations
    )
