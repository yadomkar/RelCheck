"""
RelCheck v2 — Visual Knowledge Base
=====================================
Three-layer KB construction per image:
  HARD  — GroundingDINO: objects + counts + bboxes (deterministic)
  GEOM  — Bbox geometry: pairwise spatial relationships (deterministic)
  SOFT  — VLM: actions, attributes, relationships (visual)
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Optional

from PIL import Image

from .api import vlm_call, encode_b64
from .config import BROAD_CATEGORIES
from .detection import detect_objects, dedup_detections
from .entity import extract_nouns
from .prompts import KB_DESCRIPTION_PROMPT
from .spatial import compute_spatial_facts


def build_visual_kb(
    image: Image.Image,
    caption: str,
    max_detections: int = 20,
) -> dict:
    """Build a 3-layer Visual Knowledge Base for one image.

    Args:
        image: PIL Image
        caption: Original caption text (for entity extraction)
        max_detections: Cap on number of detections to keep

    Returns:
        dict with keys: hard_facts, spatial_facts, visual_description, detections
    """
    # ── Layer 1: Object detection ──
    queries = list(set(extract_nouns(caption) + BROAD_CATEGORIES))
    raw_dets = detect_objects(image, queries)
    dets = dedup_detections(raw_dets)
    dets = sorted(dets, key=lambda x: -x[1])[:max_detections]

    # ── Layer 2: Deterministic facts ──
    counts = Counter(label for label, _, _ in dets)
    hard_facts = [f"{count}x {label}" for label, count in counts.most_common()]
    spatial_facts = compute_spatial_facts(dets)

    # ── Layer 3: VLM description ──
    det_str = "".join(f"- {label} ({count}x)\n" for label, count in counts.most_common())
    b64 = encode_b64(image)
    visual_description = vlm_call(
        [{"role": "user", "content": [
            {"type": "text", "text": KB_DESCRIPTION_PROMPT.replace("{detection_list}", det_str)},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]}],
        max_tokens=500,
    ) or ""

    return {
        "hard_facts": hard_facts,
        "spatial_facts": spatial_facts,
        "visual_description": visual_description,
        "detections": dets,
    }
