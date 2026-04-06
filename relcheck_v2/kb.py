"""
RelCheck v2 — Visual Knowledge Base
=====================================
Three-layer KB construction per image:
  HARD  — GroundingDINO: objects + counts + bboxes (deterministic)
  GEOM  — Bbox geometry: pairwise spatial relationships (deterministic)
  SOFT  — VLM: actions, attributes, relationships (visual)
"""

from __future__ import annotations

from collections import Counter

from PIL import Image

from ._logging import log
from .api import vlm_call, encode_b64
from .config import BROAD_CATEGORIES, ENABLE_RELTR
from .detection import detect_objects, dedup_detections
from .entity import extract_nouns
from .prompts import KB_DESCRIPTION_PROMPT
from .spatial import compute_spatial_facts
from .types import Detection, VisualKB


def build_visual_kb(
    image: Image.Image,
    caption: str,
    max_detections: int = 20,
) -> VisualKB:
    """Build a 3-layer Visual Knowledge Base for one image.

    Args:
        image: PIL Image
        caption: Original caption text (for entity extraction)
        max_detections: Cap on number of detections to keep

    Returns:
        VisualKB dataclass with hard_facts, spatial_facts, visual_description, detections
    """
    # ── Layer 1: Object detection ──
    queries = list(set(extract_nouns(caption) + BROAD_CATEGORIES))
    log.debug("Extracting detection queries from caption and broad categories: %d queries", len(queries))
    raw_dets = detect_objects(image, queries)
    dets = dedup_detections(raw_dets)
    dets = sorted(dets, key=lambda x: -x.score)[:max_detections]
    log.debug("Detected %d objects after deduplication and capping", len(dets))

    # ── Layer 2: Deterministic facts ──
    counts = Counter(det.label for det in dets)
    hard_facts = [f"{count}x {label}" for label, count in counts.most_common()]
    log.debug("Generated %d hard facts from object counts", len(hard_facts))

    # Convert Detection objects to tuple format for spatial_facts computation
    det_tuples = [d.as_tuple() for d in dets]
    spatial_facts = compute_spatial_facts(det_tuples)
    log.debug("Computed %d spatial facts", len(spatial_facts))

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
    log.debug("Generated visual description (%d chars)", len(visual_description))

    # ── Layer 4: RelTR scene graph (when enabled) ──
    scene_graph: list[dict] = []
    if ENABLE_RELTR:
        from .reltr import extract_scene_graph
        scene_graph = extract_scene_graph(image)
        log.info("RelTR produced %d scene graph triples", len(scene_graph))

    return VisualKB(
        hard_facts=hard_facts,
        spatial_facts=spatial_facts,
        visual_description=visual_description,
        detections=dets,
        scene_graph=scene_graph,
    )
