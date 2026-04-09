"""
Knowledge Base Builder — combines three layers:
  CLAIM  — Woodpecker claims from claim_generation pipeline
  GEOM   — Bbox geometry: pairwise spatial relationships (deterministic)
  SCENE  — RelTR: scene graph triples (visual, gated by ENABLE_RELTR)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from relcheck_v3.claim_generation.models import ObjectAnswer, VisualKnowledgeBase
from relcheck_v3.geometry import compute_spatial_facts

if TYPE_CHECKING:
    from PIL import Image


def _cfg_enable_reltr() -> bool:
    """Read ENABLE_RELTR at call time (not import time)."""
    from relcheck_v3.reltr import config
    return config.ENABLE_RELTR

log = logging.getLogger(__name__)


@dataclass
class KnowledgeBase:
    """Three-layer Knowledge Base for one image.

    Attributes:
        claims: Woodpecker VKB (count + specific + overall claims).
        spatial_facts: Pairwise spatial relationships from bbox geometry.
        scene_graph: RelTR scene graph triples (empty when ENABLE_RELTR=False).
    """

    claims: VisualKnowledgeBase
    spatial_facts: list[str] = field(default_factory=list)
    scene_graph: list[dict] = field(default_factory=list)

    def format(self) -> str:
        """Format the full KB as a labeled text string."""
        sections: list[str] = []

        # CLAIM section
        sections.append("=== CLAIM ===")
        sections.append(self.claims.format())

        # GEOM section
        sections.append("\n=== GEOM ===")
        if self.spatial_facts:
            for i, fact in enumerate(self.spatial_facts, 1):
                sections.append(f"{i}. {fact}")
        else:
            sections.append("(no spatial facts)")

        # SCENE section
        sections.append("\n=== SCENE ===")
        if self.scene_graph:
            for i, triple in enumerate(self.scene_graph, 1):
                s = triple["subject"]
                p = triple["predicate"]
                o = triple["object"]
                conf = triple.get("predicate_conf", 0.0)
                sections.append(f"{i}. {s} {p} {o} (conf={conf:.2f})")
        else:
            sections.append("(RelTR disabled or no triples)")

        return "\n".join(sections)


def build_kb(
    vkb: VisualKnowledgeBase,
    object_answers: dict[str, ObjectAnswer],
    image: "Image.Image | None" = None,
) -> KnowledgeBase:
    """Build a three-layer Knowledge Base from claim generation output.

    Args:
        vkb: The VisualKnowledgeBase from claim generation (CLAIM layer).
        object_answers: Dict of entity → ObjectAnswer with bboxes from
            Stage 3 of claim generation. Used to compute GEOM layer.
        image: PIL Image, required only when ENABLE_RELTR=True for the
            SCENE layer. Can be None if RelTR is disabled.

    Returns:
        KnowledgeBase with all three layers populated.
    """
    # ── CLAIM layer (already built by claim_generation) ──
    claims = vkb

    # ── GEOM layer: convert ObjectAnswers to detection tuples ──
    det_tuples: list[tuple[str, float, list[float]]] = []
    for name, answer in object_answers.items():
        for bbox in answer.bboxes:
            # Use confidence=1.0 since these are already validated detections
            det_tuples.append((name, 1.0, bbox))

    spatial_facts = compute_spatial_facts(det_tuples)
    log.info("GEOM layer: %d spatial facts from %d detections", len(spatial_facts), len(det_tuples))

    # ── SCENE layer: RelTR scene graph (gated, same pattern as v2) ──
    scene_graph: list[dict] = []
    if image is not None and _cfg_enable_reltr():
        from relcheck_v3.reltr.reltr import extract_scene_graph
        scene_graph = extract_scene_graph(image)
        log.info("SCENE layer: %d RelTR triples", len(scene_graph))

    return KnowledgeBase(
        claims=claims,
        spatial_facts=spatial_facts,
        scene_graph=scene_graph,
    )
