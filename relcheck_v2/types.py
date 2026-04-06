"""
RelCheck v2 — Type Definitions
================================
Dataclasses and type aliases used across the package.
Centralizes all structured data types to eliminate untyped dicts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ── Enums ────────────────────────────────────────────────────────────────


class RelationType(str, Enum):
    """Classification of a relational claim."""

    SPATIAL = "SPATIAL"
    ACTION = "ACTION"
    ATTRIBUTE = "ATTRIBUTE"


class Verdict(str, Enum):
    """Outcome of relation verification."""

    CORRECT = "CORRECT"
    INCORRECT = "INCORRECT"
    UNKNOWN = "UNKNOWN"


class Confidence(str, Enum):
    """Confidence level of a verification verdict."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class CorrectionMode(str, Enum):
    """Which correction strategy was applied."""

    ENRICH = "enrich"
    CORRECT_V2 = "correct_v2"


# ── Type aliases ─────────────────────────────────────────────────────────

BBox = list[float]
"""Bounding box as [x1, y1, x2, y2] in normalized (0–1) coordinates."""


# ── Dataclasses ──────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Detection:
    """Single object detection from GroundingDINO.

    Attributes:
        label: Cleaned object label (lowercase, no articles).
        score: Detection confidence score (0–1).
        bbox: Normalized bounding box [x1, y1, x2, y2].
    """

    label: str
    score: float
    bbox: BBox

    def as_tuple(self) -> tuple[str, float, list[float]]:
        """Convert to legacy (label, score, bbox) tuple format."""
        return (self.label, self.score, self.bbox)


@dataclass(frozen=True, slots=True)
class Triple:
    """A relational claim extracted from a caption.

    Attributes:
        subject: The entity performing or described first.
        relation: The relationship word or short phrase.
        object: The entity being related to.
        rel_type: Classification of the relation (SPATIAL, ACTION, ATTRIBUTE).
    """

    subject: str
    relation: str
    object: str
    rel_type: RelationType = RelationType.ACTION

    @classmethod
    def from_dict(cls, d: dict) -> Triple:
        """Construct from an untyped dict (e.g. LLM JSON output)."""
        raw_type = str(d.get("type", "ACTION")).upper().strip()
        try:
            rel_type = RelationType(raw_type)
        except ValueError:
            rel_type = RelationType.ACTION
        return cls(
            subject=d.get("subject", "").strip(),
            relation=d.get("relation", "").strip(),
            object=d.get("object", "").strip(),
            rel_type=rel_type,
        )

    @property
    def claim(self) -> str:
        """Human-readable claim string."""
        return f"{self.subject} {self.relation} {self.object}"


@dataclass(slots=True)
class VerificationResult:
    """Outcome of verifying a single triple.

    Attributes:
        triple: The triple that was verified.
        verdict: CORRECT, INCORRECT, or UNKNOWN.
        confidence: HIGH, MEDIUM, or LOW.
        reason: Human-readable explanation of the verdict.
        evidence_source: What produced the verdict (geometry, vqa, consensus, etc.).
    """

    triple: Triple
    verdict: Verdict
    confidence: Confidence
    reason: str = ""
    evidence_source: str = ""


@dataclass(slots=True)
class CorrectionError:
    """A confirmed hallucination error to be corrected.

    Attributes:
        triple: The hallucinated triple.
        reason: Why it was flagged.
        confidence: Confidence of the error detection.
        guidance: Instruction for the LLM corrector.
    """

    triple: Triple
    reason: str
    confidence: Confidence
    guidance: str = ""


@dataclass(slots=True)
class CorrectionResult:
    """Output of the correction pipeline for one image.

    Attributes:
        original: Original caption text.
        corrected: Corrected caption text (may be unchanged).
        errors: List of confirmed errors that were addressed.
        checks: All verification checks performed.
        mode: Which correction strategy was used.
        edit_rate: Normalized Levenshtein distance (0–1).
        n_triples: Number of triples extracted.
        n_addendum: Number of missing facts appended.
        status: 'modified' or 'unchanged'.
    """

    original: str
    corrected: str
    errors: list[CorrectionError] = field(default_factory=list)
    checks: list[VerificationResult] = field(default_factory=list)
    mode: CorrectionMode = CorrectionMode.CORRECT_V2
    edit_rate: float = 0.0
    n_triples: int = 0
    n_addendum: int = 0
    status: str = "unchanged"

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON checkpointing."""
        return {
            "caption": self.original,
            "corrected": self.corrected,
            "errors": [
                {
                    "claim": e.triple.claim,
                    "subject": e.triple.subject,
                    "relation": e.triple.relation,
                    "object": e.triple.object,
                    "reason": e.reason,
                    "confidence": e.confidence.value,
                    "type": e.triple.rel_type.value,
                }
                for e in self.errors
            ],
            "all_checks": [
                {
                    "claim": c.triple.claim,
                    "type": c.triple.rel_type.value,
                    "verdict": c.verdict.value,
                    "confidence": c.confidence.value,
                    "reason": c.reason,
                }
                for c in self.checks
            ],
            "edit_rate": self.edit_rate,
            "n_triples": self.n_triples,
            "n_addendum": self.n_addendum,
            "status": self.status,
            "mode": self.mode.value,
        }


@dataclass(slots=True)
class VisualKB:
    """Four-layer Visual Knowledge Base for one image.

    Attributes:
        hard_facts: Object counts from GroundingDINO (deterministic).
        spatial_facts: Pairwise spatial relationships from bbox geometry.
        visual_description: VLM-generated description (soft evidence).
        detections: Raw GroundingDINO detections.
        scene_graph: RelTR scene graph triples (gated by ENABLE_RELTR).
    """

    hard_facts: list[str] = field(default_factory=list)
    spatial_facts: list[str] = field(default_factory=list)
    visual_description: str = ""
    detections: list[Detection] = field(default_factory=list)
    scene_graph: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON checkpointing."""
        return {
            "hard_facts": self.hard_facts,
            "spatial_facts": self.spatial_facts,
            "visual_description": self.visual_description,
            "detections": [d.as_tuple() for d in self.detections],
            "scene_graph": self.scene_graph,
        }

    @classmethod
    def from_dict(cls, d: dict) -> VisualKB:
        """Reconstruct from a checkpoint dict."""
        dets = []
        for item in d.get("detections", []):
            if isinstance(item, Detection):
                dets.append(item)
            elif isinstance(item, (list, tuple)) and len(item) == 3:
                dets.append(Detection(label=item[0], score=item[1], bbox=item[2]))
        return cls(
            hard_facts=d.get("hard_facts", []),
            spatial_facts=d.get("spatial_facts", []),
            visual_description=d.get("visual_description", ""),
            detections=dets,
            scene_graph=d.get("scene_graph", []),
        )
