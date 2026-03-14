"""
RelCheck — Stage 2: Relation Verifier
=======================================
For each (subject, relation, object) triple, determines whether the relation
is actually supported by the image.

Uses two verification strategies depending on relation type:
  - ACTION / ATTRIBUTE  → BLIP-2 VQA (Yes/No binary question)
  - SPATIAL             → OWL-ViT bounding box detection + geometry check

Dependencies (install in Colab):
    !pip install transformers pillow torch torchvision

Author: Siddhi Patil | CS298 Spring 2026
"""

import torch
from PIL import Image
from typing import Optional
from dataclasses import dataclass

from triple_extractor import Triple, SPATIAL_KEYWORDS


# ---------------------------------------------------------------------------
# VQA Verifier  (BLIP-2)
# ---------------------------------------------------------------------------

class VQAVerifier:
    """
    Uses BLIP-2 in visual question answering mode to verify action and
    attribute triples via a binary Yes/No probe.

    Key idea: instead of letting the model hallucinate a caption, we ask it
    a focused yes/no question — "Is the woman holding the dog?" — which is
    much harder to get wrong because the model just needs to match a pattern,
    not generate a fluent sentence.
    """

    def __init__(self, model_name: str = "Salesforce/blip2-flan-t5-xl"):
        from transformers import Blip2Processor, Blip2ForConditionalGeneration

        print(f"[VQAVerifier] Loading {model_name} ...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[VQAVerifier] Using device: {self.device}")

        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()
        print("[VQAVerifier] Ready.")

    def _build_question(self, triple: Triple) -> str:
        """
        Turn a triple into a natural yes/no question.

        Examples:
            (woman, hold, dog)   → "Is the woman holding the dog?"
            (car, be, red)       → "Is the car red?"
            (man, sit, bench)    → "Is the man sitting on the bench?"
        """
        s, r, o = triple.subject, triple.relation, triple.obj

        if triple.relation_type == "ATTRIBUTE":
            return f"Is the {s} {o}?"

        # For action relations, add -ing suffix to verb lemma if needed.
        # spaCy gives us the lemma (base form), so we conjugate to present participle.
        verb_ing = _to_present_participle(r)
        return f"Is the {s} {verb_ing} the {o}?"

    def verify(self, image: Image.Image, triple: Triple) -> tuple[bool, float]:
        """
        Ask BLIP-2 the yes/no question for this triple against the image.

        Returns:
            (is_supported, confidence)
            is_supported = True  → the relation IS in the image (not hallucinated)
            is_supported = False → the relation is NOT in the image (hallucinated)
        """
        question = self._build_question(triple)

        inputs = self.processor(
            images=image,
            text=question,
            return_tensors="pt"
        ).to(self.device, torch.float16 if self.device == "cuda" else torch.float32)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=10,
            )
        answer = self.processor.decode(generated_ids[0], skip_special_tokens=True).strip().lower()

        # Parse the answer — BLIP-2 typically says "yes" or "no"
        is_yes = answer.startswith("yes")
        confidence = 0.9 if answer in ("yes", "no") else 0.6   # lower conf if ambiguous

        return is_yes, confidence

    def verify_batch(
        self, image: Image.Image, triples: list[Triple]
    ) -> list[tuple[bool, float]]:
        """Verify a list of triples against one image."""
        return [self.verify(image, t) for t in triples]


# ---------------------------------------------------------------------------
# Spatial Verifier  (OWL-ViT + geometry)
# ---------------------------------------------------------------------------

@dataclass
class BoundingBox:
    """A detected bounding box: [x_min, y_min, x_max, y_max] in [0, 1] normalized coords."""
    label: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    score: float

    @property
    def cx(self) -> float:
        """Horizontal center."""
        return (self.x_min + self.x_max) / 2

    @property
    def cy(self) -> float:
        """Vertical center (smaller = higher up in image)."""
        return (self.y_min + self.y_max) / 2

    @property
    def area(self) -> float:
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)

    def iou(self, other: "BoundingBox") -> float:
        """Intersection over union with another box."""
        inter_x1 = max(self.x_min, other.x_min)
        inter_y1 = max(self.y_min, other.y_min)
        inter_x2 = min(self.x_max, other.x_max)
        inter_y2 = min(self.y_max, other.y_max)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        union_area = self.area + other.area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def vertical_overlap(self, other: "BoundingBox") -> float:
        """How much do these boxes overlap vertically?"""
        overlap_top    = max(self.y_min, other.y_min)
        overlap_bottom = min(self.y_max, other.y_max)
        if overlap_bottom <= overlap_top:
            return 0.0
        return (overlap_bottom - overlap_top) / min(
            self.y_max - self.y_min, other.y_max - other.y_min
        )


class SpatialVerifier:
    """
    Uses OWL-ViT (open-vocabulary object detector) to find bounding boxes for
    the subject and object in a triple, then checks the spatial relation using
    simple geometry.

    Why this is better than VQA for spatial relations:
        VQA models are notoriously bad at spatial reasoning (Kamath et al., 2023,
        show 20-30% worse accuracy vs. object recognition). Bounding-box geometry
        is deterministic and doesn't rely on language priors at all.
    """

    # Confidence threshold: OWL-ViT detections below this score are ignored
    DETECTION_THRESHOLD = 0.15

    def __init__(self, model_name: str = "google/owlvit-base-patch32"):
        from transformers import OwlViTProcessor, OwlViTForObjectDetection

        print(f"[SpatialVerifier] Loading {model_name} ...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.model = OwlViTForObjectDetection.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print("[SpatialVerifier] Ready.")

    def detect(self, image: Image.Image, queries: list[str]) -> dict[str, list[BoundingBox]]:
        """
        Detect objects in image matching the text queries.

        Args:
            image:   PIL Image
            queries: list of text labels to look for, e.g. ["woman", "dog"]

        Returns:
            dict mapping each query string to a list of BoundingBox detections
            (sorted by score, highest first).
        """
        texts = [queries]   # OWL-ViT expects batch of lists

        inputs = self.processor(text=texts, images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process to get boxes in [0,1] normalized coords
        target_sizes = torch.tensor([image.size[::-1]])  # (H, W)
        # Newer transformers moved post_process off the processor onto image_processor
        _post_process = (
            self.processor.post_process_object_detection
            if hasattr(self.processor, "post_process_object_detection")
            else self.processor.image_processor.post_process_object_detection
        )
        results = _post_process(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=self.DETECTION_THRESHOLD,
        )[0]

        detections: dict[str, list[BoundingBox]] = {q: [] for q in queries}

        for score, label_idx, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            label = queries[label_idx.item()]
            # Normalize box to [0, 1]
            h, w = image.size[1], image.size[0]
            x_min, y_min, x_max, y_max = box.tolist()
            detections[label].append(BoundingBox(
                label=label,
                x_min=x_min / w,
                y_min=y_min / h,
                x_max=x_max / w,
                y_max=y_max / h,
                score=score.item(),
            ))

        # Sort by confidence
        for q in queries:
            detections[q].sort(key=lambda b: b.score, reverse=True)

        return detections

    def check_spatial_relation(
        self, subj_box: BoundingBox, rel: str, obj_box: BoundingBox
    ) -> bool:
        """
        Check if a spatial relation holds between two bounding boxes.

        Returns True if the relation is geometrically supported.
        """
        rel = rel.lower().strip()

        # --- Directional relations (centroid-based) ---
        if rel in ("above", "over"):
            return subj_box.cy < obj_box.cy    # smaller y = higher in image

        if rel in ("below", "under", "underneath"):
            return subj_box.cy > obj_box.cy

        if rel in ("left", "left of", "to the left of"):
            return subj_box.cx < obj_box.cx

        if rel in ("right", "right of", "to the right of"):
            return subj_box.cx > obj_box.cx

        # --- Contact/containment relations (overlap-based) ---
        if rel in ("on", "on top of"):
            # subject box bottom overlaps with object box top
            bottom_proximity = abs(subj_box.y_max - obj_box.y_min)
            horiz_overlap = (
                min(subj_box.x_max, obj_box.x_max) - max(subj_box.x_min, obj_box.x_min)
            ) > 0
            return bottom_proximity < 0.1 and horiz_overlap

        if rel in ("in", "inside", "within"):
            # subject box mostly inside object box
            return (
                subj_box.x_min >= obj_box.x_min - 0.05
                and subj_box.y_min >= obj_box.y_min - 0.05
                and subj_box.x_max <= obj_box.x_max + 0.05
                and subj_box.y_max <= obj_box.y_max + 0.05
            )

        if rel in ("beside", "next to", "adjacent to"):
            # boxes are near each other horizontally with little vertical distance
            horiz_gap = max(subj_box.x_min - obj_box.x_max, obj_box.x_min - subj_box.x_max, 0)
            vert_overlap = subj_box.vertical_overlap(obj_box)
            return horiz_gap < 0.15 and vert_overlap > 0.3

        if rel in ("near", "close to", "by"):
            # centroid distance below a threshold
            dist = ((subj_box.cx - obj_box.cx) ** 2 + (subj_box.cy - obj_box.cy) ** 2) ** 0.5
            return dist < 0.35

        if rel in ("behind"):
            # behind = subject appears smaller (further) and overlaps with object
            iou = subj_box.iou(obj_box)
            size_ratio = subj_box.area / (obj_box.area + 1e-6)
            return iou > 0.1 and size_ratio < 0.8

        if rel in ("in front of"):
            iou = subj_box.iou(obj_box)
            size_ratio = subj_box.area / (obj_box.area + 1e-6)
            return iou > 0.1 and size_ratio > 0.8

        # Unknown spatial relation — default to True (don't flag as hallucination)
        return True

    def verify(self, image: Image.Image, triple: Triple) -> tuple[bool, float]:
        """
        Verify a spatial triple using bounding box detection + geometry.

        Returns:
            (is_supported, confidence)
        """
        queries = [triple.subject, triple.obj]
        detections = self.detect(image, queries)

        subj_boxes = detections[triple.subject]
        obj_boxes  = detections[triple.obj]

        # If either entity is not detected, we can't verify — assume not hallucinated
        # (conservative: don't flag what we can't confirm)
        if not subj_boxes or not obj_boxes:
            return True, 0.4   # low confidence = uncertain

        subj_box = subj_boxes[0]   # top-scoring detection for subject
        obj_box  = obj_boxes[0]    # top-scoring detection for object

        is_supported = self.check_spatial_relation(subj_box, triple.relation, obj_box)
        confidence = min(subj_box.score, obj_box.score)   # limited by weakest detection

        return is_supported, confidence


# ---------------------------------------------------------------------------
# Combined Verifier (routes to the right strategy)
# ---------------------------------------------------------------------------

class RelationVerifier:
    """
    Top-level verifier that routes each triple to the appropriate strategy:
      - SPATIAL   → SpatialVerifier  (OWL-ViT + geometry)
      - ACTION    → VQAVerifier      (BLIP-2 Yes/No)
      - ATTRIBUTE → VQAVerifier      (BLIP-2 Yes/No)
      - OTHER     → VQAVerifier      (fallback)

    Populates triple.hallucinated = True/False in place.
    """

    # Only flag a triple as hallucinated if confidence is above this threshold
    CONFIDENCE_THRESHOLD = 0.45

    def __init__(self):
        self.vqa     = VQAVerifier()
        self.spatial = SpatialVerifier()

    def verify_triple(self, image: Image.Image, triple: Triple) -> Triple:
        """
        Verify one triple against the image. Sets triple.hallucinated in place.
        Returns the same Triple object (modified).
        """
        if triple.relation_type == "SPATIAL":
            is_supported, conf = self.spatial.verify(image, triple)
        else:
            is_supported, conf = self.vqa.verify(image, triple)

        # Only mark as hallucinated if we're confident enough
        if conf >= self.CONFIDENCE_THRESHOLD:
            triple.hallucinated = not is_supported
        else:
            triple.hallucinated = None   # uncertain — don't touch

        return triple

    def verify_all(self, image: Image.Image, triples: list[Triple]) -> list[Triple]:
        """Verify a list of triples. Returns the same list with hallucinated set."""
        for triple in triples:
            self.verify_triple(image, triple)
        return triples


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

_IRREGULAR_PARTICIPLES = {
    "sit": "sitting", "run": "running", "swim": "swimming",
    "lie": "lying",   "ride": "riding",  "hold": "holding",
    "eat": "eating",  "hit": "hitting",  "get": "getting",
    "put": "putting", "cut": "cutting",  "stand": "standing",
    "wear": "wearing","drive": "driving","make": "making",
    "take": "taking", "give": "giving",  "come": "coming",
    "look": "looking","walk": "walking", "play": "playing",
    "carry": "carrying", "talk": "talking",
}


def _to_present_participle(lemma: str) -> str:
    """Convert a verb lemma to its -ing form for natural question generation."""
    if lemma in _IRREGULAR_PARTICIPLES:
        return _IRREGULAR_PARTICIPLES[lemma]
    if lemma.endswith("e") and not lemma.endswith("ee"):
        return lemma[:-1] + "ing"
    return lemma + "ing"
