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
# Query cleaning utilities
# ---------------------------------------------------------------------------

# Aggressive cleaning for OWL-ViT: strip all determiners + adjectives that
# confuse the open-vocabulary detector.
_OWLVIT_STOP_WORDS = {"a", "an", "the", "two", "three", "four", "some", "many", "several", "old"}

def _clean_query(text: str) -> str:
    """
    Strip leading articles/determiners so OWL-ViT gets clean noun phrases.
    "a cat" → "cat", "an old man" → "man", "two toy cars" → "toy cars"
    OWL-ViT text-matching works significantly better without determiners.
    Only used for SpatialVerifier.detect().
    """
    words = text.strip().split()
    while words and words[0].lower() in _OWLVIT_STOP_WORDS:
        words = words[1:]
    return " ".join(words) if words else text.strip()


# Light cleaning for VQA questions: strip ONLY articles (a/an/the),
# keep adjectives like "old", "red", "large" for question specificity.
_VQA_STOP_WORDS = {"a", "an", "the"}

def _clean_query_for_vqa(text: str) -> str:
    """
    Light cleaning for VQA question building — strip only articles.
    Keeps adjectives so questions remain specific:
    "an old man" → "old man" (NOT "man")
    "a red car"  → "red car"  (NOT "car")
    This matters when multiple similar entities exist in the image.
    """
    words = text.strip().split()
    while words and words[0].lower() in _VQA_STOP_WORDS:
        words = words[1:]
    return " ".join(words) if words else text.strip()


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
        Uses light cleaning (articles only) to keep adjectives for specificity.

        Examples:
            (old woman, hold, dog)   → "Is the old woman holding the dog?"
            (car, be, red)           → "Is the car red?"
            (cat, under, couch)      → "Is the cat under the couch?"
        """
        s = _clean_query_for_vqa(triple.subject)
        o = _clean_query_for_vqa(triple.obj)
        r = triple.relation

        if triple.relation_type == "ATTRIBUTE":
            return f"Is the {s} {o}?"

        # Spatial: preposition goes directly between subject and object
        if triple.relation_type == "SPATIAL":
            return f"Is the {s} {r} the {o}?"

        # ACTION / OTHER: conjugate verb to present participle
        verb_ing = _to_present_participle(r)
        return f"Is the {s} {verb_ing} the {o}?"

    def verify(self, image: Image.Image, triple: Triple) -> tuple[bool, float]:
        """
        Ask BLIP-2 the yes/no question for this triple against the image.

        Uses token-level probabilities rather than greedy decoding to avoid
        BLIP-2's well-known yes-bias (it almost always decodes "yes" as text).
        We instead look at the softmax probability of "yes" vs "no" at the first
        token position — this gives a real confidence score.

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
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1,
                output_scores=True,
                return_dict_in_generate=True,
            )

        # scores[0] is the logit distribution over the full vocabulary at step 0
        first_token_logits = outputs.scores[0][0].float()   # shape: (vocab_size,)
        probs = torch.softmax(first_token_logits, dim=-1)

        # Get the token IDs for "yes" and "no" from the tokenizer
        yes_id = self.processor.tokenizer.encode("yes", add_special_tokens=False)[0]
        no_id  = self.processor.tokenizer.encode("no",  add_special_tokens=False)[0]

        yes_prob = probs[yes_id].item()
        no_prob  = probs[no_id].item()
        total    = yes_prob + no_prob + 1e-9

        # yes_ratio: fraction of yes vs (yes+no) mass
        yes_ratio = yes_prob / total
        is_yes    = yes_ratio >= 0.5

        # Confidence = how strongly it leans one way (0.5 = uncertain, 1.0 = certain)
        # We require yes_ratio > 0.65 to call "supported" — this combats yes-bias.
        confidence = max(yes_ratio, 1.0 - yes_ratio)

        # Override: if yes_ratio is in the ambiguous zone (0.35–0.65), treat as uncertain
        # by returning low confidence so the caller leaves hallucinated=None.
        # Tightened from [0.50, 0.65): only flag hallucination below 0.35 to reduce
        # false positives (over-correction from borderline cases).
        if 0.35 <= yes_ratio < 0.65:
            confidence = 0.3    # forces hallucinated=None (uncertain)

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
    Uses an open-vocabulary object detector to find bounding boxes for
    the subject and object in a triple, then checks the spatial relation using
    simple geometry.

    Supports both OWL-ViT v1 and OWLv2. OWLv2 (google/owlv2-base-patch16-ensemble)
    is recommended — it has significantly better detection accuracy on complex scenes
    (see VisMin, NeurIPS 2024, which uses Grounding DINO for similar reasons).

    Why this is better than VQA for spatial relations:
        VQA models are notoriously bad at spatial reasoning (Kamath et al., 2023,
        show 20-30% worse accuracy vs. object recognition). Bounding-box geometry
        is deterministic and doesn't rely on language priors at all.
    """

    # Confidence threshold: detections below this score are ignored
    DETECTION_THRESHOLD = 0.15

    def __init__(self, model_name: str = "google/owlv2-base-patch16-ensemble"):
        """
        Args:
            model_name: HuggingFace model ID. Options:
                - "google/owlv2-base-patch16-ensemble" (recommended, better accuracy)
                - "google/owlvit-base-patch32" (legacy, faster but less accurate)
        """
        print(f"[SpatialVerifier] Loading {model_name} ...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model_name = model_name

        if "owlv2" in model_name:
            from transformers import Owlv2Processor, Owlv2ForObjectDetection
            self.processor = Owlv2Processor.from_pretrained(model_name)
            self.model = Owlv2ForObjectDetection.from_pretrained(model_name).to(self.device)
        else:
            from transformers import OwlViTProcessor, OwlViTForObjectDetection
            self.processor = OwlViTProcessor.from_pretrained(model_name)
            self.model = OwlViTForObjectDetection.from_pretrained(model_name).to(self.device)

        self.model.eval()
        print("[SpatialVerifier] Ready.")

    def detect(self, image: Image.Image, queries: list[str]) -> dict[str, list[BoundingBox]]:
        """
        Detect objects in image matching the text queries.
        Works with both OWL-ViT v1 and OWLv2.

        Args:
            image:   PIL Image
            queries: list of text labels to look for, e.g. ["woman", "dog"]

        Returns:
            dict mapping each query string to a list of BoundingBox detections
            (sorted by score, highest first).
        """
        texts = [queries]   # OWL-ViT/v2 expects [[query1, query2]] for single image

        inputs = self.processor(text=texts, images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process to get boxes in pixel coords
        target_sizes = torch.tensor([image.size[::-1]])  # (H, W)

        # Try multiple API locations — transformers versions differ
        _post_process = None
        for attr_name in [
            "post_process_object_detection",           # processor-level
            "image_processor.post_process_object_detection",  # newer API
        ]:
            obj = self.processor
            for part in attr_name.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if obj is not None:
                _post_process = obj
                break

        if _post_process is None:
            print("[SpatialVerifier] WARNING: Cannot find post_process method — returning empty.")
            return {q: [] for q in queries}

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

        # Unknown spatial relation — return low confidence to trigger VQA fallback
        # instead of silently approving. This covers "across", "along", "through",
        # "around", etc. that have no geometric rule.
        return None   # sentinel: caller checks for None → VQA fallback

    def verify(self, image: Image.Image, triple: Triple) -> tuple[bool, float]:
        """
        Verify a spatial triple using bounding box detection + geometry.

        Returns:
            (is_supported, confidence)
        """
        # Clean queries: strip articles so OWL-ViT matches better
        # "a cat" → "cat", "two toy cars" → "toy cars", "an old man" → "man"
        subj_q = _clean_query(triple.subject)
        obj_q  = _clean_query(triple.obj)
        queries = [subj_q, obj_q]
        detections = self.detect(image, queries)

        subj_boxes = detections[subj_q]
        obj_boxes  = detections[obj_q]

        # If either entity is not detected, we can't verify — return low confidence
        # so the caller can fall back to VQA
        if not subj_boxes or not obj_boxes:
            return True, 0.3   # low confidence = triggers VQA fallback

        subj_box = subj_boxes[0]   # top-scoring detection for subject
        obj_box  = obj_boxes[0]    # top-scoring detection for object

        result = self.check_spatial_relation(subj_box, triple.relation, obj_box)

        # None = unknown spatial relation with no geometric rule → VQA fallback
        if result is None:
            return True, 0.3

        confidence = min(subj_box.score, obj_box.score)   # limited by weakest detection
        return result, confidence


# ---------------------------------------------------------------------------
# LLaVA Verifier  (independent cross-model verification)
# ---------------------------------------------------------------------------

class LLaVAVerifier:
    """
    Uses LLaVA-1.5-7B to verify action/attribute triples via dual-signal
    verification — fully independent from BLIP-2.

    Dual verification approach:
      1. SIGNAL 1 (VQA): Binary yes/no question → yes_ratio
         Fast and usually correct for clearly-supported relations.
      2. SIGNAL 2 (Describe-and-Compare): LLaVA describes the relationship,
         Llama-3.3-70B judges semantic consistency with the caption's claim.
         Provides evidence for correction.

    Decision logic (conservative — reduces false positives):
      - VQA high confidence YES (≥0.65) → supported (skip NLI)
      - VQA low confidence NO (<0.40) → run NLI to CONFIRM hallucination
        - NLI says inconsistent → hallucinated (with evidence)
        - NLI says consistent  → uncertain (NLI overrides VQA)
      - VQA uncertain [0.40, 0.65) → run NLI as tiebreaker

    Key insight: Binary VQA has yes-bias, so when it says "no" something is
    likely wrong. But we still double-check with NLI to avoid false positives
    from VQA noise. Evidence comes free from the describe step.

    Inspired by TIFA (ICCV 2023), VQAScore (ICLR 2025), DSG (ICLR 2024).
    """

    # VQA template (binary yes/no)
    VQA_TEMPLATE = "USER: <image>\n{question} Answer with yes or no only. ASSISTANT:"

    # Describe template (open-ended — forces complete sentence)
    DESCRIBE_TEMPLATE = (
        "USER: <image>\n{question}\n"
        "Describe what you see in one complete sentence starting with 'The'. ASSISTANT:"
    )

    # Thresholds for VQA signal
    VQA_SUPPORTED_THRESHOLD = 0.65     # above → supported without NLI
    VQA_HALLUCINATED_THRESHOLD = 0.40  # below → likely hallucinated, confirm with NLI

    def __init__(self, model=None, processor=None, together_client=None,
                 num_paraphrases: int = 3):
        """
        Args:
            model:            Pre-loaded LlavaForConditionalGeneration.
            processor:        Pre-loaded LlavaProcessor.
            together_client:  together.Together client for Llama NLI calls.
            num_paraphrases:  Number of VQA paraphrases for multi-question voting.
        """
        self.num_paraphrases = num_paraphrases
        self.together_client = together_client
        self.NLI_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

        if model is not None and processor is not None:
            self.model     = model
            self.processor = processor
            self.device    = next(model.parameters()).device
            print(f"[LLaVAVerifier] Using dual-signal verification (VQA + Describe-and-Compare).")
        else:
            self._load_model()

    def _load_model(self, model_name: str = "llava-hf/llava-1.5-7b-hf"):
        from transformers import LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
        print(f"[LLaVAVerifier] Loading {model_name} in 8-bit ...")
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        bnb            = BitsAndBytesConfig(load_in_8bit=True)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model     = LlavaForConditionalGeneration.from_pretrained(
            model_name, quantization_config=bnb, device_map="auto"
        )
        self.model.eval()
        print("[LLaVAVerifier] Ready.")

    def _build_question(self, triple: Triple) -> str:
        """Build primary yes/no question. Uses light cleaning to keep adjectives."""
        s = _clean_query_for_vqa(triple.subject)
        o = _clean_query_for_vqa(triple.obj)
        r = triple.relation
        if triple.relation_type == "ATTRIBUTE":
            return f"Is the {s} {o}?"
        if triple.relation_type == "SPATIAL":
            return f"Is the {s} {r} the {o}?"
        verb_ing = _to_present_participle(r)
        return f"Is the {s} {verb_ing} the {o}?"

    def _build_paraphrased_questions(self, triple: Triple) -> list[str]:
        """
        Generate 2-3 paraphrased questions for multi-question voting.
        Inspired by VisMin (NeurIPS 2024) multi-view evaluation approach.

        Returns a list of question strings (always includes the primary question).
        """
        s = _clean_query_for_vqa(triple.subject)
        o = _clean_query_for_vqa(triple.obj)
        r = triple.relation
        questions = [self._build_question(triple)]   # always include primary

        if self.num_paraphrases <= 1:
            return questions

        if triple.relation_type == "ACTION":
            verb_ing = _to_present_participle(r)
            # Passive voice variant
            questions.append(f"Is the {o} being {r}ed by the {s}?")
            # Descriptive variant
            questions.append(f"Can you see the {s} {verb_ing} the {o} in this image?")

        elif triple.relation_type == "SPATIAL":
            # Reversed perspective
            _reverse_map = {
                "on": "under", "under": "on", "above": "below", "below": "above",
                "left of": "right of", "right of": "left of",
                "in front of": "behind", "behind": "in front of",
            }
            rev = _reverse_map.get(r.lower())
            if rev:
                questions.append(f"Is the {o} {rev} the {s}?")
            # Descriptive variant
            questions.append(f"In this image, is the {s} positioned {r} the {o}?")

        elif triple.relation_type == "ATTRIBUTE":
            # Rephrase with "does...look"
            questions.append(f"Does the {s} look {o}?")
            questions.append(f"Would you describe the {s} as {o}?")

        else:
            # Generic rephrase for OTHER
            questions.append(f"Does this image show the {s} {r} the {o}?")

        return questions[:self.num_paraphrases]

    def _get_yes_ratio(self, image: Image.Image, question: str) -> float:
        """Run one VQA question and return the yes_ratio (0.0 to 1.0)."""
        prompt = self.VQA_TEMPLATE.format(question=question)

        inputs = self.processor(
            text=prompt, images=image, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1,
                output_scores=True,
                return_dict_in_generate=True,
            )

        first_token_logits = outputs.scores[0][0].float()
        probs = torch.softmax(first_token_logits, dim=-1)

        yes_id = self.processor.tokenizer.encode("yes", add_special_tokens=False)[0]
        no_id  = self.processor.tokenizer.encode("no",  add_special_tokens=False)[0]

        yes_prob = probs[yes_id].item()
        no_prob  = probs[no_id].item()
        total    = yes_prob + no_prob + 1e-9
        return yes_prob / total

    # ------------------------------------------------------------------
    # Step 1: Describe — ask LLaVA what the image shows
    # ------------------------------------------------------------------

    def _describe_relation(self, image: Image.Image, triple: Triple) -> str:
        """
        Ask LLaVA an open-ended question about the relationship between
        subject and object. Returns a descriptive sentence.

        This is the 'Describe' step of Describe-and-Compare.
        Forces complete sentence output (not fragments).
        """
        s = _clean_query_for_vqa(triple.subject)
        o = _clean_query_for_vqa(triple.obj)

        if triple.relation_type == "SPATIAL":
            question = f"Where is the {s} relative to the {o}?"
        elif triple.relation_type == "ATTRIBUTE":
            question = f"How would you describe the {s}?"
        else:
            question = f"What is the {s} doing with the {o}?"

        prompt = self.DESCRIBE_TEMPLATE.format(question=question)

        inputs = self.processor(
            text=prompt, images=image, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=60,
                do_sample=False,
            )

        input_len = inputs["input_ids"].shape[1]
        answer = self.processor.tokenizer.decode(
            output_ids[0][input_len:], skip_special_tokens=True
        ).strip()

        # If answer is too short (< 4 words), it's a fragment — unreliable
        if len(answer.split()) < 4:
            print(f"    [Describe] Short answer ({answer!r}), retrying with explicit prompt...")
            retry_prompt = (
                f"USER: <image>\nLook at the image. "
                f"Describe the relationship between the {s} and the {o}. "
                f"Write a full sentence. ASSISTANT: The"
            )
            inputs2 = self.processor(
                text=retry_prompt, images=image, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                output_ids2 = self.model.generate(
                    **inputs2, max_new_tokens=60, do_sample=False,
                )
            input_len2 = inputs2["input_ids"].shape[1]
            retry_answer = self.processor.tokenizer.decode(
                output_ids2[0][input_len2:], skip_special_tokens=True
            ).strip()
            # Prepend "The" since we seeded the generation with it
            if retry_answer and not retry_answer.lower().startswith("the"):
                retry_answer = "The " + retry_answer
            if len(retry_answer.split()) >= 4:
                answer = retry_answer
                print(f"    [Describe] Retry got: {answer!r}")

        return answer

    # ------------------------------------------------------------------
    # Step 2: Compare — use Llama to check semantic consistency
    # ------------------------------------------------------------------

    NLI_SYSTEM = (
        "You are a semantic consistency judge. You will be given two descriptions "
        "of a relationship in an image: one from a caption and one from an independent "
        "visual model. Determine if they are semantically consistent — i.e., do they "
        "describe the same relationship, even if worded differently?\n\n"
        "Answer ONLY 'consistent' or 'inconsistent' — nothing else."
    )

    NLI_USER_TEMPLATE = (
        "Caption claims: The relationship between '{subject}' and '{object}' is '{relation}'.\n"
        "Visual model describes: {description}\n\n"
        "Are these semantically consistent?"
    )

    def _compare_nli(self, triple: Triple, description: str) -> tuple[bool, float]:
        """
        Use Llama-3.3-70B to compare the caption's claim against LLaVA's
        description. Returns (is_consistent, confidence).

        This is the 'Compare' step of Describe-and-Compare.
        No threshold needed — Llama makes a binary semantic judgement.
        """
        import time

        if not self.together_client:
            # No Llama available — fall back to keyword heuristic
            return self._compare_heuristic(triple, description)

        user_msg = self.NLI_USER_TEMPLATE.format(
            subject=triple.subject,
            object=triple.obj,
            relation=triple.relation,
            description=description,
        )

        for attempt in range(3):
            try:
                response = self.together_client.chat.completions.create(
                    model=self.NLI_MODEL,
                    messages=[
                        {"role": "system", "content": self.NLI_SYSTEM},
                        {"role": "user",   "content": user_msg},
                    ],
                    max_tokens=10,
                    temperature=0.0,
                )
                answer = response.choices[0].message.content.strip().lower()

                if "inconsistent" in answer:
                    return False, 0.85   # high confidence hallucination
                elif "consistent" in answer:
                    return True, 0.85    # high confidence supported
                else:
                    return True, 0.3     # ambiguous → uncertain
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                print(f"  [NLI] Failed after 3 attempts: {e}")
                return True, 0.3   # fail open → uncertain

    def _compare_heuristic(self, triple: Triple, description: str) -> tuple[bool, float]:
        """
        Simple keyword-overlap fallback when Llama is unavailable.
        Checks if the relation word (or a close variant) appears in the description.
        """
        desc_lower = description.lower()
        rel_lower = triple.relation.lower()

        # Check direct match
        if rel_lower in desc_lower:
            return True, 0.7

        # Check -ing form
        verb_ing = _to_present_participle(rel_lower)
        if verb_ing in desc_lower:
            return True, 0.7

        # No match — possibly inconsistent, but low confidence
        return False, 0.5

    # ------------------------------------------------------------------
    # Combined: Describe-and-Compare verification
    # ------------------------------------------------------------------

    def verify(self, image: Image.Image, triple: Triple) -> tuple[bool, float]:
        """
        Dual-signal verification:
          Signal 1 (VQA):  Binary yes/no → yes_ratio
          Signal 2 (NLI):  Describe-and-Compare → semantic consistency

        Decision logic:
          - VQA ≥ 0.65 → supported (high confidence, skip NLI)
          - VQA < 0.40 → likely hallucinated → run NLI to confirm
              NLI inconsistent → hallucinated (both agree)
              NLI consistent   → uncertain (NLI overrides noisy VQA)
          - VQA ∈ [0.40, 0.65) → uncertain zone → run NLI as tiebreaker

        Returns (is_supported, confidence).
        """
        # ── Signal 1: Binary VQA (multi-question voting) ──────────────
        questions = self._build_paraphrased_questions(triple)
        yes_ratios = []
        for q in questions:
            try:
                yr = self._get_yes_ratio(image, q)
                yes_ratios.append(yr)
            except Exception as e:
                print(f"    [VQA] Error: {e}")
        avg_yes_ratio = sum(yes_ratios) / len(yes_ratios) if yes_ratios else 0.5
        print(f"    [VQA] ({triple.subject}, {triple.relation}, {triple.obj}): "
              f"yes_ratio={avg_yes_ratio:.3f} ({len(questions)} questions)")

        # High confidence supported → no NLI needed
        if avg_yes_ratio >= self.VQA_SUPPORTED_THRESHOLD:
            return True, 0.80

        # ── Signal 2: Describe-and-Compare (NLI confirmation) ─────────
        try:
            description = self._describe_relation(image, triple)
        except Exception as e:
            print(f"    [Describe] Failed: {e}")
            # VQA said uncertain or hallucinated, but NLI failed → uncertain
            return True, 0.3

        if not description or len(description.split()) < 3:
            # Can't get a useful description → rely on VQA alone
            if avg_yes_ratio < self.VQA_HALLUCINATED_THRESHOLD:
                return False, 0.55  # weak hallucination signal
            return True, 0.3       # uncertain

        # Store description as evidence (used by corrector if hallucinated)
        triple.vqa_evidence = description

        is_consistent, nli_conf = self._compare_nli(triple, description)
        print(f"    [NLI] consistent={is_consistent}, conf={nli_conf:.2f}, "
              f"evidence={description!r}")

        # ── Decision logic ────────────────────────────────────────────
        if avg_yes_ratio < self.VQA_HALLUCINATED_THRESHOLD:
            # VQA says hallucinated — NLI confirms or overrides
            if not is_consistent:
                return False, 0.85   # BOTH agree → high confidence hallucination
            else:
                return True, 0.35    # NLI disagrees → uncertain (fail open)
        else:
            # VQA uncertain [0.40, 0.65) — NLI is the tiebreaker
            if not is_consistent and nli_conf >= 0.7:
                return False, 0.65   # NLI confident → hallucinated
            elif is_consistent:
                return True, 0.60    # NLI says OK → supported
            else:
                return True, 0.35    # both uncertain → fail open

    # Keep gather_evidence for backward compatibility — now just returns
    # the already-stored vqa_evidence from the verify() step.
    def gather_evidence(self, image: Image.Image, triple: Triple) -> str:
        """Return stored evidence from verify(), or generate fresh if needed."""
        if triple.vqa_evidence:
            return triple.vqa_evidence
        return self._describe_relation(image, triple)

    def verify_batch(self, image: Image.Image, triples: list[Triple]) -> list[tuple[bool, float]]:
        return [self.verify(image, t) for t in triples]


# ---------------------------------------------------------------------------
# Combined Verifier (routes to the right strategy)
# ---------------------------------------------------------------------------

class RelationVerifier:
    """
    Top-level verifier that routes each triple to the appropriate strategy:
      - SPATIAL             → SpatialVerifier (OWL-ViT + geometry)
                              fallback → LLaVA VQA if OWL-ViT confidence low
      - ACTION / ATTRIBUTE  → LLaVAVerifier (independent cross-model VQA)
                              fallback → BLIP-2 VQAVerifier if LLaVA not available
      - OTHER               → LLaVAVerifier / VQAVerifier (fallback)

    Using LLaVA for action/attribute verification eliminates the circular
    self-verification problem (BLIP-2 verifying its own captions).

    Populates triple.hallucinated = True/False in place.
    """

    CONFIDENCE_THRESHOLD = 0.45

    def __init__(self, llava_model=None, llava_processor=None,
                 num_paraphrases: int = 3, together_client=None):
        """
        Args:
            llava_model:      Injected LLaVA model for cross-model verification.
            llava_processor:  Injected LLaVA processor.
            num_paraphrases:  Kept for backward compat.
            together_client:  together.Together client for Llama NLI (Describe-and-Compare).
        If LLaVA is not provided, falls back to BLIP-2 VQA (with yes-bias caveat).
        """
        self.spatial      = SpatialVerifier()
        self.skip_spatial = False
        self.skip_vqa     = False

        if llava_model is not None:
            self.llava = LLaVAVerifier(
                model=llava_model,
                processor=llava_processor,
                together_client=together_client,
                num_paraphrases=num_paraphrases,
            )
            self.vqa   = None
            print("[RelationVerifier] Using Describe-and-Compare verification (LLaVA + Llama NLI).")
        else:
            self.llava = None
            self.vqa   = VQAVerifier()
            print("[RelationVerifier] LLaVA not available — using BLIP-2 VQA (yes-bias caveat).")

    def _action_attr_verify(self, image: Image.Image, triple: Triple) -> tuple[bool, float]:
        """Route action/attribute triple to LLaVA (preferred) or BLIP-2 VQA (fallback)."""
        if self.llava is not None:
            return self.llava.verify(image, triple)
        return self.vqa.verify(image, triple)

    def verify_triple(self, image: Image.Image, triple: Triple) -> Triple:
        """
        Verify one triple. Sets triple.hallucinated in place.

        Routing:
          SPATIAL  → OWL-ViT geometry → LLaVA fallback if low confidence
          ACTION / ATTRIBUTE / OTHER → LLaVA (or BLIP-2 if LLaVA unavailable)
        """
        # Route by relation type
        if triple.relation_type == "SPATIAL" and not self.skip_spatial:
            is_supported, conf = self.spatial.verify(image, triple)
            if conf < self.CONFIDENCE_THRESHOLD:
                # OWL-ViT couldn't find the objects — fall back to cross-model VQA
                if self.skip_vqa:
                    triple.hallucinated = None
                    return triple
                is_supported, conf = self._action_attr_verify(image, triple)
        else:
            # ACTION / ATTRIBUTE / OTHER → VQA
            if self.skip_vqa:
                triple.hallucinated = None
                return triple
            is_supported, conf = self._action_attr_verify(image, triple)

        if conf >= self.CONFIDENCE_THRESHOLD:
            triple.hallucinated = not is_supported
        else:
            triple.hallucinated = None   # uncertain

        return triple

    def verify_all(self, image: Image.Image, triples: list[Triple]) -> list[Triple]:
        """Verify a list of triples. Returns the same list with hallucinated set."""
        for triple in triples:
            self.verify_triple(image, triple)
        return triples

    def gather_evidence(self, image: Image.Image, triple: Triple) -> str:
        """
        Gather descriptive VQA evidence for a hallucinated triple.
        Uses LLaVA to describe what the image actually shows for this relationship.
        Falls back to empty string if LLaVA is unavailable.
        """
        if self.llava is not None:
            try:
                return self.llava.gather_evidence(image, triple)
            except Exception as e:
                print(f"[RelationVerifier] Evidence gathering failed: {e}")
                return ""
        return ""

    def gather_evidence_batch(self, image: Image.Image, triples: list[Triple]) -> None:
        """
        Gather VQA evidence for all hallucinated triples and store in triple.vqa_evidence.
        Only queries triples where hallucinated=True.
        """
        for t in triples:
            if t.hallucinated is True:
                evidence = self.gather_evidence(image, t)
                t.vqa_evidence = evidence
                if evidence:
                    print(f"  [Evidence] ({t.subject}, {t.relation}, {t.obj}) → \"{evidence}\"")


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
