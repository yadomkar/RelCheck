"""
RelCheck v2 — Configuration
============================
All constants, thresholds, model IDs, and keyword sets in one place.
Nothing here imports from other relcheck_v2 modules (leaf dependency).
"""

from __future__ import annotations

# ════════════════════════════════════════════════════════════════════════════
# MODEL IDS
# ════════════════════════════════════════════════════════════════════════════

LLM_MODEL: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
VLM_MODEL: str = "Qwen/Qwen3-VL-8B-Instruct"
GDINO_ID: str = "IDEA-Research/grounding-dino-tiny"

# ════════════════════════════════════════════════════════════════════════════
# DETECTION THRESHOLDS
# ════════════════════════════════════════════════════════════════════════════

GDINO_BOX_THRESHOLD: float = 0.15
GDINO_TEXT_THRESHOLD: float = 0.10

# ════════════════════════════════════════════════════════════════════════════
# VQA DECISION THRESHOLDS
# ════════════════════════════════════════════════════════════════════════════

YES_SUPPORTED: float = 0.65   # avg yes_ratio >= this → supported
YES_UNSUPPORTED: float = 0.40  # avg yes_ratio <  this → unsupported
# Between 0.40 and 0.65 → uncertain

# ════════════════════════════════════════════════════════════════════════════
# CAPTION PROCESSING
# ════════════════════════════════════════════════════════════════════════════

SHORT_CAPTION_THRESHOLD: int = 30  # words; below → enrichment, above → correction

# ════════════════════════════════════════════════════════════════════════════
# ENRICHMENT & CORRECTION MAGIC NUMBERS
# ════════════════════════════════════════════════════════════════════════════

ENRICHMENT_MAX_SENTENCES: int = 10
"""Maximum number of sentences in enriched caption.
Prevents runaway rewrites that lose information."""

VERIFY_KB_MIN_LENGTH: int = 15
"""Minimum caption length (characters) to attempt KB verification.
Skips verification on very short captions that can't contain enough info."""

CORRECTION_LENGTH_RATIO_MIN: float = 0.70
"""Minimum ratio of corrected_length / original_length.
Below this → reject correction (deletes too much valid content)."""

CORRECTION_LENGTH_RATIO_MAX: float = 1.30
"""Maximum ratio of corrected_length / original_length.
Above this → reject correction (adds too much speculative content)."""

ADDENDUM_MAX_WORDS_ADDED: int = 30
"""Maximum words to add via addendum when enriching.
Prevents bloat while allowing meaningful additions."""

ADDENDUM_SURVIVAL_RATIO: float = 0.80
"""Minimum survival ratio of addendum words after KB verification.
Below this → addendum marked as low-confidence, may be rejected."""

# ════════════════════════════════════════════════════════════════════════════
# CROP-BASED VQA PARAMETERS
# ════════════════════════════════════════════════════════════════════════════

CROP_PADDING: float = 0.15
"""Padding ratio (±15%) around entity bbox for standard crop.
Provides context without including unrelated objects."""

CROP_PADDING_WIDE: float = 0.25
"""Padding ratio (±25%) for wide crops (relation context).
Used when both subject and object need visible context."""

# ════════════════════════════════════════════════════════════════════════════
# SPATIAL GEOMETRY THRESHOLDS
# ════════════════════════════════════════════════════════════════════════════

SPATIAL_DEADZONE: float = 0.08
"""Centroid distance within this → ambiguous spatial relation.
Used to mark uncertain geometries for VQA fallback."""

ADJACENCY_GAP_RATIO: float = 0.30
"""Maximum gap (as fraction of object size) to mark objects as adjacent.
Gap larger than this → objects are separated, not touching."""

MOUNTING_TOP_RATIO: float = 0.65
"""Centroid position threshold (normalized [0,1]) for "on" vs "standing near".
Below 0.65 in vertical space → likely 'on' or 'over' relation."""

CONTAINMENT_OVERLAP_MIN: float = 0.50
"""Minimum overlap ratio for "inside" relation.
Below this → containment is ambiguous, requires VQA confirmation."""

KEYPOINT_CONFIDENCE_MIN: float = 0.3
"""Minimum confidence score for pose keypoints.
Below this → keypoint considered unreliable for spatial analysis."""

NEAR_DISTANCE_THRESHOLD: float = 0.3
"""Distance ratio threshold for "near" or "next to" relations.
Normalized bbox distance ≤ 0.3 → considered spatially close."""

# ════════════════════════════════════════════════════════════════════════════
# ABSTRACT / NON-DETECTABLE ENTITIES (skip GDino object detection)
# ════════════════════════════════════════════════════════════════════════════

ABSTRACT_ENTITIES: frozenset[str] = frozenset({
    # Relative locations / scene regions
    "center", "middle", "left side", "right side", "left", "right",
    "top", "bottom", "foreground", "background", "front", "back",
    "corner", "edge", "side", "area", "portion", "section", "part",
    "left side of the image", "right side of the image",
    "left side of the scene", "right side of the scene",
    "center of the image", "center of the scene",
    "left portion", "right portion", "upper portion", "lower portion",
    # Abstract concepts that LLaVA likes to reference
    "game", "scene", "setting", "environment", "event", "activity",
    "action", "moment", "situation", "occasion", "atmosphere",
    "meal", "conversation", "view", "shot", "frame",
    # Relative descriptors often used as "objects" in triples
    "each other", "one another", "it", "them", "itself",
})
"""Entity names that are scene locations or abstract concepts, not physical
objects detectable by GroundingDINO. When a triple's subject or object is
in this set, skip object-existence checks to avoid false positive deletions."""

# ════════════════════════════════════════════════════════════════════════════
# SPATIAL RELATION KEYWORDS
# ════════════════════════════════════════════════════════════════════════════

# ── Spatial relation keywords ────────────────────────────────────────────

SPATIAL_RELS: frozenset[str] = frozenset({
    "left of", "to the left of", "to the left",
    "right of", "to the right of", "to the right",
    "above", "on top of", "over", "on",
    "below", "under", "beneath", "underneath",
    "in front of", "behind", "in back of",
    "inside", "outside",
})

# ════════════════════════════════════════════════════════════════════════════
# ACTION RELATION KEYWORDS
# ════════════════════════════════════════════════════════════════════════════

ACTION_WORDS: frozenset[str] = frozenset({
    "riding", "holding", "carrying", "eating", "drinking", "wearing",
    "pushing", "pulling", "walking", "running", "sitting", "standing",
    "playing", "using", "throwing", "catching", "driving", "feeding",
    "reading", "watching", "kicking", "touching", "hugging", "kissing",
    "jumping", "climbing", "mounted", "chained",
})

# ════════════════════════════════════════════════════════════════════════════
# SPATIAL RELATION OPPOSITES (for contradiction detection)
# ════════════════════════════════════════════════════════════════════════════

SPATIAL_OPPOSITES: dict[str, str] = {
    "left":         "right",
    "right":        "left",
    "above":        "below",
    "below":        "above",
    "on top of":    "below",
    "under":        "above",
    "over":         "under",
    "in front of":  "behind",
    "behind":       "in front of",
    "to the left":  "to the right",
    "to the right": "to the left",
}

# ════════════════════════════════════════════════════════════════════════════
# COUNTERFACTUAL RELATIONS (for contrastive VQA)
# ════════════════════════════════════════════════════════════════════════════

COUNTERFACTUAL_MAP: dict[str, str] = {
    "riding":       "standing next to",
    "sitting on":   "standing near",
    "holding":      "standing next to",
    "carrying":     "walking away from",
    "wearing":      "next to",
    "eating":       "looking at",
    "pulling":      "pushing",
    "pushing":      "pulling",
    "throwing":     "holding",
    "catching":     "dropping",
    "driving":      "standing near",
    "leading":      "following",
    "playing with": "ignoring",
    "using":        "near",
    "standing on":  "next to",
    "lying on":     "sitting near",
    "hanging from": "standing near",
    "leaning on":   "standing near",
}

# ════════════════════════════════════════════════════════════════════════════
# ENTITY SYNONYMS (visual entities, hand-curated)
# ════════════════════════════════════════════════════════════════════════════

ENTITY_SYNONYMS: dict[str, list[str]] = {
    "person":       ["person", "man", "woman", "child", "boy", "girl", "individual", "human", "people"],
    "car":          ["car", "vehicle", "automobile", "sedan", "suv", "truck"],
    "couch":        ["couch", "sofa", "settee", "loveseat"],
    "chair":        ["chair", "seat", "stool"],
    "dog":          ["dog", "puppy", "canine"],
    "cat":          ["cat", "kitten", "feline"],
    "horse":        ["horse", "pony"],
    "bicycle":      ["bicycle", "bike", "cycle"],
    "motorcycle":   ["motorcycle", "motorbike"],
    "truck":        ["truck", "lorry", "van"],
    "bus":          ["bus", "coach"],
    "boat":         ["boat", "ship", "vessel"],
    "airplane":     ["airplane", "plane", "aircraft", "jet"],
    "bird":         ["bird", "pigeon", "seagull", "sparrow"],
    "cow":          ["cow", "cattle", "bull"],
    "sheep":        ["sheep", "lamb"],
    "bench":        ["bench", "seat"],
    "dining table": ["table", "dining table", "desk"],
    "tv":           ["tv", "television", "monitor", "screen", "display"],
    "laptop":       ["laptop", "computer", "notebook"],
    "cell phone":   ["phone", "cell phone", "smartphone", "mobile"],
    "backpack":     ["backpack", "bag", "rucksack"],
    "handbag":      ["handbag", "bag", "purse"],
    "sports ball":  ["ball"],
    "cup":          ["cup", "mug", "glass"],
    "bottle":       ["bottle"],
    "bowl":         ["bowl"],
    "book":         ["book", "magazine"],
    "vase":         ["vase"],
    "clock":        ["clock", "watch"],
    "potted plant": ["plant", "flower", "pot"],
    "fire hydrant": ["fire hydrant", "hydrant"],
    "traffic light": ["traffic light", "stoplight"],
    "umbrella":     ["umbrella"],
    "frisbee":      ["frisbee"],
    "skateboard":   ["skateboard", "board"],
    "surfboard":    ["surfboard", "board"],
    "skis":         ["skis", "ski"],
    "kite":         ["kite"],
    "baseball bat": ["bat", "baseball bat"],
    "tennis racket": ["racket", "tennis racket"],
    "suitcase":     ["suitcase", "luggage", "bag"],
    "tie":          ["tie", "necktie"],
    "keyboard":     ["keyboard"],
    "mouse":        ["mouse"],
    "remote":       ["remote", "controller"],
    "pizza":        ["pizza"],
    "cake":         ["cake"],
    "sandwich":     ["sandwich"],
    "refrigerator": ["refrigerator", "fridge"],
    "oven":         ["oven", "stove"],
    "sink":         ["sink", "basin", "washbasin", "washbowl", "double basin"],
    "toilet":       ["toilet"],
    "bed":          ["bed"],
    "barbell":      ["barbell", "dumbbell", "weight", "bar"],
    "dumbbell":     ["dumbbell", "barbell", "weight"],
    "treadmill":    ["treadmill", "exercise machine"],
    "basin":        ["basin", "sink", "washbasin"],
    "faucet":       ["faucet", "tap", "spigot"],
    "box":          ["box", "crate", "chest", "container", "drawer"],
    "crate":        ["crate", "box", "container"],
    "basket":       ["basket", "bin", "hamper"],
    "shelf":        ["shelf", "rack", "ledge"],
    "barrel":       ["barrel", "drum", "keg"],
    "cart":         ["cart", "trolley", "wagon"],
}

# ════════════════════════════════════════════════════════════════════════════
# BROAD DETECTION CATEGORIES (for KB construction)
# ════════════════════════════════════════════════════════════════════════════

BROAD_CATEGORIES: list[str] = [
    "person", "man", "woman", "child", "boy", "girl",
    "dog", "cat", "bird", "horse", "animal",
    "car", "bicycle", "motorcycle", "bus", "truck",
    "chair", "table", "bench", "couch", "bed",
    "food", "plate", "bowl", "cup", "bottle", "glass",
    "bag", "umbrella", "phone", "book", "sign",
    "hat", "jacket", "vest", "helmet", "glasses",
    "tree", "flower", "grass", "water",
]

# ════════════════════════════════════════════════════════════════════════════
# CAPTIONER CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

CAPTIONER_MODELS: dict[str, str | None] = {
    "blip2": None,                          # loaded locally
    "qwen":  "Qwen/Qwen3-VL-8B-Instruct",
    "llava": "llava-hf/llava-1.5-7b-hf",   # loaded locally
}

DESCRIBE_PROMPT: str = (
    "Describe this image in detail. Include all objects, "
    "their spatial positions relative to each other, any actions "
    "or interactions taking place, and notable attributes like colors and sizes."
)

# ════════════════════════════════════════════════════════════════════════════
# R-POPE EVALUATION
# ════════════════════════════════════════════════════════════════════════════

RPOPE_PROMPT_TMPL: str = (
    "Based ONLY on this description, answer the question with Yes or No.\n\n"
    'Description: "CAPTION_PLACEHOLDER"\n'
    "Question: QUESTION_PLACEHOLDER\n\n"
    "Answer ONLY Yes or No."
)

# ════════════════════════════════════════════════════════════════════════════
# GOOGLE DRIVE PATHS (Colab defaults)
# ════════════════════════════════════════════════════════════════════════════

DRIVE_IMAGES_DIR: str = "/content/drive/MyDrive/RelCheck_Data/images"
RBENCH_PATH: str = "/content/drive/MyDrive/RelCheck_Data/rbench_data.json"

# ════════════════════════════════════════════════════════════════════════════
# NLI PRE-FILTER
# ════════════════════════════════════════════════════════════════════════════

ENABLE_NLI: bool = True
"""Master switch for the NLI-based KB verification pre-filter.
When False (default), the pipeline behaves identically to the non-NLI version.
When True, a batch NLI entailment check runs after triple extraction."""
