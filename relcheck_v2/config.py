"""
RelCheck v2 — Configuration
============================
All constants, thresholds, model IDs, and keyword sets in one place.
Nothing here imports from other relcheck_v2 modules (leaf dependency).
"""

from __future__ import annotations

# ── Model IDs ────────────────────────────────────────────────────────────

LLM_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
VLM_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
GDINO_ID = "IDEA-Research/grounding-dino-tiny"

# ── Detection thresholds ─────────────────────────────────────────────────

GDINO_BOX_THRESHOLD: float = 0.15
GDINO_TEXT_THRESHOLD: float = 0.10

# ── Spatial geometry ─────────────────────────────────────────────────────

SPATIAL_DEADZONE: float = 0.08  # centroid distance within this → ambiguous

# ── VQA decision thresholds ──────────────────────────────────────────────

YES_SUPPORTED: float = 0.65   # avg yes_ratio >= this → supported
YES_UNSUPPORTED: float = 0.40  # avg yes_ratio <  this → unsupported
# Between 0.40 and 0.65 → uncertain

# ── Caption length threshold ─────────────────────────────────────────────

SHORT_CAPTION_THRESHOLD: int = 30  # words; below → enrichment, above → correction

# ── Spatial relation keywords ────────────────────────────────────────────

SPATIAL_RELS: frozenset[str] = frozenset({
    "left of", "to the left of", "to the left",
    "right of", "to the right of", "to the right",
    "above", "on top of", "over", "on",
    "below", "under", "beneath", "underneath",
    "in front of", "behind", "in back of",
    "inside", "outside",
})

# ── Action keywords ──────────────────────────────────────────────────────

ACTION_WORDS: frozenset[str] = frozenset({
    "riding", "holding", "carrying", "eating", "drinking", "wearing",
    "pushing", "pulling", "walking", "running", "sitting", "standing",
    "playing", "using", "throwing", "catching", "driving", "feeding",
    "reading", "watching", "kicking", "touching", "hugging", "kissing",
    "jumping", "climbing", "mounted", "chained",
})

# ── Spatial opposites (for contradiction detection) ──────────────────────

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

# ── Counterfactual relations (for contrastive VQA) ───────────────────────

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

# ── Entity synonyms (visual entities, hand-curated) ─────────────────────

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

# ── Broad detection categories (for KB construction) ─────────────────────

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

# ── Captioner configuration ─────────────────────────────────────────────

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

# ── R-POPE evaluation ──────────────────────────────────────────────────

RPOPE_PROMPT_TMPL: str = (
    "Based ONLY on this description, answer the question with Yes or No.\n\n"
    'Description: "CAPTION_PLACEHOLDER"\n'
    "Question: QUESTION_PLACEHOLDER\n\n"
    "Answer ONLY Yes or No."
)

# ── Google Drive paths (Colab defaults) ──────────────────────────────────

DRIVE_IMAGES_DIR = "/content/drive/MyDrive/RelCheck_Data/images"
RBENCH_PATH = "/content/drive/MyDrive/RelCheck_Data/rbench_data.json"
