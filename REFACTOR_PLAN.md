# RelCheck — Code Refactor Plan

**Goal:** Senior SWE/MLE-level code quality. Zero behavioral change — identical outputs before and after.

**Approach:** Create a new `relcheck_v2/` package from scratch. The old `relcheck/` (Session 2-3, OWL-ViT + BLIP-2 VQA + Mistral) is untouched. The existing notebooks are untouched. We build clean modules, then rewire the notebooks to import from `relcheck_v2/`.

---

## Target structure

```
RelCheck/
├── relcheck_v2/                    # NEW — all refactored code lives here
│   ├── __init__.py                 # Package init, version string
│   ├── config.py                   # Constants, thresholds, model IDs, keyword sets
│   ├── prompts.py                  # All LLM prompt templates (6 templates)
│   ├── api.py                      # LLM/VLM clients, retry logic, encode_b64
│   ├── models.py                   # Model loading: GDINO, LLaVA, BLIP-2, Qwen
│   ├── detection.py                # detect_objects (batched GDINO), dedup, clean_label
│   ├── spatial.py                  # Bbox geometry rules, compute_spatial_facts
│   ├── entity.py                   # Entity matching, normalization, synonyms, core_noun
│   ├── verification.py             # verify_triple, verify_action_triple, classify_rel
│   ├── kb.py                       # Visual KB construction (GDINO + VLM)
│   ├── correction.py               # correct_long_caption_v2, enrich_short_caption
│   ├── injection.py                # Synthetic test: question→statement, parse_question
│   ├── evaluation.py               # rpope_judge, summary printing
│   └── captioning.py               # Caption generation (BLIP-2, LLaVA, Qwen wrappers)
│
├── relcheck/                       # OLD — untouched (Session 2-3 code)
├── RelCheck_Synthetic_Test.ipynb   # Will be rewired to import from relcheck_v2/
├── RelCheck_600.ipynb              # Will be rewired to import from relcheck_v2/
└── ...
```

---

## External libraries to adopt

These replace hand-rolled code with battle-tested, industry-standard implementations.

### High priority (big code reduction + better quality)

| Library | Replaces | What changes |
|---------|----------|-------------|
| **`rapidfuzz`** | `levenshtein()` (15 lines of DP) | `rapidfuzz.distance.Levenshtein.distance(s1, s2)` — C extension, ~100x faster |
| **`rapidfuzz`** | `_entity_matches()` (13 lines of substring heuristics) | `rapidfuzz.fuzz.token_sort_ratio(a, b) > 80` — handles word reordering, partial matches |
| **`spacy`** (en_core_web_sm) | `extract_nouns_simple()` (12 lines + 50-word stoplist) | `[chunk.root.text for chunk in nlp(caption).noun_chunks]` — handles multi-word nouns ("baseball bat") |
| **`spacy`** | `_core_noun()` (21 lines × 2 copies, regex heuristics) | `doc[-1].lemma_` via dependency parse — linguistically correct head noun extraction |
| **`torchvision`** | IoU calculation inside `dedup()` (unreadable one-liner) | `torchvision.ops.box_iou(boxes_a, boxes_b)` — already have torch loaded for GDINO |
| **`tenacity`** | Retry logic in `llm_call()` (hand-rolled backoff) | `@retry(wait=wait_exponential(min=1, max=8), stop=stop_after_attempt(3))` — industry standard |

### Medium priority (cleaner code, marginal quality improvement)

| Library | Replaces | What changes |
|---------|----------|-------------|
| **`nltk.stem`** | Implicit lemmatization in `_candidate_synonyms` | `WordNetLemmatizer().lemmatize(word)` — normalizes "riding"→"ride", "cats"→"cat" before synonym lookup |
| **`pydantic`** | Loose dicts for triples, verdicts, KB entries | Typed dataclasses with validation — catches malformed LLM outputs early |

### Not replacing (custom logic is better)

| Function | Why keep custom |
|----------|----------------|
| `_spatial_verdict_from_boxes` | Domain-specific geometry rules — no library does "is subject left-of object given bboxes" |
| `compute_spatial_facts` | Same — pairwise centroid spatial reasoning is RelCheck-specific |
| `_verify_triple` | Core pipeline logic — orchestration, not a library concern |
| `_ENTITY_SYNONYMS` dict | Hand-curated for visual entities (man/person/boy). WordNet gives too many unrelated senses. Keep the map, supplement with lemmatizer |
| `_COUNTERFACTUAL_MAP` | Domain-specific relation opposites — no library for this |
| `_has_garble` | Trim from 189 lines to ~30 lines of key regex checks. Not worth a heavy dependency (LanguageTool needs Java) |

### Install line for Colab

```python
!pip install rapidfuzz tenacity pydantic -q
# spacy and torchvision are already pre-installed in Colab
# nltk is already pre-installed in Colab
!python -m spacy download en_core_web_sm -q
```

---

## Step-by-step extraction plan

Each step is one commit. Each step produces a working module that can be tested independently. No existing files are modified until the final rewiring step.

---

### Step 0 — Baseline capture (before any code changes)

Run `RelCheck_Synthetic_Test.ipynb` with N_IMAGES=5. Save all JSON outputs as `baseline_*.json`. Every subsequent step diffs against these to confirm zero behavioral change.

---

### Step 1 — `config.py` + `prompts.py`

**`config.py`** — Extract every constant, threshold, keyword set, and model ID scattered across Cells 1, 4, 6, 7 into one file.

```python
# config.py
from dataclasses import dataclass

# ── Model IDs ──
LLM_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
VLM_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
GDINO_ID  = "IDEA-Research/grounding-dino-tiny"

# ── Detection thresholds ──
GDINO_BOX_THRESHOLD  = 0.15
GDINO_TEXT_THRESHOLD  = 0.10
SPATIAL_DEADZONE      = 0.08

# ── VQA decision thresholds ──
YES_SUPPORTED   = 0.65
YES_UNSUPPORTED = 0.40

# ── Caption thresholds ──
SHORT_CAPTION_THRESHOLD = 30  # words

# ── Keyword sets ──
SPATIAL_RELS: set[str] = {"left of", "to the left of", ...}
ACTION_WORDS: set[str] = {"riding", "holding", ...}
COUNTERFACTUAL_MAP: dict[str, str] = {"riding": "standing next to", ...}
SPATIAL_OPPOSITES: dict[str, str] = {"left": "right", ...}
ENTITY_SYNONYMS: dict[str, list[str]] = {"man": ["person", "boy", ...], ...}

# ── KB construction ──
BROAD_CATEGORIES: list[str] = ["person", "man", "woman", ...]
```

**`prompts.py`** — All 6 prompt templates as named constants.

```python
# prompts.py
ANALYSIS_PROMPT = """You are a caption quality improver..."""
VERIFY_PROMPT = """Check if this caption is accurate..."""
TRIPLE_EXTRACT_PROMPT = """Extract all relational claims..."""
TRIPLE_CORRECT_PROMPT = """Edit exactly one relationship..."""
BATCH_CORRECT_PROMPT = """You are correcting an image caption..."""
MISSING_FACTS_PROMPT = """You are improving an image caption..."""
```

**Risk:** None. Just extracting constants.

---

### Step 2 — `api.py`

Replace 3 copies of `encode_b64`/`_get_b64`, the retry logic in `llm_call`, and the `vlm_call` wrapper.

```python
# api.py
import base64, time
from io import BytesIO
from PIL import Image
from tenacity import retry, wait_exponential, stop_after_attempt
from together import Together

from .config import LLM_MODEL, VLM_MODEL

_client: Together | None = None

def init_client(api_key: str) -> None:
    global _client
    _client = Together(api_key=api_key)

def encode_b64(image: Image.Image, quality: int = 85) -> str:
    buf = BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()

@retry(wait=wait_exponential(min=1, max=8), stop=stop_after_attempt(3))
def llm_call(messages: list[dict], model: str = LLM_MODEL,
             max_tokens: int = 600) -> str | None:
    resp = _client.chat.completions.create(
        model=model, messages=messages,
        temperature=0.0, max_tokens=max_tokens)
    return resp.choices[0].message.content.strip()

def vlm_call(messages: list[dict], max_tokens: int = 10) -> str | None:
    return llm_call(messages, model=VLM_MODEL, max_tokens=max_tokens)

def vlm_yesno(image: Image.Image, question: str) -> tuple[float, float]:
    """Yes/no VQA via text answer. Returns (yes_ratio, confidence)."""
    ...
```

**Kills:** 3 duplicate `encode_b64` functions, hand-rolled retry loops, 2 duplicate `vlm_call` definitions.

---

### Step 3 — `entity.py`

Replace hand-rolled noun extraction and entity matching with spaCy + rapidfuzz.

```python
# entity.py
import spacy
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from nltk.stem import WordNetLemmatizer

from .config import ENTITY_SYNONYMS

_nlp = None
_lemmatizer = WordNetLemmatizer()

def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp

def extract_nouns(text: str) -> list[str]:
    """Extract noun phrases using spaCy noun chunks."""
    doc = _get_nlp()(text)
    return list({chunk.root.lemma_.lower() for chunk in doc.noun_chunks})

def core_noun(phrase: str) -> str:
    """Extract head noun via spaCy dependency parse."""
    doc = _get_nlp()(phrase.strip())
    # Last non-stop token is typically the head noun
    tokens = [t for t in doc if not t.is_stop and t.is_alpha]
    return tokens[-1].lemma_.lower() if tokens else phrase.lower().strip()

def normalize(text: str) -> str:
    """Lowercase, strip articles."""
    text = text.lower().strip()
    for art in ("a ", "an ", "the ", "some "):
        if text.startswith(art):
            text = text[len(art):]
    return text.strip()

def entity_matches(a: str, b: str, threshold: int = 80) -> bool:
    """Fuzzy entity matching via rapidfuzz token_sort_ratio."""
    if not a or not b:
        return False
    # Exact core noun match (fast path)
    if core_noun(a) == core_noun(b):
        return True
    # Synonym check
    syns_a = candidate_synonyms(core_noun(a))
    syns_b = candidate_synonyms(core_noun(b))
    if syns_a & syns_b:
        return True
    # Fuzzy fallback
    return fuzz.token_sort_ratio(normalize(a), normalize(b)) >= threshold

def candidate_synonyms(name: str) -> set[str]:
    """All synonyms for a name, plus the name itself and its lemma."""
    lemma = _lemmatizer.lemmatize(name)
    syns = {name, lemma}
    for key in (name, lemma):
        syns.update(ENTITY_SYNONYMS.get(key, []))
    return syns

def levenshtein_distance(s1: str, s2: str) -> int:
    """Edit distance via rapidfuzz (C extension)."""
    return Levenshtein.distance(s1, s2)

def edit_rate(before: str, after: str) -> float:
    """Normalized Levenshtein distance."""
    max_len = max(len(before), len(after), 1)
    return levenshtein_distance(before, after) / max_len
```

**Kills:** `levenshtein` (15 lines), `_core_noun` (21 lines × 2 copies), `normalize_entity` (10 lines), `_clean_label` (6 lines), `_entity_matches` (13 lines), `extract_nouns_simple` (12 lines + stopword set), `_candidate_synonyms` (35 lines). Total: ~130 lines replaced.

---

### Step 4 — `detection.py`

Clean GDINO detection with torchvision IoU for dedup.

```python
# detection.py
import torch
from torchvision.ops import box_iou
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from .config import GDINO_ID, GDINO_BOX_THRESHOLD, GDINO_TEXT_THRESHOLD
from .entity import normalize

def detect_objects(image: Image.Image, queries: list[str],
                   batch_size: int = 4) -> list[tuple[str, float, list[float]]]:
    """Batched GDINO detection with list-of-lists format."""
    ...

def dedup_detections(dets: list[tuple], iou_threshold: float = 0.5
                     ) -> list[tuple[str, float, list[float]]]:
    """Remove duplicate detections using torchvision IoU."""
    ...

def find_best_bbox(entity: str, detections: list[tuple]) -> list[float] | None:
    """Find highest-confidence bbox matching entity name."""
    ...

def crop_to_bboxes(image: Image.Image, box1: list[float], box2: list[float],
                   padding: float = 0.15) -> Image.Image:
    """Crop image to region containing both bounding boxes."""
    ...
```

**Kills:** Unreadable IoU one-liner in `dedup`, 2 copies of `_find_best_bbox` (different interfaces), 2 copies of `_crop_to_bboxes`.

---

### Step 5 — `spatial.py`

Deterministic geometry functions. These stay mostly as-is since they're domain-specific, but cleaned up.

```python
# spatial.py
from .config import SPATIAL_DEADZONE, SPATIAL_OPPOSITES

def spatial_verdict(subj_box: list[float], obj_box: list[float],
                    relation: str) -> bool | None:
    """Deterministic spatial relation check from bounding box centroids.
    Returns True (supported), False (contradicted), None (ambiguous)."""
    ...

def compute_spatial_facts(detections: list[tuple]) -> list[str]:
    """Compute pairwise spatial relationships from detections."""
    ...

def check_spatial_contradictions(caption: str,
                                 spatial_facts: list[str]) -> list[dict]:
    """Find caption claims that contradict deterministic spatial KB."""
    ...
```

---

### Step 6 — `verification.py`

The core verify_triple orchestrator + VQA action verification.

```python
# verification.py
from PIL import Image
from .config import YES_SUPPORTED, YES_UNSUPPORTED, COUNTERFACTUAL_MAP
from .detection import find_best_bbox, crop_to_bboxes
from .spatial import spatial_verdict
from .api import vlm_call, encode_b64

def classify_relation(rel: str) -> str:
    """Classify relation as SPATIAL, ACTION, or ATTRIBUTE."""
    ...

def verify_triple(subject: str, relation: str, object_: str,
                  detections: list[tuple], image: Image.Image) -> bool | None:
    """Full verification: geometry → crop VQA → contrastive.
    Returns True (supported), False (hallucinated), None (uncertain)."""
    ...

def verify_action_triple(subject: str, relation: str, object_: str,
                         kb: dict, image: Image.Image,
                         n_questions: int = 3) -> dict:
    """VQA-based action/attribute verification with contrastive question."""
    ...

def decide_verdict(triple: dict, geometry_result: dict | None = None,
                   vqa_result: dict | None = None) -> dict:
    """Apply threshold rules to produce final verdict."""
    ...
```

---

### Step 7 — `kb.py`

Visual knowledge base construction.

```python
# kb.py
from PIL import Image
from .detection import detect_objects, dedup_detections
from .spatial import compute_spatial_facts
from .entity import extract_nouns
from .api import vlm_call, encode_b64
from .config import BROAD_CATEGORIES

def build_visual_kb(image: Image.Image, caption: str) -> dict:
    """Build 3-layer KB: hard facts + spatial facts + VLM description."""
    ...
```

---

### Step 8 — `correction.py`

Break the 614-line `_correct_long_caption_v2` into composable sub-functions.

```python
# correction.py
from PIL import Image
from .prompts import BATCH_CORRECT_PROMPT, ANALYSIS_PROMPT, VERIFY_PROMPT
from .entity import edit_rate
from .api import llm_call

def extract_triples(caption: str) -> list[dict]:
    """LLM triple extraction + type classification."""
    ...

def verify_all_triples(triples: list[dict], kb: dict,
                       image: Image.Image) -> list[dict]:
    """3-layer verification on each triple."""
    ...

def select_corrections(verified: list[dict], kb: dict,
                       image: Image.Image) -> list[dict]:
    """For INCORRECT triples, determine correct relation or DELETE."""
    ...

def apply_corrections(caption: str, corrections: list[dict]) -> str:
    """Surgical text editing via LLM batch correction."""
    ...

def correct_long_caption(img_id: str, caption: str, kb: dict,
                         image: Image.Image,
                         cross_captions: dict | None = None) -> dict:
    """Orchestrator: extract → verify → select → apply → post-process."""
    triples = extract_triples(caption)
    verified = verify_all_triples(triples, kb, image)
    corrections = select_corrections(verified, kb, image)
    corrected = apply_corrections(caption, corrections) if corrections else caption
    return post_process(img_id, caption, corrected, kb, verified, corrections)

def enrich_short_caption(img_id: str, caption: str, kb: dict) -> dict:
    """KB-guided full rewrite for short (< 30 word) captions."""
    ...

def enrich_caption_v3(img_id: str, caption: str, kb: dict,
                      image: Image.Image | None = None,
                      cross_captions: dict | None = None) -> dict:
    """Unified entry point: auto-selects enrichment vs correction."""
    ...
```

**Kills:** 614-line monolith → 5 functions of ≤80 lines each + a 10-line orchestrator.

---

### Step 9 — `injection.py` + `evaluation.py` + `captioning.py`

Smaller modules for the remaining notebook-specific code.

**`injection.py`** — `question_to_statement`, `parse_question` (from Cell 4).

**`evaluation.py`** — `rpope_judge`, `print_rpope_summary` (from Cell 8).

**`captioning.py`** — `caption_image`, `caption_image_llava`, `caption_image_blip2`, `caption_image_api` (from Cell 3).

---

### Step 10 — `models.py`

Centralized model loading with lazy initialization.

```python
# models.py
import torch
from transformers import (AutoProcessor, AutoModelForZeroShotObjectDetection,
                          LlavaForConditionalGeneration, Blip2ForConditionalGeneration,
                          Blip2Processor, AutoProcessor as LlavaProcessor)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_gdino_model = None
_gdino_processor = None

def get_gdino():
    global _gdino_model, _gdino_processor
    if _gdino_model is None:
        from .config import GDINO_ID
        _gdino_processor = AutoProcessor.from_pretrained(GDINO_ID)
        _gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_ID).to(DEVICE)
    return _gdino_model, _gdino_processor

# Same lazy pattern for LLaVA, BLIP-2...
```

---

### Step 11 — Rewire notebooks

Replace inline code in `RelCheck_Synthetic_Test.ipynb` cells with imports:

**Cell 1 (291 → ~15 lines):**
```python
!pip install rapidfuzz tenacity pydantic -q
!python -m spacy download en_core_web_sm -q
import sys; sys.path.insert(0, '/content/RelCheck')

from relcheck_v2.config import *
from relcheck_v2.api import init_client, llm_call, vlm_call, vlm_yesno, encode_b64
from relcheck_v2.models import get_gdino, get_llava, get_blip2
from relcheck_v2.detection import detect_objects
from relcheck_v2.verification import verify_triple, classify_relation
from relcheck_v2.entity import extract_nouns, entity_matches

init_client(TOGETHER_API_KEY)
```

**Cell 5 (detection — 75 → ~20 lines):**
```python
from relcheck_v2.verification import verify_triple
# ... thin loop calling verify_triple() ...
```

**Cell 7 (correction — 2,451 → ~30 lines):**
```python
from relcheck_v2.correction import correct_long_caption
# ... thin loop calling correct_long_caption() ...
```

Same treatment for `RelCheck_600.ipynb` and `RelCheck_PlanA.ipynb`.

---

### Step 12 — Verification + cleanup

1. Run `RelCheck_Synthetic_Test.ipynb` with N_IMAGES=5
2. Diff all JSON outputs against `baseline_*.json` — must be byte-identical
3. Run with N_IMAGES=20 for full validation
4. Delete stale standalone files: `test_vlm_logprob.py`, `test_gdino_geometry.py`, `cell_llava.py`, `viability_cell4_scout_captioner.py`, `eval_cells.py`
5. Add `relcheck_v2/` to git, commit, push

---

## Execution order (by priority)

| Order | Step | What | Risk | Time |
|-------|------|------|------|------|
| 1 | Step 0 | Baseline capture | None | 10 min |
| 2 | Step 1 | config.py + prompts.py | None | 20 min |
| 3 | Step 2 | api.py (tenacity, dedup encode_b64) | Low | 20 min |
| 4 | Step 3 | entity.py (spaCy, rapidfuzz) | Low | 30 min |
| 5 | Step 4 | detection.py (torchvision IoU) | Low | 25 min |
| 6 | Step 5 | spatial.py | None | 15 min |
| 7 | Step 6 | verification.py | Low | 25 min |
| 8 | Step 7 | kb.py | Low | 15 min |
| 9 | Step 8 | correction.py (break 614-line fn) | Medium | 45 min |
| 10 | Step 9 | injection.py + evaluation.py + captioning.py | Low | 20 min |
| 11 | Step 10 | models.py (lazy loading) | Low | 15 min |
| 12 | Step 11 | Rewire notebooks | Medium | 30 min |
| 13 | Step 12 | Verification + cleanup | None | 20 min |

**Total: ~4.5 hours**

---

## Rules

1. **No existing files modified** until Step 11 (rewiring). All new code goes in `relcheck_v2/`.
2. **One step = one commit.** Message format: `refactor(v2): Step N — module_name`
3. **External libraries** are additive only — `rapidfuzz`, `tenacity`, `pydantic`. spaCy, torchvision, nltk are already in Colab.
4. **Verify after each step** where possible. Steps 1-10 can have unit tests. Step 11-12 diffs against baseline JSONs.
5. **If a step breaks results, revert.** Don't debug — revert, rethink, retry.
6. **No new features.** Refactoring only.
