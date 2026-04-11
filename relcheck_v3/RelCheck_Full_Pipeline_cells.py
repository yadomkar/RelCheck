# ============================================================
# RelCheck v3 — Full End-to-End Pipeline (Colab)
# ============================================================
# Copy-paste each cell block into Google Colab.
# Cells are separated by: # ── CELL N ──
#
# End-to-end: generates hallucinated captions (Task 2),
# builds claims (Stages 1-4), constructs 3-layer KB
# (CLAIM + GEOM + SCENE), runs GPT-5.4 correction (Stage 5),
# and evaluates with caption editing metrics + POPE.
#
# Requires GPU (T4 or A100) + OpenAI API key.
#
# Cells:
#   0  — Config (edit this!)
#   1  — Setup: installs, clone repo, download weights, unzip images
#   2  — Filter images by RelTR vocab coverage
#   3  — Run hallucination generation (Task 2)
#   4  — Run claim generation (Stages 1-4)
#   5  — Build 3-layer Knowledge Bases
#   6  — Run GPT-5.4 correction (Stage 5)
#   7  — Save all results to Drive
#   8  — Display qualitative examples
#   9  — KB & correction statistics
#   10 — Caption editing evaluation (BLEU, ROUGE, CIDEr, SPICE)
#   11 — Side-by-side comparison (GT vs Hallu vs Corrected)
#   12 — Cost & timing summary


# ── CELL 0 — Config ─────────────────────────────────────────
OPENAI_API_KEY = ""               # <-- paste your OpenAI key
HF_TOKEN = ""                     # <-- paste your HuggingFace token
MAX_SAMPLES = 25                  # number of images to process (None = all)

# Paths on Drive
DRIVE_BASE = "/content/drive/MyDrive/RelCheck_Data"
COCO_ZIP = f"{DRIVE_BASE}/coco_zips/val2014.zip"
COCO_CAPS_PATH = f"{DRIVE_BASE}/coco_zips/annotations/captions_val2014.json"
COCO_INSTANCES_PATH = f"{DRIVE_BASE}/coco_zips/annotations/instances_val2014.json"
IMAGE_DIR = "/content/coco_val2014/val2014"

# Output
SAVE_DIR = f"{DRIVE_BASE}/full_pipeline_v3"

# GroundingDINO
GDINO_DIR = "/content/groundingdino_weights"
GDINO_CHECKPOINT = f"{GDINO_DIR}/groundingdino_swint_ogc.pth"

# RelTR (scene graph — optional, set False to skip)
ENABLE_RELTR = True
RELTR_CHECKPOINT = f"{DRIVE_BASE}/checkpoint0149.pth"

# Correction (Stage 5)
CORRECTION_MODEL = "gpt-5.4"     # GPT-5.4 for thinking + correction
REASONING_EFFORT = "high"         # none / low / medium / high / xhigh
MAX_EDIT_CHARS = 50               # max Levenshtein distance allowed
MIN_EDIT_CHARS = 5                # min Levenshtein distance required


# ── CELL 1 — Setup ──────────────────────────────────────────
# Uncomment and run these pip installs on first use:
# !pip install openai>=1.0 pydantic>=2.0 python-Levenshtein tqdm pandas Pillow transformers torch -q
# !pip install groundingdino-py -q
# !pip install pycocoevalcap -q
# !python -m spacy download en_core_web_md -q

import os, sys, json, time, zipfile
from google.colab import drive

drive.mount("/content/drive")
os.makedirs(SAVE_DIR, exist_ok=True)

# Clone / update repo
REPO_DIR = "/content/RelCheck"
if not os.path.exists(os.path.join(REPO_DIR, ".git")):
    os.system(f"git clone https://github.com/yadomkar/RelCheck.git {REPO_DIR}")
else:
    os.system(f"cd {REPO_DIR} && git pull -q")
sys.path.insert(0, REPO_DIR)

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

# GroundingDINO checkpoint
os.makedirs(GDINO_DIR, exist_ok=True)
if not os.path.exists(GDINO_CHECKPOINT):
    os.system(
        f"wget -q -O {GDINO_CHECKPOINT} "
        "https://github.com/IDEA-Research/GroundingDINO/releases/download/"
        "v0.1.0-alpha/groundingdino_swint_ogc.pth"
    )
print(f"GroundingDINO checkpoint: {os.path.exists(GDINO_CHECKPOINT)}")

import groundingdino
GDINO_CONFIG = os.path.join(
    os.path.dirname(groundingdino.__file__),
    "config", "GroundingDINO_SwinT_OGC.py",
)

# RelTR setup
if ENABLE_RELTR:
    RELTR_DIR = "/content/RelTR"
    if not os.path.exists(RELTR_DIR):
        os.system(f"git clone https://github.com/yrcong/RelTR.git {RELTR_DIR}")

import relcheck_v3.reltr.config as reltr_cfg
reltr_cfg.ENABLE_RELTR = ENABLE_RELTR
reltr_cfg.RELTR_CHECKPOINT_PATH = RELTR_CHECKPOINT
print(f"ENABLE_RELTR = {reltr_cfg.ENABLE_RELTR}")

# Unzip COCO images to local disk (faster I/O than Drive)
if not os.path.exists(IMAGE_DIR) or len(os.listdir(IMAGE_DIR)) < 100:
    print(f"Unzipping {COCO_ZIP}...")
    os.makedirs(os.path.dirname(IMAGE_DIR), exist_ok=True)
    with zipfile.ZipFile(COCO_ZIP, "r") as zf:
        zf.extractall(os.path.dirname(IMAGE_DIR))
print(f"Images: {len(os.listdir(IMAGE_DIR))} files")

# GPU check
import torch
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    mem_gb = torch.cuda.get_device_properties(0).total_mem / 1024**3
    print(f"GPU memory: {mem_gb:.1f} GB")
print("Setup complete.")


# ── CELL 2 — Filter images by RelTR vocab coverage ──────────
# Only keep COCO images whose annotated categories overlap with
# RelTR's Visual Genome object vocabulary.

from relcheck_v3.reltr.reltr import RELTR_OBJECT_CLASSES

# COCO → RelTR category mapping (38 exact + clear synonyms)
COCO_TO_RELTR = {
    # Exact matches
    "person": "person", "car": "car", "motorcycle": "motorcycle",
    "airplane": "airplane", "bus": "bus", "train": "train", "truck": "truck",
    "boat": "boat", "bench": "bench", "bird": "bird", "cat": "cat",
    "dog": "dog", "horse": "horse", "sheep": "sheep", "cow": "cow",
    "elephant": "elephant", "bear": "bear", "zebra": "zebra",
    "giraffe": "giraffe",
    "umbrella": "umbrella", "tie": "tie", "kite": "kite",
    "skateboard": "skateboard", "surfboard": "surfboard",
    "bottle": "bottle", "cup": "cup", "fork": "fork", "bowl": "bowl",
    "banana": "banana", "orange": "orange", "pizza": "pizza",
    "chair": "chair", "bed": "bed", "toilet": "toilet",
    "laptop": "laptop", "sink": "sink", "book": "book",
    "clock": "clock", "vase": "vase",
    # Synonyms
    "bicycle": "bike", "skis": "ski", "tennis racket": "racket",
    "wine glass": "glass", "baseball glove": "glove",
    "dining table": "table", "potted plant": "plant",
    "tv": "screen", "couch": "seat",
    "backpack": "bag", "handbag": "bag", "suitcase": "bag",
    "cell phone": "phone",
}

RELTR_SET = set(RELTR_OBJECT_CLASSES)

# Load COCO instance annotations for category info
if not os.path.exists(COCO_INSTANCES_PATH):
    raise FileNotFoundError(
        f"COCO instances annotation not found: {COCO_INSTANCES_PATH}\n"
        "Download: http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
    )

with open(COCO_INSTANCES_PATH) as f:
    instances = json.load(f)

cat_id_to_name = {c["id"]: c["name"] for c in instances["categories"]}

# image_id → set of COCO category names
image_categories: dict[int, set[str]] = {}
for ann in instances["annotations"]:
    img_id = ann["image_id"]
    cat_name = cat_id_to_name[ann["category_id"]]
    image_categories.setdefault(img_id, set()).add(cat_name)

# Filter: keep images where ≥1 COCO category maps to a RelTR class
reltr_compatible_ids: set[int] = set()
for img_id, cats in image_categories.items():
    mapped = {COCO_TO_RELTR.get(c) for c in cats} - {None}
    if mapped & RELTR_SET:
        reltr_compatible_ids.add(img_id)

print(f"Total COCO images with annotations: {len(image_categories)}")
print(f"RelTR-compatible images: {len(reltr_compatible_ids)}")

# Load captions, filter to RelTR-compatible images (1 per image)
with open(COCO_CAPS_PATH) as f:
    coco_caps = json.load(f)

seen = set()
filtered_annotations = []
for ann in coco_caps["annotations"]:
    img_id = ann["image_id"]
    if img_id in reltr_compatible_ids and img_id not in seen:
        fname = f"COCO_val2014_{int(img_id):012d}.jpg"
        if os.path.exists(os.path.join(IMAGE_DIR, fname)):
            seen.add(img_id)
            filtered_annotations.append({
                "image_id": img_id,
                "caption": ann["caption"],
            })

print(f"Filtered annotations (1/image, RelTR-compatible): {len(filtered_annotations)}")

FILTERED_ANN_PATH = f"{SAVE_DIR}/filtered_annotations.json"
with open(FILTERED_ANN_PATH, "w") as f:
    json.dump(filtered_annotations, f)


# ── CELL 3 — Run Hallucination Generation (Task 2) ──────────
# Generate synthetic hallucinated captions using GPT-4o-mini.
# Each caption has exactly one hallucination of type:
# OBJECT_EXISTENCE, ATTRIBUTE, INTERACTION, or COUNT.

from relcheck_v3.hallucination_generation.run import main as hallu_main

HALLU_SAVE_DIR = f"{SAVE_DIR}/hallucination_gen"
os.makedirs(HALLU_SAVE_DIR, exist_ok=True)

# Check for existing checkpoint
hallu_csv = f"{HALLU_SAVE_DIR}/output.csv"
if os.path.exists(hallu_csv):
    import pandas as pd
    existing = pd.read_csv(hallu_csv)
    print(f"Found existing hallu gen output: {len(existing)} records")
    print(f"  Accepted: {(existing['status']=='accepted').sum()}")
    print("  → Delete output.csv to re-run, or skip to Cell 4")
else:
    print(f"Running hallucination generation on {MAX_SAMPLES} samples...")
    t0 = time.time()
    hallu_main(
        dataset_name="coco-ee",
        annotation_path=FILTERED_ANN_PATH,
        image_dir=IMAGE_DIR,
        openai_api_key=OPENAI_API_KEY,
        output_dir=HALLU_SAVE_DIR,
        max_samples=MAX_SAMPLES,
        dry_run=False,
    )
    print(f"Hallu gen done in {(time.time()-t0)/60:.1f} min")

# Load results
import pandas as pd
hallu_df = pd.read_csv(hallu_csv)
accepted = hallu_df[hallu_df["status"] == "accepted"].copy()
print(f"\nAccepted: {len(accepted)} / {len(hallu_df)}")
print(f"\nType distribution:")
print(accepted["hallucination_type"].value_counts().to_string())


# ── CELL 4 — Run Claim Generation (Stages 1-4) ─────────────
# Build Visual Knowledge Base per sample using Woodpecker pipeline:
# KeyConceptExtractor → QuestionFormulator → VisualValidator → ClaimGenerator

from relcheck_v3.claim_generation.config import ClaimGenConfig
from relcheck_v3.claim_generation.models import InputSample
from relcheck_v3.claim_generation.pipeline import ClaimGenerationPipeline

CLAIM_SAVE_DIR = f"{SAVE_DIR}/claim_gen"
os.makedirs(CLAIM_SAVE_DIR, exist_ok=True)

# Build InputSamples from accepted hallucinations
claim_samples = []
for _, row in accepted.iterrows():
    img_id = str(row["image_id"])
    img_path = os.path.join(IMAGE_DIR, f"COCO_val2014_{int(img_id):012d}.jpg")
    if os.path.exists(img_path):
        claim_samples.append(InputSample(
            image_id=img_id,
            image_path=img_path,
            ref_cap=row["ref_cap"],
        ))

print(f"Running claim generation on {len(claim_samples)} samples...")
config = ClaimGenConfig(
    openai_api_key=OPENAI_API_KEY,
    detector_config=GDINO_CONFIG,
    detector_model_path=GDINO_CHECKPOINT,
    output_dir=CLAIM_SAVE_DIR,
    checkpoint_interval=50,
)

pipeline = ClaimGenerationPipeline(config)
t0 = time.time()
claim_results = pipeline.process_batch(claim_samples)
claim_elapsed = time.time() - t0
print(f"Claim gen done in {claim_elapsed/60:.1f} min")

successful_claims = [r for r in claim_results if r.success]
print(f"Successful: {len(successful_claims)} / {len(claim_results)}")


# ── CELL 5 — Build 3-Layer Knowledge Bases ──────────────────
# CLAIM (Woodpecker VKB) + GEOM (bbox geometry) + SCENE (RelTR)

from PIL import Image
from tqdm import tqdm
from relcheck_v3.kb import build_kb, KnowledgeBase

# Lookup from image_id → hallu row for metadata
hallu_lookup = {}
for _, row in accepted.iterrows():
    hallu_lookup[str(row["image_id"])] = row

kb_results: list[tuple] = []  # (claim_result, hallu_row, kb)
kb_errors = 0

t0 = time.time()
for r in tqdm(successful_claims, desc="Building KBs"):
    try:
        img_path = os.path.join(
            IMAGE_DIR, f"COCO_val2014_{int(r.image_id):012d}.jpg"
        )
        pil_image = None
        if ENABLE_RELTR and os.path.exists(img_path):
            pil_image = Image.open(img_path).convert("RGB")

        kb = build_kb(
            vkb=r.visual_knowledge_base,
            object_answers=r.object_answers,
            image=pil_image,
        )

        hallu_row = hallu_lookup.get(r.image_id)
        kb_results.append((r, hallu_row, kb))
    except Exception as e:
        kb_errors += 1
        print(f"[{r.image_id}] KB build failed: {e}")

kb_elapsed = time.time() - t0
print(f"Built {len(kb_results)} KBs in {kb_elapsed:.1f}s ({kb_errors} errors)")


# ── CELL 6 — Run GPT-5.4 Correction (Stage 5) ──────────────
# Stage 5a: GPT-5.4 with reasoning_effort=high identifies
#           the hallucinated claim via KB cross-reference
# Stage 5b: GPT-5.4 with reasoning_effort=none applies
#           Woodpecker-style surgical correction
#
# Edit-distance gate (5-50 chars) rejects too-large/small edits.

from relcheck_v3.correction.config import CorrectionConfig
from relcheck_v3.correction.corrector import HallucinationCorrector

correction_config = CorrectionConfig(
    openai_api_key=OPENAI_API_KEY,
    thinking_model=CORRECTION_MODEL,
    reasoning_effort=REASONING_EFFORT,
    max_edit_chars=MAX_EDIT_CHARS,
    min_edit_chars=MIN_EDIT_CHARS,
)
corrector = HallucinationCorrector(correction_config)

# Checkpoint support
CORRECTION_CKPT = f"{SAVE_DIR}/correction_checkpoint.json"
if os.path.exists(CORRECTION_CKPT):
    with open(CORRECTION_CKPT) as f:
        correction_cache = json.load(f)
    print(f"Loaded correction checkpoint: {len(correction_cache)} entries")
else:
    correction_cache = {}

correction_results = []  # (claim_result, hallu_row, kb, corrected_cap, result_dict)
corrected_count = 0
passthrough_count = 0
error_count = 0

t0 = time.time()
for i, (r, hallu_row, kb) in enumerate(tqdm(kb_results, desc="Stage 5 Correction")):

    # Check cache
    if r.image_id in correction_cache:
        cached = correction_cache[r.image_id]
        # Discard stale cache entries from the old two-stage corrector schema.
        if "edits" not in cached:
            print(f"[{r.image_id}] Discarding stale cache entry")
        else:
            correction_results.append(
                (r, hallu_row, kb, cached["corrected_cap"], cached)
            )
            if cached.get("was_corrected"):
                corrected_count += 1
            else:
                passthrough_count += 1
            continue

    try:
        kb_text = kb.format()
        result = corrector.run(r.ref_cap, kb_text)

        edits_dicts = [e.model_dump() for e in result.edits]
        result_dict = {
            "corrected_cap": result.corrected_caption,
            "edits": edits_dicts,
            "was_corrected": result.was_corrected,
            "passthrough_reason": result.passthrough_reason,
        }

        correction_results.append(
            (r, hallu_row, kb, result.corrected_caption, result_dict)
        )

        if result.was_corrected:
            corrected_count += 1
        else:
            passthrough_count += 1

        correction_cache[r.image_id] = result_dict

    except Exception as e:
        error_count += 1
        correction_results.append((r, hallu_row, kb, r.ref_cap, {
            "corrected_cap": r.ref_cap,
            "edits": [],
            "was_corrected": False,
            "passthrough_reason": "api_error",
        }))
        print(f"[{r.image_id}] Correction failed: {e}")

    # Checkpoint every 25 samples
    if (i + 1) % 25 == 0:
        with open(CORRECTION_CKPT, "w") as f:
            json.dump(correction_cache, f)
        print(f"  Checkpoint saved ({i+1}/{len(kb_results)})")

# Final checkpoint
with open(CORRECTION_CKPT, "w") as f:
    json.dump(correction_cache, f)

correction_elapsed = time.time() - t0
print(f"\nStage 5 done in {correction_elapsed/60:.1f} min")
print(f"  Corrected:    {corrected_count}")
print(f"  Passthrough:  {passthrough_count}")
print(f"  Errors:       {error_count}")


# ── CELL 7 — Save All Results to Drive ──────────────────────
# Master JSONL with all stages + summary CSV

MASTER_JSONL = f"{SAVE_DIR}/full_pipeline_results.jsonl"

with open(MASTER_JSONL, "w") as f:
    for r, hallu_row, kb, corrected_cap, result_dict in correction_results:
        rec = {
            "image_id": r.image_id,
            "gt_cap": hallu_row["gt_cap"] if hallu_row is not None else "",
            "ref_cap": r.ref_cap,
            "corrected_cap": corrected_cap,
            "hallucination_type": (
                hallu_row["hallucination_type"] if hallu_row is not None else ""
            ),
            "hallu_reason": hallu_row["reason"] if hallu_row is not None else "",
            "was_corrected": result_dict.get("was_corrected", False),
            "passthrough_reason": result_dict.get("passthrough_reason"),
            "edits": result_dict.get("edits", []),
            "kb_text": kb.format(),
            "n_claims": (
                len(kb.claims.count_claims)
                + len(kb.claims.specific_claims)
                + len(kb.claims.overall_claims)
            ),
            "n_spatial_facts": len(kb.spatial_facts),
            "n_scene_triples": len(kb.scene_graph),
            "claim_timings": {
                "s1": r.timings.stage1_seconds,
                "s2": r.timings.stage2_seconds,
                "s3": r.timings.stage3_seconds,
                "s4": r.timings.stage4_seconds,
                "total": r.timings.total_seconds,
            },
        }
        f.write(json.dumps(rec) + "\n")

print(f"Saved {len(correction_results)} records to {MASTER_JSONL}")

# Summary CSV
summary_rows = []
for r, hallu_row, kb, corrected_cap, result_dict in correction_results:
    edits = result_dict.get("edits", [])
    summary_rows.append({
        "image_id": r.image_id,
        "gt_cap": hallu_row["gt_cap"] if hallu_row is not None else "",
        "ref_cap": r.ref_cap,
        "corrected_cap": corrected_cap,
        "was_corrected": result_dict.get("was_corrected", False),
        "passthrough_reason": result_dict.get("passthrough_reason") or "",
        "n_edits": len(edits),
        "hallucination_type": (
            hallu_row["hallucination_type"] if hallu_row is not None else ""
        ),
        "edit_layers": ",".join(
            e.get("contradicted_by", "") for e in edits
        ),
    })

summary_df = pd.DataFrame(summary_rows)
summary_csv = f"{SAVE_DIR}/full_pipeline_summary.csv"
summary_df.to_csv(summary_csv, index=False)
print(f"Saved summary CSV to {summary_csv}")
display(summary_df.head(10))


# ── CELL 8 — Display Qualitative Examples ───────────────────
# Show side-by-side: image, GT caption, hallucinated caption,
# 3-layer KB, correction output.

import matplotlib.pyplot as plt

show_n = min(10, len(correction_results))

for r, hallu_row, kb, corrected_cap, result_dict in correction_results[:show_n]:
    was_corrected = result_dict.get("was_corrected", False)
    status = "CORRECTED" if was_corrected else "PASSTHROUGH"

    print(f"{'='*70}")
    print(f"Image ID:   {r.image_id}  [{status}]")
    if hallu_row is not None:
        print(f"Type:       {hallu_row['hallucination_type']}")
        print(f"GT Cap:     {hallu_row['gt_cap']}")
        print(f"Hallu Reason: {hallu_row['reason']}")
    print(f"Ref Cap:    {r.ref_cap}")
    print(f"Corrected:  {corrected_cap}")

    edits = result_dict.get("edits", [])
    if edits:
        print(f"\nApplied {len(edits)} edit(s):")
        for i, e in enumerate(edits, 1):
            print(f"  [{i}] \"{e['original_span']}\" → \"{e['replacement']}\"")
            print(f"      contradicted_by: {e['contradicted_by']}")
            print(f"      evidence:        {e['evidence']}")
            print(f"      confidence:      {e['confidence']}")
    elif result_dict.get("passthrough_reason"):
        print(f"\nPassthrough reason: {result_dict['passthrough_reason']}")

    print(f"\n3-Layer KB:")
    print(kb.format())

    # Show image
    img_path = os.path.join(
        IMAGE_DIR, f"COCO_val2014_{int(r.image_id):012d}.jpg"
    )
    if os.path.exists(img_path):
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.imshow(Image.open(img_path))
        ax.set_title(f"ID: {r.image_id} [{status}]")
        ax.axis("off")
        plt.tight_layout()
        plt.show()
    print()


# ── CELL 9 — KB & Correction Statistics ─────────────────────
if correction_results:
    # KB layer sizes
    claim_counts = []
    spatial_counts = []
    scene_counts = []

    for r, _, kb, _, _ in correction_results:
        claim_counts.append(
            len(kb.claims.count_claims)
            + len(kb.claims.specific_claims)
            + len(kb.claims.overall_claims)
        )
        spatial_counts.append(len(kb.spatial_facts))
        scene_counts.append(len(kb.scene_graph))

    print(f"KB Statistics ({len(correction_results)} samples):")
    print(f"  Claims:        avg={sum(claim_counts)/len(claim_counts):.1f}, "
          f"min={min(claim_counts)}, max={max(claim_counts)}")
    print(f"  Spatial facts: avg={sum(spatial_counts)/len(spatial_counts):.1f}, "
          f"min={min(spatial_counts)}, max={max(spatial_counts)}")
    print(f"  Scene triples: avg={sum(scene_counts)/len(scene_counts):.1f}, "
          f"min={min(scene_counts)}, max={max(scene_counts)}")
    zero_scene = sum(1 for c in scene_counts if c == 0)
    print(f"  Zero scene:    {zero_scene}/{len(correction_results)}")

    # Correction stats
    total = len(correction_results)
    n_corrected = 0
    confidence_dist = {"high": 0, "medium": 0, "low": 0}
    type_correction_rate = {}

    for r, hallu_row, kb, corrected_cap, result_dict in correction_results:
        was_corrected = result_dict.get("was_corrected", False)
        if was_corrected:
            n_corrected += 1
        for e in result_dict.get("edits", []):
            conf = e.get("confidence")
            if conf:
                confidence_dist[conf] = confidence_dist.get(conf, 0) + 1
        if hallu_row is not None:
            htype = hallu_row["hallucination_type"]
            if htype not in type_correction_rate:
                type_correction_rate[htype] = {"total": 0, "corrected": 0}
            type_correction_rate[htype]["total"] += 1
            if was_corrected:
                type_correction_rate[htype]["corrected"] += 1

    print(f"\nCorrection Statistics:")
    print(f"  Total samples:  {total}")
    print(f"  Corrected:      {n_corrected} ({n_corrected/total*100:.1f}%)")
    print(f"  Passthrough:    {total - n_corrected}")

    print(f"\nConfidence distribution:")
    for conf, count in sorted(confidence_dist.items()):
        print(f"  {conf}: {count}")

    # Per-layer attribution: which KB layer caught how many hallucinations
    layer_dist = {}
    for _, _, _, _, result_dict in correction_results:
        for e in result_dict.get("edits", []):
            layer = e.get("contradicted_by")
            if layer:
                layer_dist[layer] = layer_dist.get(layer, 0) + 1

    print(f"\nEdits attributed by KB layer:")
    for layer, count in sorted(layer_dist.items(), key=lambda x: -x[1]):
        print(f"  {layer}: {count}")

    # Passthrough reason histogram
    passthrough_dist = {}
    for _, _, _, _, result_dict in correction_results:
        reason = result_dict.get("passthrough_reason")
        if reason:
            passthrough_dist[reason] = passthrough_dist.get(reason, 0) + 1

    if passthrough_dist:
        print(f"\nPassthrough reasons:")
        for reason, count in sorted(passthrough_dist.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")

    print(f"\nCorrection rate by hallucination type:")
    for htype, stats in sorted(type_correction_rate.items()):
        rate = stats["corrected"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {htype}: {stats['corrected']}/{stats['total']} ({rate:.0f}%)")

    # Edit distance distribution for corrected samples
    from Levenshtein import distance as lev_distance
    edit_dists = []
    for r, _, _, corrected_cap, _ in correction_results:
        if corrected_cap != r.ref_cap:
            edit_dists.append(lev_distance(r.ref_cap, corrected_cap))

    if edit_dists:
        print(f"\nEdit distance (corrected only, n={len(edit_dists)}):")
        print(f"  avg={sum(edit_dists)/len(edit_dists):.1f}, "
              f"min={min(edit_dists)}, max={max(edit_dists)}")

    # Timing breakdown
    s1 = [r.timings.stage1_seconds for r, _, _, _, _ in correction_results]
    s2 = [r.timings.stage2_seconds for r, _, _, _, _ in correction_results]
    s3 = [r.timings.stage3_seconds for r, _, _, _, _ in correction_results]
    s4 = [r.timings.stage4_seconds for r, _, _, _, _ in correction_results]

    print(f"\nClaim gen timing (avg per sample):")
    print(f"  S1 (concepts):   {sum(s1)/len(s1):.2f}s")
    print(f"  S2 (questions):  {sum(s2)/len(s2):.2f}s")
    print(f"  S3 (validation): {sum(s3)/len(s3):.2f}s")
    print(f"  S4 (claims):     {sum(s4)/len(s4):.2f}s")


# ── CELL 10 — Caption Editing Evaluation ─────────────────────
# Compute BLEU-1, BLEU-4, ROUGE-L, CIDEr, SPICE
# for three conditions: Ref-Cap (baseline), Corrected, and GT

from relcheck_v3.eval.models import CEPrediction, CaptionEditingScores
from relcheck_v3.eval.metrics import CaptionMetrics

# Build prediction lists for each condition
refcap_preds = []
corrected_preds = []

for r, hallu_row, kb, corrected_cap, _ in correction_results:
    gt_cap = hallu_row["gt_cap"] if hallu_row is not None else ""
    if not gt_cap:
        continue

    # Baseline: hallucinated caption as-is
    refcap_preds.append(CEPrediction(
        image_id=r.image_id,
        ref_cap=r.ref_cap,
        edited_cap=r.ref_cap,  # no editing
        gt_cap=gt_cap,
    ))

    # RelCheck: corrected caption
    corrected_preds.append(CEPrediction(
        image_id=r.image_id,
        ref_cap=r.ref_cap,
        edited_cap=corrected_cap,
        gt_cap=gt_cap,
    ))

if refcap_preds:
    print(f"Evaluating {len(refcap_preds)} samples...\n")

    refcap_scores = CaptionMetrics.compute(refcap_preds)
    corrected_scores = CaptionMetrics.compute(corrected_preds)

    # Display as table
    metrics = ["bleu_1", "bleu_4", "rouge_l", "cider", "spice"]
    labels = ["BLEU-1", "BLEU-4", "ROUGE-L", "CIDEr", "SPICE"]

    print(f"{'Metric':<12} {'Ref-Cap':>10} {'RelCheck':>10} {'Delta':>10}")
    print("-" * 44)
    for label, metric in zip(labels, metrics):
        ref_val = getattr(refcap_scores, metric)
        cor_val = getattr(corrected_scores, metric)
        delta = cor_val - ref_val
        sign = "+" if delta >= 0 else ""
        print(f"{label:<12} {ref_val:>10.2f} {cor_val:>10.2f} {sign}{delta:>9.2f}")

    # Save scores
    scores_path = f"{SAVE_DIR}/caption_editing_scores.json"
    with open(scores_path, "w") as f:
        json.dump({
            "ref_cap": refcap_scores.model_dump(),
            "relcheck": corrected_scores.model_dump(),
            "n_samples": len(refcap_preds),
        }, f, indent=2)
    print(f"\nSaved scores to {scores_path}")
else:
    print("No samples with GT captions available for evaluation.")


# ── CELL 11 — Side-by-Side Comparison ───────────────────────
# For corrected samples, show GT vs Hallucinated vs Corrected

corrected_examples = [
    (r, hallu_row, kb, cc, rd)
    for r, hallu_row, kb, cc, rd in correction_results
    if rd.get("was_corrected") and hallu_row is not None
]

print(f"Showing {min(10, len(corrected_examples))} corrected examples:\n")

for r, hallu_row, kb, corrected_cap, result_dict in corrected_examples[:10]:
    print(f"{'─'*60}")
    print(f"Image {r.image_id} | Type: {hallu_row['hallucination_type']}")
    print()
    print(f"  GT:        {hallu_row['gt_cap']}")
    print(f"  Hallu:     {r.ref_cap}")
    print(f"  Corrected: {corrected_cap}")
    for e in result_dict.get("edits", []):
        print(f"  Detected:  \"{e['original_span']}\" → "
              f"\"{e['replacement']}\" "
              f"[{e['contradicted_by']}, {e['confidence']}]")
    print()


# ── CELL 12 — Cost & Timing Summary ─────────────────────────
# Rough cost estimates based on OpenAI pricing

n_total = len(correction_results)

print(f"Pipeline Summary ({n_total} images)")
print(f"{'='*50}")

# Timing
print(f"\nTiming:")
print(f"  Claim gen (Stages 1-4): {claim_elapsed/60:.1f} min")
print(f"  KB building:            {kb_elapsed:.1f} s")
print(f"  Correction (Stage 5):   {correction_elapsed/60:.1f} min")
total_time = claim_elapsed + kb_elapsed + correction_elapsed
print(f"  Total:                  {total_time/60:.1f} min")
if n_total > 0:
    print(f"  Per image:              {total_time/n_total:.1f} s")

# Cost estimates (approximate)
# Claim gen (GPT-5.4-mini): ~$0.15/M input, $0.60/M output
# Hallu gen (GPT-4o-mini): ~$0.15/M input, $0.60/M output
# Correction (GPT-5.4): ~$2.50/M input, $15/M output
# Rough: ~200 input tokens + ~100 output tokens per Stage 5 call (x2 calls)
est_s5_input = n_total * 2 * 200 / 1_000_000  # M tokens
est_s5_output = n_total * 2 * 100 / 1_000_000
est_s5_cost = est_s5_input * 2.50 + est_s5_output * 15.0

# Claim gen: ~4 calls × ~150 tokens each
est_cg_input = n_total * 4 * 150 / 1_000_000
est_cg_output = n_total * 4 * 100 / 1_000_000
est_cg_cost = est_cg_input * 0.15 + est_cg_output * 0.60

print(f"\nEstimated API cost:")
print(f"  Claim gen (GPT-5.4-mini):  ~${est_cg_cost:.2f}")
print(f"  Correction (GPT-5.4):      ~${est_s5_cost:.2f}")
print(f"  Total:                     ~${est_cg_cost + est_s5_cost:.2f}")

print(f"\nAll outputs saved to: {SAVE_DIR}")
print(f"  full_pipeline_results.jsonl  — master JSONL (all stages)")
print(f"  full_pipeline_summary.csv    — summary table")
print(f"  caption_editing_scores.json  — BLEU/ROUGE/CIDEr/SPICE")
print(f"  correction_checkpoint.json   — resumable checkpoint")
