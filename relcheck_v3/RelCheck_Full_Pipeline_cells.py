# ============================================================
# RelCheck v3 — Full Pipeline: Hallu Gen → Claim Gen → KB Build
# ============================================================
# Copy-paste each cell block into Google Colab.
# Cells are separated by: # ── CELL N ──
#
# End-to-end: generates hallucinated captions, builds claims,
# computes geometry, runs RelTR scene graph, and displays the
# full 3-layer Knowledge Base.
#
# Images are pre-filtered to those whose COCO categories overlap
# with RelTR's Visual Genome vocabulary.
#
# Requires GPU (T4 or A100).


# ── CELL 0 — Config ─────────────────────────────────────────
OPENAI_API_KEY = ""               # <-- paste your OpenAI key
HF_TOKEN = ""                     # <-- paste your HuggingFace token
MAX_SAMPLES = 25                  # number of images to process

# Paths on Drive
DRIVE_BASE = "/content/drive/MyDrive/RelCheck_Data"
COCO_ZIP = f"{DRIVE_BASE}/coco_zips/val2014.zip"
COCO_CAPS_PATH = f"{DRIVE_BASE}/coco_zips/annotations/captions_val2014.json"
COCO_INSTANCES_PATH = f"{DRIVE_BASE}/coco_zips/annotations/instances_val2014.json"
IMAGE_DIR = "/content/coco_val2014/val2014"

# Output
SAVE_DIR = f"{DRIVE_BASE}/full_pipeline"

# GroundingDINO
GDINO_DIR = "/content/groundingdino_weights"
GDINO_CHECKPOINT = f"{GDINO_DIR}/groundingdino_swint_ogc.pth"

# RelTR
ENABLE_RELTR = True
RELTR_CHECKPOINT = f"{DRIVE_BASE}/checkpoint0149.pth"


# ── CELL 1 — Setup ──────────────────────────────────────────
# !pip install openai>=1.0 pydantic>=2.0 python-Levenshtein tqdm pandas Pillow transformers torch -q
# !pip install groundingdino-py -q
# !python -m spacy download en_core_web_md -q

import os, sys, json, zipfile
from google.colab import drive

drive.mount("/content/drive")
os.makedirs(SAVE_DIR, exist_ok=True)

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
        "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
    )
print(f"GroundingDINO checkpoint: {os.path.exists(GDINO_CHECKPOINT)}")

import groundingdino
GDINO_CONFIG = os.path.join(os.path.dirname(groundingdino.__file__), "config", "GroundingDINO_SwinT_OGC.py")

# RelTR setup
if ENABLE_RELTR:
    RELTR_DIR = "/content/RelTR"
    if not os.path.exists(RELTR_DIR):
        os.system(f"git clone https://github.com/yrcong/RelTR.git {RELTR_DIR}")

import relcheck_v3.reltr.config as reltr_cfg
reltr_cfg.ENABLE_RELTR = ENABLE_RELTR
reltr_cfg.RELTR_CHECKPOINT_PATH = RELTR_CHECKPOINT
print(f"ENABLE_RELTR = {reltr_cfg.ENABLE_RELTR}")

# Unzip COCO images
if not os.path.exists(IMAGE_DIR) or len(os.listdir(IMAGE_DIR)) < 100:
    print(f"Unzipping {COCO_ZIP}...")
    os.makedirs(os.path.dirname(IMAGE_DIR), exist_ok=True)
    with zipfile.ZipFile(COCO_ZIP, "r") as zf:
        zf.extractall(os.path.dirname(IMAGE_DIR))
print(f"Images: {len(os.listdir(IMAGE_DIR))} files")

import torch
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("Setup complete.")


# ── CELL 2 — Filter images by RelTR vocab coverage ──────────
# Only keep COCO images whose annotated categories overlap with
# RelTR's Visual Genome object vocabulary.

from relcheck_v3.reltr.reltr import RELTR_OBJECT_CLASSES

# Build a set of RelTR-compatible COCO category names
# Only map COCO categories that have a genuine RelTR equivalent.
# 38 COCO categories have exact matches; a few more have clear synonyms.
COCO_TO_RELTR = {
    # Exact matches (38)
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
    # Clear synonyms
    "bicycle": "bike",
    "skis": "ski",
    "tennis racket": "racket",
    "wine glass": "glass",
    "baseball glove": "glove",
    "dining table": "table",
    "potted plant": "plant",
    "tv": "screen",
    "couch": "seat",
    "backpack": "bag",
    "handbag": "bag",
    "suitcase": "bag",
    "cell phone": "phone",
}

RELTR_SET = set(RELTR_OBJECT_CLASSES)

# Load COCO instance annotations to get category info per image
if not os.path.exists(COCO_INSTANCES_PATH):
    raise FileNotFoundError(
        f"COCO instances annotation not found at {COCO_INSTANCES_PATH}\n"
        "Download from: http://images.cocodataset.org/annotations/annotations_trainval2014.zip\n"
        "Unzip and place instances_val2014.json in the annotations folder on Drive."
    )

with open(COCO_INSTANCES_PATH) as f:
    instances = json.load(f)

cat_id_to_name = {c["id"]: c["name"] for c in instances["categories"]}

# Build image_id → set of COCO category names
image_categories: dict[int, set[str]] = {}
for ann in instances["annotations"]:
    img_id = ann["image_id"]
    cat_name = cat_id_to_name[ann["category_id"]]
    image_categories.setdefault(img_id, set()).add(cat_name)

# Filter: keep images where at least one COCO category maps to a RelTR class
reltr_compatible_ids: set[int] = set()
for img_id, cats in image_categories.items():
    mapped = {COCO_TO_RELTR.get(c) for c in cats} - {None}
    if mapped & RELTR_SET:
        reltr_compatible_ids.add(img_id)

print(f"Total COCO images with annotations: {len(image_categories)}")
print(f"RelTR-compatible images: {len(reltr_compatible_ids)}")

# Load captions and filter to RelTR-compatible images
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
            filtered_annotations.append({"image_id": img_id, "caption": ann["caption"]})

print(f"Filtered annotations (1 per image, RelTR-compatible): {len(filtered_annotations)}")

# Save filtered annotations
FILTERED_ANN_PATH = f"{SAVE_DIR}/filtered_annotations.json"
with open(FILTERED_ANN_PATH, "w") as f:
    json.dump(filtered_annotations, f)


# ── CELL 3 — Run Hallucination Generation ────────────────────
import time
from relcheck_v3.hallucination_generation.run import main as hallu_main

HALLU_SAVE_DIR = f"{SAVE_DIR}/hallucination_gen"
os.makedirs(HALLU_SAVE_DIR, exist_ok=True)

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

# Load accepted results
import pandas as pd
hallu_csv = f"{HALLU_SAVE_DIR}/output.csv"
hallu_df = pd.read_csv(hallu_csv)
accepted = hallu_df[hallu_df["status"] == "accepted"]
print(f"Accepted: {len(accepted)} / {len(hallu_df)}")
print(f"\nType distribution:")
print(accepted["hallucination_type"].value_counts().to_string())


# ── CELL 4 — Run Claim Generation ────────────────────────────
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
print(f"Claim gen done in {(time.time()-t0)/60:.1f} min")

successful_claims = [r for r in claim_results if r.success]
print(f"Successful: {len(successful_claims)} / {len(claim_results)}")


# ── CELL 5 — Build Knowledge Bases ──────────────────────────
from PIL import Image
from tqdm import tqdm
from relcheck_v3.kb import build_kb, KnowledgeBase

kb_results: list[tuple] = []  # (claim_result, hallu_row, kb)
errors = 0

# Build a lookup from image_id to hallu row for display
hallu_lookup = {}
for _, row in accepted.iterrows():
    hallu_lookup[str(row["image_id"])] = row

t0 = time.time()
for r in tqdm(successful_claims, desc="Building KBs"):
    try:
        img_path = os.path.join(IMAGE_DIR, f"COCO_val2014_{int(r.image_id):012d}.jpg")
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
        errors += 1
        print(f"[{r.image_id}] KB build failed: {e}")

print(f"Built {len(kb_results)} KBs in {time.time()-t0:.1f}s ({errors} errors)")


# ── CELL 6 — Save Results ───────────────────────────────────
# Save full KB output as JSONL
kb_jsonl_path = f"{SAVE_DIR}/kb_output.jsonl"
with open(kb_jsonl_path, "w") as f:
    for r, hallu_row, kb in kb_results:
        rec = {
            "image_id": r.image_id,
            "gt_cap": hallu_row["gt_cap"] if hallu_row is not None else "",
            "ref_cap": r.ref_cap,
            "hallucination_type": hallu_row["hallucination_type"] if hallu_row is not None else "",
            "reason": hallu_row["reason"] if hallu_row is not None else "",
            "kb_text": kb.format(),
            "n_spatial_facts": len(kb.spatial_facts),
            "n_scene_triples": len(kb.scene_graph),
        }
        f.write(json.dumps(rec) + "\n")
print(f"Saved {len(kb_results)} records to {kb_jsonl_path}")


# ── CELL 7 — Display Results ────────────────────────────────
import matplotlib.pyplot as plt

show_n = min(10, len(kb_results))

for r, hallu_row, kb in kb_results[:show_n]:
    print(f"{'='*70}")
    print(f"Image ID:          {r.image_id}")
    if hallu_row is not None:
        print(f"GT Caption:        {hallu_row['gt_cap']}")
        print(f"Hallucinated Cap:  {hallu_row['ref_cap']}")
        print(f"Hallu Type:        {hallu_row['hallucination_type']}")
        print(f"Reason:            {hallu_row['reason']}")
    else:
        print(f"Ref Caption:       {r.ref_cap}")
    print()
    print(kb.format())

    # Show image
    img_path = os.path.join(IMAGE_DIR, f"COCO_val2014_{int(r.image_id):012d}.jpg")
    if os.path.exists(img_path):
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.imshow(Image.open(img_path))
        ax.set_title(f"ID: {r.image_id}")
        ax.axis("off")
        plt.tight_layout()
        plt.show()
    print()


# ── CELL 8 — Statistics ─────────────────────────────────────
if kb_results:
    spatial_counts = [len(kb.spatial_facts) for _, _, kb in kb_results]
    scene_counts = [len(kb.scene_graph) for _, _, kb in kb_results]
    claim_counts = [
        len(kb.claims.count_claims) + len(kb.claims.specific_claims) + len(kb.claims.overall_claims)
        for _, _, kb in kb_results
    ]

    print(f"KB Statistics ({len(kb_results)} samples):")
    print(f"  Claims:         avg={sum(claim_counts)/len(claim_counts):.1f}, "
          f"min={min(claim_counts)}, max={max(claim_counts)}")
    print(f"  Spatial facts:  avg={sum(spatial_counts)/len(spatial_counts):.1f}, "
          f"min={min(spatial_counts)}, max={max(spatial_counts)}")
    print(f"  Scene triples:  avg={sum(scene_counts)/len(scene_counts):.1f}, "
          f"min={min(scene_counts)}, max={max(scene_counts)}")

    # Images with zero scene triples (RelTR found nothing)
    zero_scene = sum(1 for c in scene_counts if c == 0)
    print(f"  Zero scene triples: {zero_scene}/{len(kb_results)}")
