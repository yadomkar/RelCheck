# ============================================================
# RelCheck v3 — Claim Generation Pipeline (Woodpecker Stages 1–4)
# ============================================================
# Copy-paste each cell block into Google Colab.
# Cells are separated by: # ── CELL N ──
#
# Takes hallucinated captions from the hallucination generation
# pipeline and builds a Visual Knowledge Base (VKB) per sample
# using Woodpecker's 4-stage approach.
#
# Requires GPU (T4 or A100) for Grounding DINO + QA2Claim T5.


# ── CELL 0 — Config ─────────────────────────────────────────
OPENAI_API_KEY = ""               # <-- paste your OpenAI key
HF_TOKEN = ""                     # <-- paste your HuggingFace token (free, from hf.co/settings/tokens)
MAX_SAMPLES = 10                  # None = process all, set small for testing
CHECKPOINT_INTERVAL = 50          # Save checkpoint every N samples

# Input: hallucination generation output
HALLU_GEN_DIR = "/content/drive/MyDrive/RelCheck_Data/hallucination_gen"
HALLU_GEN_JSONL = f"{HALLU_GEN_DIR}/output.jsonl"

# Image zip on Drive (unzipped to local disk each session for speed)
COCO_ZIP = "/content/drive/MyDrive/RelCheck_Data/coco_zips/val2014.zip"
IMAGE_DIR = "/content/coco_val2014/val2014"

# Output directory for claim generation results
SAVE_DIR = "/content/drive/MyDrive/RelCheck_Data/claim_generation"


# ── CELL 1 — Setup ──────────────────────────────────────────
# !pip install openai>=1.0 pydantic>=2.0 tqdm pandas Pillow transformers torch -q
# !pip install groundingdino-py -q
# !python -m spacy download en_core_web_md -q

import os, sys, json
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

# Download GroundingDINO config + checkpoint (same as Woodpecker uses)
GDINO_DIR = "/content/groundingdino_weights"
GDINO_CONFIG = os.path.join(GDINO_DIR, "GroundingDINO_SwinT_OGC.py")
GDINO_CHECKPOINT = os.path.join(GDINO_DIR, "groundingdino_swint_ogc.pth")
os.makedirs(GDINO_DIR, exist_ok=True)

if not os.path.exists(GDINO_CONFIG):
    os.system(
        f"wget -q -O {GDINO_CONFIG} "
        "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    )
    print(f"Downloaded config: {GDINO_CONFIG}")

if not os.path.exists(GDINO_CHECKPOINT):
    os.system(
        f"wget -q -O {GDINO_CHECKPOINT} "
        "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
    )
    print(f"Downloaded checkpoint: {GDINO_CHECKPOINT}")

# Verify imports
from relcheck_v3.claim_generation.config import ClaimGenConfig
from relcheck_v3.claim_generation.models import InputSample
print("Setup complete.")

# Check GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Unzip COCO val2014 images to local disk (faster I/O than Drive)
import zipfile
print(f"\nUnzipping {COCO_ZIP} to {IMAGE_DIR}...")
os.makedirs(os.path.dirname(IMAGE_DIR), exist_ok=True)
with zipfile.ZipFile(COCO_ZIP, "r") as zf:
    zf.extractall(os.path.dirname(IMAGE_DIR))
print(f"Done — {len(os.listdir(IMAGE_DIR))} files in {IMAGE_DIR}")


# ── CELL 2 — Load Hallucination Generation Output ───────────
# Read the accepted hallucinated captions from the hallu gen pipeline
# and build InputSample objects for the claim generation pipeline.

import json

samples = []
skipped = 0

with open(HALLU_GEN_JSONL, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)

        # Only use accepted records (successfully generated hallucinations)
        if rec.get("status") != "accepted":
            skipped += 1
            continue

        image_id = str(rec["image_id"])
        ref_cap = rec["ref_cap"]

        # Build image path (COCO val2014 naming convention)
        img_path = os.path.join(IMAGE_DIR, f"COCO_val2014_{int(image_id):012d}.jpg")
        if not os.path.exists(img_path):
            skipped += 1
            continue

        samples.append(InputSample(
            image_id=image_id,
            image_path=img_path,
            ref_cap=ref_cap,
        ))

print(f"Loaded {len(samples)} accepted samples, skipped {skipped}")
if samples:
    print(f"\nFirst sample:")
    print(f"  image_id: {samples[0].image_id}")
    print(f"  ref_cap:  {samples[0].ref_cap[:80]}...")


# ── CELL 3 — Run Claim Generation Pipeline ──────────────────
import time
from relcheck_v3.claim_generation.pipeline import ClaimGenerationPipeline

config = ClaimGenConfig(
    openai_api_key=OPENAI_API_KEY,
    detector_config=GDINO_CONFIG,
    detector_model_path=GDINO_CHECKPOINT,
    output_dir=SAVE_DIR,
    checkpoint_interval=CHECKPOINT_INTERVAL,
    max_samples=MAX_SAMPLES,
)

print(f"Starting claim generation pipeline...")
print(f"  Samples:    {len(samples)}")
print(f"  Max:        {MAX_SAMPLES or 'all'}")
print(f"  Checkpoint: every {CHECKPOINT_INTERVAL}")
print(f"  Output:     {SAVE_DIR}")
print()

pipeline = ClaimGenerationPipeline(config)

t0 = time.time()
results = pipeline.process_batch(samples)
elapsed = time.time() - t0

successful = sum(1 for r in results if r.success)
failed = sum(1 for r in results if not r.success)
print(f"\nPipeline finished in {elapsed/60:.1f} minutes")
print(f"  Successful: {successful}")
print(f"  Failed:     {failed}")


# ── CELL 4 — Inspect Results ────────────────────────────────
import pandas as pd

csv_path = f"{SAVE_DIR}/output.csv"
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    print(f"Total records: {len(df)}")
    print(f"\nSuccess rate: {df['success'].mean():.1%}")
    print(f"\nSample results:")
    display(df.head(10))
else:
    print(f"No results found at {csv_path}")


# ── CELL 5 — View VKB Examples ──────────────────────────────
# Show Visual Knowledge Base output for a few samples
from PIL import Image
import matplotlib.pyplot as plt

successful_results = [r for r in results if r.success]
show_n = min(5, len(successful_results))

for r in successful_results[:show_n]:
    print(f"{'='*60}")
    print(f"Image ID:     {r.image_id}")
    print(f"Ref-Cap:      {r.ref_cap}")
    print(f"Key Concepts: {r.key_concepts}")
    print(f"\nVisual Knowledge Base:")
    print(r.vkb_text)
    print(f"\nTimings: S1={r.timings.stage1_seconds:.2f}s "
          f"S2={r.timings.stage2_seconds:.2f}s "
          f"S3={r.timings.stage3_seconds:.2f}s "
          f"S4={r.timings.stage4_seconds:.2f}s "
          f"Total={r.timings.total_seconds:.2f}s")

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


# ── CELL 6 — Timing Analysis ────────────────────────────────
# Analyze per-stage timing breakdown
if successful_results:
    s1 = [r.timings.stage1_seconds for r in successful_results]
    s2 = [r.timings.stage2_seconds for r in successful_results]
    s3 = [r.timings.stage3_seconds for r in successful_results]
    s4 = [r.timings.stage4_seconds for r in successful_results]
    total = [r.timings.total_seconds for r in successful_results]

    print(f"Timing breakdown ({len(successful_results)} samples):")
    print(f"  Stage 1 (Concept Extraction): {sum(s1)/len(s1):.2f}s avg")
    print(f"  Stage 2 (Question Formulation): {sum(s2)/len(s2):.2f}s avg")
    print(f"  Stage 3 (Visual Validation):  {sum(s3)/len(s3):.2f}s avg")
    print(f"  Stage 4 (Claim Generation):   {sum(s4)/len(s4):.2f}s avg")
    print(f"  Total:                        {sum(total)/len(total):.2f}s avg")
    print(f"  Total wall time:              {sum(total):.1f}s")
