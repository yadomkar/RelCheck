# ============================================================
# RelCheck v3 — Synthetic Hallucination Generation (Kim et al. Task 2)
# ============================================================
# Copy-paste each cell block into Google Colab.
# Cells are separated by: # ── CELL N ──
#
# Generates hallucinated captions using GPT-4o-mini following
# Kim et al.'s Context-Aware Caption Editing methodology.
# All pipeline logic lives in relcheck_v3.hallucination_generation.


# ── CELL 0 — Config ─────────────────────────────────────────
DATASET_NAME = "coco-ee"          # "coco-ee" or "flickr30k-ee"
OPENAI_API_KEY = ""               # <-- paste your OpenAI key
MAX_SAMPLES = None                # None = process all, set to e.g. 50 for testing
DRY_RUN = False                   # True = skip API calls, use placeholders
SAVE_DIR = "/content/drive/MyDrive/RelCheck_Data/hallucination_gen"

# Dataset paths (set after download in Cell 2)
ANNOTATION_PATH = ""
IMAGE_DIR = ""


# ── CELL 1 — Setup ──────────────────────────────────────────
# !pip install openai>=1.0 pydantic>=2.0 python-Levenshtein tqdm pandas Pillow -q

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

# Verify import
from relcheck_v3.hallucination_generation.config import PipelineConfig
print("Setup complete.")


# ── CELL 2 — Download COCO-EE Data ──────────────────────────
# COCO-EE uses COCO val2014 images + custom annotation JSON.
# Adjust paths if you already have the data on Drive.

COCO_DIR = "/content/coco_val2014"
COCO_EE_ANN = f"{SAVE_DIR}/coco_ee_annotations.json"

# Download COCO val2014 images if not present
if not os.path.exists(f"{COCO_DIR}/COCO_val2014_000000000042.jpg"):
    os.makedirs(COCO_DIR, exist_ok=True)
    print("Downloading COCO val2014 images (~6GB)...")
    os.system(f"wget -q http://images.cocodataset.org/zips/val2014.zip -O /content/val2014.zip")
    os.system(f"unzip -q /content/val2014.zip -d /content/")
    # Images land in /content/val2014/ — move to our dir
    os.system(f"mv /content/val2014/* {COCO_DIR}/")
    os.system("rm /content/val2014.zip")
    print("COCO val2014 images ready.")
else:
    print(f"COCO val2014 images already at {COCO_DIR}")

# Download COCO captions for annotation extraction
COCO_CAPS_PATH = f"{COCO_DIR}/captions_val2014.json"
if not os.path.exists(COCO_CAPS_PATH):
    print("Downloading COCO captions annotation...")
    os.system(
        f"wget -q http://images.cocodataset.org/annotations/annotations_trainval2014.zip "
        f"-O /content/annotations2014.zip"
    )
    os.system("unzip -q /content/annotations2014.zip -d /content/annotations_tmp/")
    os.system(f"cp /content/annotations_tmp/annotations/captions_val2014.json {COCO_CAPS_PATH}")
    os.system("rm -rf /content/annotations_tmp /content/annotations2014.zip")

# Build COCO-EE annotation file (list of {image_id, caption} dicts)
if not os.path.exists(COCO_EE_ANN):
    print("Building COCO-EE annotation file...")
    with open(COCO_CAPS_PATH) as f:
        coco_data = json.load(f)

    # Take one caption per image (first occurrence)
    seen = set()
    annotations = []
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in seen:
            seen.add(img_id)
            annotations.append({"image_id": img_id, "caption": ann["caption"]})

    with open(COCO_EE_ANN, "w") as f:
        json.dump(annotations, f)
    print(f"Created {COCO_EE_ANN} with {len(annotations)} image-caption pairs")
else:
    with open(COCO_EE_ANN) as f:
        annotations = json.load(f)
    print(f"Loaded {len(annotations)} annotations from {COCO_EE_ANN}")

# Set paths for pipeline
ANNOTATION_PATH = COCO_EE_ANN
IMAGE_DIR = COCO_DIR
print(f"Annotation: {ANNOTATION_PATH}")
print(f"Image dir:  {IMAGE_DIR}")


# ── CELL 3 — Run Pipeline ───────────────────────────────────
import time
from relcheck_v3.hallucination_generation.run import main

print(f"Starting hallucination generation pipeline...")
print(f"  Dataset:     {DATASET_NAME}")
print(f"  Max samples: {MAX_SAMPLES or 'all'}")
print(f"  Dry run:     {DRY_RUN}")
print(f"  Output:      {SAVE_DIR}")
print()

t0 = time.time()
main(
    dataset_name=DATASET_NAME,
    annotation_path=ANNOTATION_PATH,
    image_dir=IMAGE_DIR,
    openai_api_key=OPENAI_API_KEY,
    output_dir=SAVE_DIR,
    max_samples=MAX_SAMPLES,
    dry_run=DRY_RUN,
)
elapsed = time.time() - t0
print(f"\nPipeline finished in {elapsed/60:.1f} minutes")


# ── CELL 4 — Inspect Results ────────────────────────────────
import pandas as pd

# Load results
csv_path = f"{SAVE_DIR}/output.csv"
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    print(f"Total records: {len(df)}")
    print(f"\nStatus distribution:")
    print(df["status"].value_counts().to_string())
    print(f"\nHallucination type distribution:")
    print(df["hallucination_type"].value_counts().to_string())
    print(f"\nEdit distance stats (accepted only):")
    accepted = df[df["status"] == "accepted"]
    if len(accepted) > 0:
        print(f"  Mean:   {accepted['edit_distance'].mean():.1f}")
        print(f"  Median: {accepted['edit_distance'].median():.1f}")
        print(f"  Min:    {accepted['edit_distance'].min()}")
        print(f"  Max:    {accepted['edit_distance'].max()}")
    print(f"\nSample accepted records:")
    display(accepted.head(10))
else:
    print(f"No results found at {csv_path}")


# ── CELL 5 — View Summary Stats ─────────────────────────────
summary_path = f"{SAVE_DIR}/summary_stats.json"
if os.path.exists(summary_path):
    with open(summary_path) as f:
        stats = json.load(f)
    print(json.dumps(stats, indent=2))
else:
    print("No summary stats found. Run the pipeline first.")


# ── CELL 6 — Sample Hallucinations ──────────────────────────
# Show a few examples of generated hallucinations
from IPython.display import display, HTML
from PIL import Image
import matplotlib.pyplot as plt

if os.path.exists(csv_path):
    samples = accepted.sample(min(5, len(accepted)))
    for _, row in samples.iterrows():
        print(f"{'='*60}")
        print(f"Image ID: {row['image_id']}")
        print(f"Type:     {row['hallucination_type']}")
        print(f"GT-Cap:   {row['gt_cap']}")
        print(f"Ref-Cap:  {row['ref_cap']}")
        print(f"Reason:   {row['reason']}")
        print(f"Edit Dist: {row['edit_distance']}")

        # Try to show the image
        img_id = str(row["image_id"])
        if DATASET_NAME == "coco-ee":
            img_path = f"{IMAGE_DIR}/COCO_val2014_{int(img_id):012d}.jpg"
        else:
            img_path = f"{IMAGE_DIR}/{img_id}.jpg"

        if os.path.exists(img_path):
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            ax.imshow(Image.open(img_path))
            ax.set_title(f"ID: {img_id} | {row['hallucination_type']}")
            ax.axis("off")
            plt.tight_layout()
            plt.show()
        print()
