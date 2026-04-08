# ============================================================
# RelCheck v3 — Caption Evaluation Pipeline (Kim et al. Tables 2 & 3)
# ============================================================
# Copy-paste each cell block into Google Colab.
# Cells are separated by: # %%
#
# Runs caption editing quality (Table 2) and POPE hallucination
# robustness (Table 3) evaluation using LLaVA-1.5-7B, mPLUG-Owl2-7B,
# and the passthrough (Ref-Caps) baseline.
#
# All pipeline logic lives in relcheck_v3.eval.
#
# Requirements: 5.1, 6.1, 9.1, 10.1, 11.1, 11.2


# %% [markdown]
# # Caption Evaluation Pipeline — Kim et al. Tables 2 & 3
#
# This notebook runs the full evaluation pipeline from Kim et al.'s
# ICCV 2025 Workshop paper. It covers:
#
# 1. **Caption Editing Quality (Table 2)** — BLEU-1, BLEU-4, ROUGE-L,
#    CIDEr, SPICE on COCO-CE and Flickr30K-CE test sets.
# 2. **POPE Hallucination Robustness (Table 3)** — Accuracy and F1
#    across 3 domains × 3 settings.
#
# Models: Passthrough (Ref-Caps baseline), LLaVA-1.5-7B, mPLUG-Owl2-7B.


# %% [markdown]
# ## Cell 1 — Setup & Installation

# %%
# ── CELL 1 — Setup & Installation ───────────────────────────
# Install core dependencies and clone the RelCheck repo + mPLUG-Owl2.

import os
import sys

# --- Core Python dependencies ---
# !pip install pydantic>=2.0 tqdm pandas Pillow -q
# !pip install pycocoevalcap -q
# !pip install transformers accelerate bitsandbytes -q

# --- Clone RelCheck repo ---
REPO_DIR = "/content/RelCheck"
if not os.path.exists(os.path.join(REPO_DIR, ".git")):
    os.system(f"git clone https://github.com/yadomkar/RelCheck.git {REPO_DIR}")
else:
    os.system(f"cd {REPO_DIR} && git pull -q")
sys.path.insert(0, REPO_DIR)

# --- Install mPLUG-Owl2 from custom repo (Req 6.1) ---
# mPLUG-Owl2 requires a local package install from the official repo.
MPLUG_REPO = "/content/mPLUG-Owl"
MPLUG_PKG = os.path.join(MPLUG_REPO, "mPLUG-Owl2")

if not os.path.exists(os.path.join(MPLUG_REPO, ".git")):
    os.system(f"git clone https://github.com/X-PLUG/mPLUG-Owl.git {MPLUG_REPO}")
else:
    os.system(f"cd {MPLUG_REPO} && git pull -q")

# Install the mPLUG-Owl2 package in editable mode
os.system(f"pip install -e {MPLUG_PKG} -q")

# Verify imports
from relcheck_v3.eval.config import EvalConfig
from relcheck_v3.eval.models import EvalType
print("✓ relcheck_v3.eval imported successfully")

try:
    import mplug_owl2  # noqa: F401
    print("✓ mPLUG-Owl2 package installed")
except ImportError:
    print("⚠ mPLUG-Owl2 not installed — mPLUG-Owl2 evaluation will be skipped")

print("\nSetup complete.")


# %% [markdown]
# ## Cell 2 — Mount Google Drive

# %%
# ── CELL 2 — Mount Google Drive ─────────────────────────────
# Mount Drive for model caching and data access.
# Models are cached here to avoid re-downloading across sessions.

from google.colab import drive

drive.mount("/content/drive")

# Create output directory on Drive for persistent results
OUTPUT_DIR = "/content/drive/MyDrive/RelCheck_Data/eval_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}")


# %% [markdown]
# ## Cell 3 — Configure Paths

# %%
# ── CELL 3 — Configure Paths ────────────────────────────────
# Set all paths for test sets, POPE data, image directories, and output.
# Update these to match your Google Drive layout.

# --- Caption Editing test sets (Table 2) ---
# These are the COCO-CE (5,366 samples) and Flickr30K-CE (4,910 samples)
# test set JSONs produced by the hallucination generation pipeline.
COCO_CE_PATH = "/content/drive/MyDrive/RelCheck_Data/eval/coco_ce_test.json"
FLICKR_CE_PATH = "/content/drive/MyDrive/RelCheck_Data/eval/flickr30k_ce_test.json"

# --- POPE benchmark data (Table 3) ---
# Directory containing POPE question files for all 9 domain×setting combos.
# Expected files: coco_pope_{random,popular,adversarial}.json, etc.
POPE_DATA_DIR = "/content/drive/MyDrive/RelCheck_Data/eval/pope"

# --- Image directories (one per domain) ---
COCO_IMAGE_DIR = "/content/drive/MyDrive/RelCheck_Data/coco_zips/val2014"
FLICKR_IMAGE_DIR = "/content/drive/MyDrive/RelCheck_Data/flickr30k/images"
AOKVQA_IMAGE_DIR = "/content/drive/MyDrive/RelCheck_Data/coco_zips/val2014"  # AOKVQA uses COCO images
GQA_IMAGE_DIR = "/content/drive/MyDrive/RelCheck_Data/gqa/images"

# --- Processing limits ---
# Set to None for full evaluation, or a small number for quick testing.
MAX_SAMPLES = 50  # <-- Set to None for full run

# --- Checkpoint interval ---
CHECKPOINT_INTERVAL = 500

print("Path configuration:")
print(f"  COCO-CE:        {COCO_CE_PATH}")
print(f"  Flickr30K-CE:   {FLICKR_CE_PATH}")
print(f"  POPE data:      {POPE_DATA_DIR}")
print(f"  COCO images:    {COCO_IMAGE_DIR}")
print(f"  Flickr images:  {FLICKR_IMAGE_DIR}")
print(f"  AOKVQA images:  {AOKVQA_IMAGE_DIR}")
print(f"  GQA images:     {GQA_IMAGE_DIR}")
print(f"  Output:         {OUTPUT_DIR}")
print(f"  Max samples:    {MAX_SAMPLES or 'all'}")


# %% [markdown]
# ## Cell 4 — Run Passthrough Baseline (Ref-Caps)

# %%
# ── CELL 4 — Run Passthrough Baseline (Ref-Caps) ────────────
# Quick sanity check: the passthrough editor returns Ref-Cap unchanged.
# This computes the "Ref-Caps" baseline row in Table 2.
# No GPU needed — runs instantly.

import time
from relcheck_v3.eval.__main__ import main
from relcheck_v3.eval.config import EvalConfig
from relcheck_v3.eval.models import EvalType

passthrough_config = EvalConfig(
    model_name="passthrough",
    eval_type=EvalType.CAPTION_EDITING,
    coco_ce_path=COCO_CE_PATH,
    flickr_ce_path=FLICKR_CE_PATH,
    coco_image_dir=COCO_IMAGE_DIR,
    flickr_image_dir=FLICKR_IMAGE_DIR,
    output_dir=OUTPUT_DIR,
    max_samples=MAX_SAMPLES,
    checkpoint_interval=CHECKPOINT_INTERVAL,
)

print("Running passthrough (Ref-Caps) baseline...")
print("This returns Ref-Cap unchanged — used to compute baseline scores.\n")

t0 = time.time()
passthrough_results = main(config=passthrough_config)
elapsed = time.time() - t0

print(f"\n{'='*60}")
print(f"Passthrough baseline finished in {elapsed:.1f}s")
if "caption_editing" in passthrough_results:
    for test_set, scores in passthrough_results["caption_editing"].items():
        print(f"\n  {test_set}:")
        print(f"    B-1={scores.bleu_1:.1f}  B-4={scores.bleu_4:.1f}  "
              f"R={scores.rouge_l:.1f}  C={scores.cider:.1f}  S={scores.spice:.1f}")


# %% [markdown]
# ## Cell 5 — Run LLaVA-1.5 Evaluation

# %%
# ── CELL 5 — Run LLaVA-1.5 Evaluation ──────────────────────
# Full evaluation with LLaVA-1.5-7B (8-bit quantized).
# Runs both Caption Editing (Table 2) and POPE (Table 3).
# Requires GPU — loads ~8GB in 8-bit quantization.
# (Req 5.1, 9.1, 10.1)

import time
from relcheck_v3.eval.__main__ import main
from relcheck_v3.eval.config import EvalConfig
from relcheck_v3.eval.models import EvalType

llava_config = EvalConfig(
    model_name="llava-1.5",
    eval_type=EvalType.BOTH,
    coco_ce_path=COCO_CE_PATH,
    flickr_ce_path=FLICKR_CE_PATH,
    pope_data_dir=POPE_DATA_DIR,
    coco_image_dir=COCO_IMAGE_DIR,
    flickr_image_dir=FLICKR_IMAGE_DIR,
    aokvqa_image_dir=AOKVQA_IMAGE_DIR,
    gqa_image_dir=GQA_IMAGE_DIR,
    output_dir=OUTPUT_DIR,
    max_samples=MAX_SAMPLES,
    checkpoint_interval=CHECKPOINT_INTERVAL,
)

print("Running LLaVA-1.5-7B evaluation (Caption Editing + POPE)...")
print("This will load the model in 8-bit quantization on GPU.\n")

t0 = time.time()
llava_results = main(config=llava_config)
elapsed = time.time() - t0

print(f"\n{'='*60}")
print(f"LLaVA-1.5 evaluation finished in {elapsed/60:.1f} minutes")

if "caption_editing" in llava_results:
    for test_set, scores in llava_results["caption_editing"].items():
        print(f"\n  Caption Editing — {test_set}:")
        print(f"    B-1={scores.bleu_1:.1f}  B-4={scores.bleu_4:.1f}  "
              f"R={scores.rouge_l:.1f}  C={scores.cider:.1f}  S={scores.spice:.1f}")

if "pope" in llava_results:
    print("\n  POPE:")
    for combo_key, scores in llava_results["pope"].items():
        print(f"    {combo_key}: Acc={scores.accuracy:.1f}  F1={scores.f1:.1f}")


# %% [markdown]
# ## Cell 6 — Run mPLUG-Owl2 Evaluation

# %%
# ── CELL 6 — Run mPLUG-Owl2 Evaluation ─────────────────────
# Full evaluation with mPLUG-Owl2-7B (8-bit quantized).
# Runs both Caption Editing (Table 2) and POPE (Table 3).
# Requires GPU + mPLUG-Owl2 package installed (Cell 1).
# (Req 6.1, 9.1, 10.1)

import time
import torch
from relcheck_v3.eval.__main__ import main
from relcheck_v3.eval.config import EvalConfig
from relcheck_v3.eval.models import EvalType

# Free GPU memory from previous model before loading mPLUG-Owl2
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"GPU memory freed. Available: {torch.cuda.mem_get_info()[0]/1024**2:.0f} MB")

mplug_config = EvalConfig(
    model_name="mplug-owl2",
    eval_type=EvalType.BOTH,
    coco_ce_path=COCO_CE_PATH,
    flickr_ce_path=FLICKR_CE_PATH,
    pope_data_dir=POPE_DATA_DIR,
    coco_image_dir=COCO_IMAGE_DIR,
    flickr_image_dir=FLICKR_IMAGE_DIR,
    aokvqa_image_dir=AOKVQA_IMAGE_DIR,
    gqa_image_dir=GQA_IMAGE_DIR,
    output_dir=OUTPUT_DIR,
    max_samples=MAX_SAMPLES,
    checkpoint_interval=CHECKPOINT_INTERVAL,
)

print("Running mPLUG-Owl2-7B evaluation (Caption Editing + POPE)...")
print("This will load the model in 8-bit quantization on GPU.\n")

t0 = time.time()
mplug_results = main(config=mplug_config)
elapsed = time.time() - t0

print(f"\n{'='*60}")
print(f"mPLUG-Owl2 evaluation finished in {elapsed/60:.1f} minutes")

if "caption_editing" in mplug_results:
    for test_set, scores in mplug_results["caption_editing"].items():
        print(f"\n  Caption Editing — {test_set}:")
        print(f"    B-1={scores.bleu_1:.1f}  B-4={scores.bleu_4:.1f}  "
              f"R={scores.rouge_l:.1f}  C={scores.cider:.1f}  S={scores.spice:.1f}")

if "pope" in mplug_results:
    print("\n  POPE:")
    for combo_key, scores in mplug_results["pope"].items():
        print(f"    {combo_key}: Acc={scores.accuracy:.1f}  F1={scores.f1:.1f}")


# %% [markdown]
# ## Cell 7 — Display Results

# %%
# ── CELL 7 — Display Results ────────────────────────────────
# Load and display the exported Table 2 and Table 3 results.
# Shows computed values alongside the paper's reported baselines.
# (Req 11.1, 11.2)

import json
import pandas as pd

print("=" * 70)
print("TABLE 2 — Caption Editing Quality")
print("=" * 70)

# Display Table 2 for each test set
for test_set_tag in ["coco_ce", "flickr30k_ce"]:
    csv_path = os.path.join(OUTPUT_DIR, f"table2_{test_set_tag}.csv")
    txt_path = os.path.join(OUTPUT_DIR, f"table2_{test_set_tag}.txt")

    if os.path.exists(txt_path):
        print(f"\n{open(txt_path).read()}")
    elif os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"\n--- {test_set_tag.upper().replace('_', '-')} ---")
        display(df)  # noqa: F821 — Colab built-in
    else:
        print(f"\n  No Table 2 results found for {test_set_tag}")
        print(f"  Expected at: {csv_path}")

print("\n" + "=" * 70)
print("TABLE 3 — POPE Hallucination Robustness")
print("=" * 70)

txt_path = os.path.join(OUTPUT_DIR, "table3_pope.txt")
csv_path = os.path.join(OUTPUT_DIR, "table3_pope.csv")

if os.path.exists(txt_path):
    print(f"\n{open(txt_path).read()}")
elif os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    display(df)  # noqa: F821 — Colab built-in
else:
    print("\n  No Table 3 results found.")
    print(f"  Expected at: {csv_path}")

# Display aggregate JSON summary
agg_path = os.path.join(OUTPUT_DIR, "aggregate_scores.json")
if os.path.exists(agg_path):
    print("\n" + "=" * 70)
    print("AGGREGATE SCORES (JSON)")
    print("=" * 70)
    with open(agg_path) as f:
        agg = json.load(f)
    print(json.dumps(agg, indent=2))
