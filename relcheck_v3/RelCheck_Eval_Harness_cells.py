# ============================================================
# RelCheck v3 — Multi-Benchmark Evaluation Harness
# ============================================================
# Copy-paste each cell block into Google Colab.
# Cells are separated by: # %%
#
# Runs the evaluation harness across POPE, MME, and AMBER
# benchmarks with five correction system variants:
#   RawMLLM, Woodpecker, RelCheck-Claim, RelCheck-Claim+Geom,
#   RelCheck-Full.
#
# Produces a thesis-ready master results table with ablation
# deltas and stratified reporting by RelTR tag.
#
# Requires GPU (T4 or A100).
#
# Requirements: 1.1, 1.2, 1.3, 1.6
from __future__ import annotations

# %% [markdown]
# # RelCheck v3 — Multi-Benchmark Evaluation Harness
#
# This notebook runs the full evaluation matrix:
# 5 correction systems × 3 benchmarks = 15 runs.
#
# **Systems**: RawMLLM, Woodpecker, RelCheck-Claim,
# RelCheck-Claim+Geom, RelCheck-Full
#
# **Benchmarks**: POPE, MME, AMBER

# %%
# ── Cell 1: Setup ───────────────────────────────────────────
# Install dependencies, mount Google Drive, clone the repo.

import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
_log = logging.getLogger("eval_harness_cells")

# Install required packages (uncomment in Colab)
# !pip install -q openai>=1.0 pydantic>=2.0 tqdm pandas scikit-learn \
#     transformers torch accelerate bitsandbytes Pillow tenacity \
#     groundingdino-py tabulate
# !python -m spacy download en_core_web_sm -q

from google.colab import drive  # type: ignore[import-untyped]

drive.mount("/content/drive")

REPO_DIR = "/content/RelCheck"
if not os.path.exists(os.path.join(REPO_DIR, ".git")):
    os.system(f"git clone https://github.com/yadomkar/RelCheck.git {REPO_DIR}")
else:
    os.system(f"cd {REPO_DIR} && git pull -q")
sys.path.insert(0, REPO_DIR)

_log.info("Repository ready at %s", REPO_DIR)

# %%
# ── Cell 2: Configuration ──────────────────────────────────
# Set paths, API keys, and select benchmark/system/model.

# --- API Keys ---
OPENAI_API_KEY = ""  # <-- paste your OpenAI key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# --- Drive paths ---
DRIVE_BASE = "/content/drive/MyDrive/RelCheck_Data"

# POPE / COCO
POPE_DATA_DIR = f"{DRIVE_BASE}/pope"
COCO_IMAGE_DIR = "/content/coco_val2014/val2014"
COCO_INSTANCES_PATH = (
    f"{DRIVE_BASE}/coco_zips/annotations/instances_val2014.json"
)

# MME
MME_DATA_DIR = f"{DRIVE_BASE}/mme"

# AMBER
AMBER_DATA_DIR = f"{DRIVE_BASE}/amber/data"
AMBER_IMAGE_DIR = f"{DRIVE_BASE}/amber/images"

# --- Output & cache ---
RESULTS_DIR = f"{DRIVE_BASE}/eval_harness/results"
CACHE_DIR = f"{DRIVE_BASE}/eval_harness/cache"
EXPORT_DIR = f"{DRIVE_BASE}/eval_harness/export"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

# --- GroundingDINO ---
GDINO_DIR = "/content/groundingdino_weights"
GDINO_CHECKPOINT = f"{GDINO_DIR}/groundingdino_swint_ogc.pth"
os.makedirs(GDINO_DIR, exist_ok=True)
if not os.path.exists(GDINO_CHECKPOINT):
    os.system(
        f"wget -q -O {GDINO_CHECKPOINT} "
        "https://github.com/IDEA-Research/GroundingDINO/releases/download/"
        "v0.1.0-alpha/groundingdino_swint_ogc.pth"
    )

import groundingdino  # type: ignore[import-untyped]

GDINO_CONFIG = os.path.join(
    os.path.dirname(groundingdino.__file__),
    "config",
    "GroundingDINO_SwinT_OGC.py",
)

# --- RelTR ---
ENABLE_RELTR = True
RELTR_CHECKPOINT = f"{DRIVE_BASE}/checkpoint0149.pth"

if ENABLE_RELTR:
    RELTR_DIR = "/content/RelTR"
    if not os.path.exists(RELTR_DIR):
        os.system(f"git clone https://github.com/yrcong/RelTR.git {RELTR_DIR}")

import relcheck_v3.reltr.config as reltr_cfg

reltr_cfg.ENABLE_RELTR = ENABLE_RELTR
if ENABLE_RELTR:
    reltr_cfg.RELTR_CHECKPOINT = RELTR_CHECKPOINT

# --- Default model selection ---
MLLM_MODEL_ID = "llava-hf/llava-1.5-7b-hf"
CORRECTOR_MODEL = "gpt-5.4"

_log.info("Configuration complete.")

# %%
# ── Cell 3: MLLM Setup ─────────────────────────────────────
# Download and prepare the selected MLLM weights.

from pathlib import Path

from relcheck_v3.mllm.setup import (
    setup_llava15,
    setup_llava_v1,
    setup_minigpt4,
    setup_mplug_owl,
)

_SETUP_DISPATCH = {
    "llava-hf/llava-1.5-7b-hf": setup_llava15,
    "liuhaotian/llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3": setup_llava_v1,
    "MAGAer13/mplug-owl-llama-7b": setup_mplug_owl,
    "Vision-CAIR/MiniGPT-4": setup_minigpt4,
}

setup_fn = _SETUP_DISPATCH.get(MLLM_MODEL_ID)
if setup_fn is not None:
    _log.info("Setting up MLLM: %s", MLLM_MODEL_ID)
    setup_fn(weights_dir=Path("/content/weights/"))
    _log.info("MLLM setup complete.")
else:
    _log.warning("No setup function for model %s — skipping.", MLLM_MODEL_ID)

# %%
# ── Cell 4: Single Run ─────────────────────────────────────
# Run evaluation for one benchmark × system combination.

from relcheck_v3.eval.eval_runner import run_eval

# Choose which benchmark and system to run.
SINGLE_BENCHMARK = "pope"  # "pope", "mme", or "amber"
SINGLE_SYSTEM = "raw"  # "raw", "woodpecker", "claim", "claim+geom", "full"

# Map benchmark names to their data/image directories.
_BENCHMARK_DIRS: dict[str, dict[str, str]] = {
    "pope": {"data_dir": POPE_DATA_DIR, "image_dir": COCO_IMAGE_DIR},
    "mme": {"data_dir": MME_DATA_DIR, "image_dir": MME_DATA_DIR},
    "amber": {"data_dir": AMBER_DATA_DIR, "image_dir": AMBER_IMAGE_DIR},
}

dirs = _BENCHMARK_DIRS[SINGLE_BENCHMARK]
single_result = run_eval(
    benchmark=SINGLE_BENCHMARK,
    system=SINGLE_SYSTEM,
    mllm=MLLM_MODEL_ID,
    corrector=CORRECTOR_MODEL,
    data_dir=dirs["data_dir"],
    image_dir=dirs["image_dir"],
    output_dir=RESULTS_DIR,
    cache_dir=CACHE_DIR,
    coco_instances_path=(
        COCO_INSTANCES_PATH if SINGLE_BENCHMARK == "pope" else None
    ),
)
_log.info("Single run complete: %s × %s", SINGLE_BENCHMARK, SINGLE_SYSTEM)

# %%
# ── Cell 5: Full Matrix ────────────────────────────────────
# Run all 5 systems × 3 benchmarks = 15 evaluation runs.

ALL_SYSTEMS = ["raw", "woodpecker", "claim", "claim+geom", "full"]
ALL_BENCHMARKS = ["pope", "mme", "amber"]

matrix_results: list[dict] = []

for benchmark in ALL_BENCHMARKS:
    dirs = _BENCHMARK_DIRS[benchmark]
    coco_path = COCO_INSTANCES_PATH if benchmark == "pope" else None

    for system_id in ALL_SYSTEMS:
        _log.info("Running: %s × %s", benchmark, system_id)
        try:
            result = run_eval(
                benchmark=benchmark,
                system=system_id,
                mllm=MLLM_MODEL_ID,
                corrector=CORRECTOR_MODEL,
                data_dir=dirs["data_dir"],
                image_dir=dirs["image_dir"],
                output_dir=RESULTS_DIR,
                cache_dir=CACHE_DIR,
                coco_instances_path=coco_path,
            )
            matrix_results.append(result)
            _log.info("Completed: %s × %s", benchmark, system_id)
        except Exception:
            _log.exception("Failed: %s × %s", benchmark, system_id)

_log.info(
    "Full matrix complete: %d / %d runs succeeded.",
    len(matrix_results),
    len(ALL_SYSTEMS) * len(ALL_BENCHMARKS),
)

# %%
# ── Cell 6: Aggregation ────────────────────────────────────
# Build master table, ablation delta, and export to CSV/Markdown.

from relcheck_v3.eval.results_aggregator import ResultsAggregator

aggregator = ResultsAggregator(results_dir=RESULTS_DIR)

master_table = aggregator.build_master_table()
_log.info("Master table shape: %s", master_table.shape)

ablation_delta = aggregator.build_ablation_delta(baseline="woodpecker")
_log.info("Ablation delta computed against Woodpecker baseline.")

aggregator.export(output_dir=EXPORT_DIR)
_log.info("Results exported to %s", EXPORT_DIR)

# %%
# ── Cell 7: Display ────────────────────────────────────────
# Show results in the notebook.

import pandas as pd

_log.info("=== Master Results Table ===")
display(master_table)  # type: ignore[name-defined]  # noqa: F821

_log.info("=== Ablation Delta (vs. Woodpecker) ===")
display(ablation_delta)  # type: ignore[name-defined]  # noqa: F821

# Stratified tables (SCENE present vs. empty) if available.
stratified = aggregator.build_stratified_tables()
for tag_key, df in stratified.items():
    if df.notna().any().any():
        _log.info("=== Stratified: %s ===", tag_key)
        display(df)  # type: ignore[name-defined]  # noqa: F821

_log.info("Evaluation harness complete.")
