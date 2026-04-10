# ============================================================
# RelCheck v3 — Smoke Test: MME × 50 images × 2 systems
# ============================================================
# Quick validation that the harness works end-to-end.
# Runs RawMLLM and RelCheckFull on 50 MME existence samples.
#
# Copy-paste each cell into Colab. Requires GPU (T4 or A100).
from __future__ import annotations

# %% [markdown]
# # Smoke Test: MME × 50 images
# Quick check that the eval harness works. Runs:
# - RawMLLM (passthrough baseline)
# - RelCheckFull (all 3 KB layers + GPT-5.4 thinking correction)
# on 50 MME existence samples with LLaVA v1 13B.

# %%
# ── Cell 1: Setup ───────────────────────────────────────────
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
_log = logging.getLogger("smoke_test")

!pip install -q openai>=1.0 pydantic>=2.0 tqdm pandas scikit-learn \
    transformers torch accelerate bitsandbytes Pillow tenacity \
    groundingdino-py tabulate

from google.colab import drive  # type: ignore[import-untyped]
drive.mount("/content/drive")

REPO_DIR = "/content/RelCheck"
if not os.path.exists(os.path.join(REPO_DIR, ".git")):
    os.system(f"git clone https://github.com/yadomkar/RelCheck.git {REPO_DIR}")
else:
    os.system(f"cd {REPO_DIR} && git pull -q")
sys.path.insert(0, REPO_DIR)

_log.info("Setup complete.")

# %%
# ── Cell 2: Config ──────────────────────────────────────────
OPENAI_API_KEY = ""  # <-- paste your key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

DRIVE_BASE = "/content/drive/MyDrive/RelCheck_Data"
MME_DATA_DIR = f"{DRIVE_BASE}/mme"
RESULTS_DIR = f"{DRIVE_BASE}/eval_harness/smoke_test/results"
CACHE_DIR = f"{DRIVE_BASE}/eval_harness/smoke_test/cache"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# GroundingDINO
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
    "config", "GroundingDINO_SwinT_OGC.py",
)

# RelTR
ENABLE_RELTR = True
RELTR_CHECKPOINT = f"{DRIVE_BASE}/checkpoint0149.pth"
if ENABLE_RELTR:
    RELTR_DIR = "/content/RelTR"
    if not os.path.exists(RELTR_DIR):
        os.system(f"git clone https://github.com/yrcong/RelTR.git {RELTR_DIR}")

import relcheck_v3.reltr.config as reltr_cfg
reltr_cfg.ENABLE_RELTR = ENABLE_RELTR
reltr_cfg.RELTR_CHECKPOINT_PATH = RELTR_CHECKPOINT

# MLLM — LLaVA v1 13B (Woodpecker's baseline, hallucinates more than 1.5)
MLLM_MODEL_ID = "liuhaotian/llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3"

_log.info("Config complete.")

# %%
# ── Cell 3: Setup LLaVA v1 ──────────────────────────────────
from pathlib import Path
from relcheck_v3.mllm.setup import setup_llava_v1

_log.info("Setting up LLaVA v1 13B...")
setup_llava_v1(weights_dir=Path("/content/weights/"))
_log.info("LLaVA v1 ready.")

# %%
# ── Cell 4: Load MME samples (cap at 50) ────────────────────
from relcheck_v3.benchmarks.mme import MMELoader

loader = MMELoader()
all_samples = list(loader.iter_samples(MME_DATA_DIR))
_log.info("Total MME samples: %d", len(all_samples))

# Keep only existence subtask, cap at 50
samples = [s for s in all_samples if s.split == "existence"][:50]
_log.info("Using %d existence samples for smoke test", len(samples))

# %%
# ── Cell 5: Run MLLM on samples ─────────────────────────────
from tqdm import tqdm
from relcheck_v3.mllm.wrapper import MLLMWrapper

mllm = MLLMWrapper(
    model_id=MLLM_MODEL_ID,
    cache_dir="/content/weights/",
    output_cache_dir=f"{CACHE_DIR}/mllm/",
)

mllm_outputs: dict[str, str] = {}
for s in tqdm(samples, desc="MLLM inference"):
    mllm_outputs[s.sample_id] = mllm.answer_yesno(s.image_path, s.question)

_log.info("MLLM inference done: %d outputs", len(mllm_outputs))

# %%
# ── Cell 6: Run RawMLLM (baseline) ──────────────────────────
from relcheck_v3.systems.raw_mllm import RawMLLM
from relcheck_v3.eval.harness_metrics import mme_extract_yesno, mme_metrics

raw = RawMLLM()
raw_preds = []
for s in samples:
    output = raw.correct(s.image_path, mllm_outputs[s.sample_id])
    raw_preds.append({
        "image_name": s.metadata.get("image_name", s.sample_id),
        "question": s.question,
        "predicted": mme_extract_yesno(output),
        "ground_truth": s.label,
        "subtask": s.split,
    })

raw_metrics = mme_metrics(raw_preds)
_log.info("RawMLLM MME-existence: acc=%.3f, acc+=%.3f, score=%.1f",
          raw_metrics["accuracy"], raw_metrics["accuracy_plus"], raw_metrics["score"])

# %%
# ── Cell 7: Run RelCheckFull ────────────────────────────────
from relcheck_v3.systems.relcheck_full import RelCheckFull
from relcheck_v3.eval.answer_extractor import AnswerExtractor
from relcheck_v3.mllm.cache import InferenceCache

full_system = RelCheckFull(
    openai_api_key=OPENAI_API_KEY,
    corrector_model="gpt-5.4",
    gdino_config=GDINO_CONFIG,
    gdino_checkpoint=GDINO_CHECKPOINT,
    reltr_checkpoint=RELTR_CHECKPOINT,
    cache_dir=f"{CACHE_DIR}/systems/full/",
)

judge = AnswerExtractor(
    openai_api_key=OPENAI_API_KEY,
    model="gpt-5.4-mini",
    cache=InferenceCache(Path(f"{CACHE_DIR}/answers/")),
)

full_preds = []
for s in tqdm(samples, desc="RelCheckFull"):
    mllm_out = mllm_outputs[s.sample_id]
    try:
        corrected = full_system.correct(s.image_path, mllm_out)
        answer = judge.extract_yesno(corrected, s.question)
    except Exception as e:
        _log.error("Failed %s: %s", s.sample_id, e)
        answer = mme_extract_yesno(mllm_out)

    full_preds.append({
        "image_name": s.metadata.get("image_name", s.sample_id),
        "question": s.question,
        "predicted": answer,
        "ground_truth": s.label,
        "subtask": s.split,
    })

full_metrics = mme_metrics(full_preds)
_log.info("RelCheckFull MME-existence: acc=%.3f, acc+=%.3f, score=%.1f",
          full_metrics["accuracy"], full_metrics["accuracy_plus"], full_metrics["score"])

# %%
# ── Cell 8: Compare ─────────────────────────────────────────
import pandas as pd

df = pd.DataFrame([
    {"system": "RawMLLM", **raw_metrics},
    {"system": "RelCheckFull", **full_metrics},
])
print("\n=== Smoke Test Results (MME-existence, 50 samples) ===")
print(df.to_markdown(index=False))

delta = full_metrics["score"] - raw_metrics["score"]
_log.info("RelCheckFull gain over RawMLLM: %.1f points", delta)
_log.info("Smoke test complete.")
