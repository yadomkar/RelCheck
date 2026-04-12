# ============================================================
# RelCheck v3 — POPE Benchmark: LLaVA v1 × 3 systems
# ============================================================
# Standalone POPE evaluation. New Colab session.
# Uses LLaVA v1 13B + COCO val2014 images (on Drive).
# 3 settings: random, popular, adversarial.
from __future__ import annotations

# %%
# ── POPE Cell 0: Setup ─────────────────────────────────────
import logging, os, sys, time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
_log = logging.getLogger("pope_run")

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_XET"] = "1"

!pip install -q openai>=1.0 pydantic>=2.0 tqdm pandas scikit-learn \
    Pillow tenacity groundingdino-py supervision==0.6.0 addict yapf \
    tabulate python-Levenshtein spacy
!pip install tokenizers==0.15.1 -q
!pip install transformers==4.37.2 --no-deps -q
!pip install huggingface_hub==0.25.2 --no-deps -q
!python -m spacy download en_core_web_md -q
import transformers; print("transformers:", transformers.__version__)

from google.colab import drive
drive.mount("/content/drive")

REPO_DIR = "/content/RelCheck"
if not os.path.exists(os.path.join(REPO_DIR, ".git")):
    os.system(f"git clone https://github.com/yadomkar/RelCheck.git {REPO_DIR}")
else:
    os.system(f"cd {REPO_DIR} && git pull myfork main -q 2>/dev/null || git pull -q")
sys.path.insert(0, REPO_DIR)

LLAVA_DIR = "/content/LLaVA"
if not os.path.exists(LLAVA_DIR):
    os.system(f"git clone https://github.com/haotian-liu/LLaVA.git {LLAVA_DIR}")
os.system(f"pip install -e {LLAVA_DIR} --no-deps -q")
sys.path.insert(0, LLAVA_DIR)

_log.info("Setup complete.")

# %%
# ── POPE Cell 0b: Config ───────────────────────────────────
OPENAI_API_KEY = ""  # <-- paste your key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

DRIVE_BASE = "/content/drive/MyDrive/RelCheck_Data"

LLAVA_WEIGHTS = "/content/llava_weights"
LLAVA_WEIGHTS_DRIVE = f"{DRIVE_BASE}/weights/llava-v1-13b"
if not os.path.exists(f"{LLAVA_WEIGHTS}/config.json"):
    if os.path.exists(f"{LLAVA_WEIGHTS_DRIVE}/config.json"):
        _log.info("Copying LLaVA weights from Drive...")
        os.makedirs(LLAVA_WEIGHTS, exist_ok=True)
        os.system(f"cp {LLAVA_WEIGHTS_DRIVE}/* {LLAVA_WEIGHTS}/")
    else:
        _log.error("LLaVA weights not found!")

GDINO_DIR = "/content/groundingdino_weights"
GDINO_CHECKPOINT = f"{GDINO_DIR}/groundingdino_swint_ogc.pth"
os.makedirs(GDINO_DIR, exist_ok=True)
if not os.path.exists(GDINO_CHECKPOINT):
    os.system(f"wget -q -O {GDINO_CHECKPOINT} https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth")

import groundingdino
GDINO_CONFIG = os.path.join(os.path.dirname(groundingdino.__file__), "config", "GroundingDINO_SwinT_OGC.py")

ENABLE_RELTR = True
RELTR_CHECKPOINT = f"{DRIVE_BASE}/checkpoint0149.pth"
if ENABLE_RELTR:
    RELTR_DIR = "/content/RelTR"
    if not os.path.exists(RELTR_DIR):
        os.system(f"git clone https://github.com/yrcong/RelTR.git {RELTR_DIR}")

import relcheck_v3.reltr.config as reltr_cfg
reltr_cfg.ENABLE_RELTR = ENABLE_RELTR
reltr_cfg.RELTR_CHECKPOINT_PATH = RELTR_CHECKPOINT

_log.info("Config complete.")

# %%
# ── POPE Cell 1: Download POPE data + unzip COCO ───────────
import json, zipfile

POPE_DIR = f"{DRIVE_BASE}/pope"
POPE_DATA_DIR = f"{POPE_DIR}/data"
COCO_ZIP = f"{DRIVE_BASE}/coco_zips/val2014.zip"
COCO_IMAGE_DIR = "/content/coco_val2014/val2014"
COCO_INSTANCES_PATH = f"{DRIVE_BASE}/coco_zips/annotations/instances_val2014.json"
POPE_RESULTS_DIR = f"{DRIVE_BASE}/eval_harness/pope/results"
POPE_CACHE_DIR = f"{DRIVE_BASE}/eval_harness/pope/cache"
os.makedirs(POPE_DATA_DIR, exist_ok=True)
os.makedirs(POPE_RESULTS_DIR, exist_ok=True)
os.makedirs(POPE_CACHE_DIR, exist_ok=True)

# Clone POPE repo to get the pre-built question files
_POPE_REPO = "/content/POPE_repo"
if not os.path.exists(_POPE_REPO):
    _log.info("Cloning POPE repo...")
    os.system(f"git clone --depth 1 https://github.com/RUCAIBox/POPE.git {_POPE_REPO}")

# Copy question files (they're in output/coco/)
for fname in ["coco_pope_random.json", "coco_pope_popular.json", "coco_pope_adversarial.json"]:
    src = f"{_POPE_REPO}/output/coco/{fname}"
    dest = f"{POPE_DATA_DIR}/{fname}"
    if not os.path.exists(dest):
        if os.path.exists(src):
            os.system(f"cp '{src}' '{dest}'")
            _log.info("Copied %s", fname)
        else:
            _log.error("POPE file not found: %s", src)
    else:
        _log.info("Already have %s", fname)

# Unzip COCO val2014 images to local disk
if not os.path.exists(COCO_IMAGE_DIR) or len(os.listdir(COCO_IMAGE_DIR)) < 100:
    if os.path.exists(COCO_ZIP):
        _log.info("Unzipping COCO val2014...")
        os.makedirs(os.path.dirname(COCO_IMAGE_DIR), exist_ok=True)
        with zipfile.ZipFile(COCO_ZIP, "r") as zf:
            zf.extractall(os.path.dirname(COCO_IMAGE_DIR))
        _log.info("COCO images: %d files", len(os.listdir(COCO_IMAGE_DIR)))
    else:
        _log.error("COCO zip not found at %s", COCO_ZIP)
else:
    _log.info("COCO images already extracted: %d files", len(os.listdir(COCO_IMAGE_DIR)))

# Verify a POPE file
with open(f"{POPE_DATA_DIR}/coco_pope_random.json") as f:
    first_line = json.loads(f.readline())
    _log.info("POPE sample: %s", first_line)

# %%
# ── POPE Cell 2: Load samples ──────────────────────────────
from relcheck_v3.benchmarks.pope import POPELoader

all_pope = list(POPELoader().iter_samples(
    POPE_DATA_DIR, COCO_IMAGE_DIR, COCO_INSTANCES_PATH
))

pope_splits = {}
for s in all_pope:
    pope_splits.setdefault(s.split, []).append(s)

for name, samples in sorted(pope_splits.items()):
    _log.info("POPE %s: %d samples", name, len(samples))
_log.info("POPE total: %d samples", len(all_pope))

# %%
# ── POPE Cell 3: Reload LLaVA v1 + inference ───────────────
import torch
from PIL import Image
from tqdm import tqdm
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, process_images
from transformers import AutoTokenizer

# Checkpoint
POPE_MLLM_CKPT = f"{POPE_CACHE_DIR}/pope_mllm_outputs.json"
if os.path.exists(POPE_MLLM_CKPT):
    with open(POPE_MLLM_CKPT) as f:
        pope_mllm_outputs = json.load(f)
    _log.info("Loaded POPE MLLM checkpoint: %d outputs", len(pope_mllm_outputs))
else:
    pope_mllm_outputs = {}

remaining = [s for s in all_pope if s.sample_id not in pope_mllm_outputs]
_log.info("POPE MLLM: %d remaining / %d total", len(remaining), len(all_pope))

if remaining:
    _log.info("Loading LLaVA v1...")
    tokenizer = AutoTokenizer.from_pretrained(LLAVA_WEIGHTS, use_fast=False)
    model = LlavaLlamaForCausalLM.from_pretrained(
        LLAVA_WEIGHTS, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).cuda()
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
        vision_tower.to(device="cuda", dtype=torch.float16)
    _log.info("LLaVA v1 loaded.")

    for i, s in enumerate(tqdm(remaining, desc="POPE MLLM inference")):
        try:
            conv = conv_templates["llava_v1"].copy()
            conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + s.question)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            img = Image.open(s.image_path).convert("RGB")
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
            img_tensor = process_images([img], vision_tower.image_processor, model.config).to(dtype=torch.float16, device="cuda")
            with torch.inference_mode():
                out = model.generate(input_ids, images=img_tensor, max_new_tokens=50, do_sample=False)
            pope_mllm_outputs[s.sample_id] = tokenizer.decode(out[0], skip_special_tokens=True).strip()
        except Exception as e:
            _log.error("POPE inference failed %s: %s", s.sample_id, e)
            pope_mllm_outputs[s.sample_id] = ""

        if (i + 1) % 100 == 0:
            with open(POPE_MLLM_CKPT, "w") as f:
                json.dump(pope_mllm_outputs, f)
            _log.info("POPE MLLM checkpoint: %d/%d", i + 1, len(remaining))

    with open(POPE_MLLM_CKPT, "w") as f:
        json.dump(pope_mllm_outputs, f)

    del model, tokenizer, vision_tower
    import gc; gc.collect(); torch.cuda.empty_cache()
    _log.info("LLaVA freed.")

_log.info("POPE MLLM done: %d outputs", len(pope_mllm_outputs))

# %%
# ── POPE Cell 4: RawMLLM baseline ──────────────────────────
from relcheck_v3.eval.harness_metrics import mme_extract_yesno

pope_raw_results: dict[str, dict] = {}
for split_name, samples in pope_splits.items():
    preds = [mme_extract_yesno(pope_mllm_outputs.get(s.sample_id, "")) for s in samples]
    gts = [s.label for s in samples]
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    pope_raw_results[split_name] = {
        "accuracy": accuracy_score(gts, preds),
        "precision": precision_score(gts, preds, pos_label="yes", zero_division=0),
        "recall": recall_score(gts, preds, pos_label="yes", zero_division=0),
        "f1": f1_score(gts, preds, pos_label="yes", zero_division=0),
        "yes_rate": sum(1 for p in preds if p == "yes") / len(preds),
    }
    _log.info("POPE RawMLLM %s: acc=%.3f f1=%.3f", split_name, pope_raw_results[split_name]["accuracy"], pope_raw_results[split_name]["f1"])

# %%
# ── POPE Cell 5: Woodpecker + RelCheckFull correction ──────
from pathlib import Path
import time

# Reinitialize correction systems if not in memory
try:
    _ = wp_pipeline
    _log.info("Correction systems already in memory.")
except NameError:
    from relcheck_v3.claim_generation.config import ClaimGenConfig
    from relcheck_v3.claim_generation.pipeline import ClaimGenerationPipeline
    from relcheck_v3.systems.relcheck_full import RelCheckFull
    from relcheck_v3.eval.answer_extractor import AnswerExtractor
    from relcheck_v3.mllm.cache import InferenceCache
    import openai

    resolved_key = os.environ.get("OPENAI_API_KEY", OPENAI_API_KEY)
    wp_pipeline = ClaimGenerationPipeline(ClaimGenConfig(
        openai_api_key=resolved_key, gpt_model_id="gpt-5.4-mini",
        detector_config=GDINO_CONFIG, detector_model_path=GDINO_CHECKPOINT,
    ))
    wp_client = openai.OpenAI(api_key=resolved_key)
    full_system = RelCheckFull(
        openai_api_key=resolved_key, corrector_model="gpt-5.4",
        gdino_config=GDINO_CONFIG, gdino_checkpoint=GDINO_CHECKPOINT,
        reltr_checkpoint=RELTR_CHECKPOINT,
        cache_dir=f"{POPE_CACHE_DIR}/systems/full/",
    )
    judge = AnswerExtractor(
        openai_api_key=resolved_key, model="gpt-5.4-mini",
        cache=InferenceCache(Path(f"{POPE_CACHE_DIR}/answers/")),
    )
    _log.info("Correction systems initialized.")

# Checkpoint
POPE_DIAG_CKPT = f"{POPE_CACHE_DIR}/pope_diagnostics_checkpoint.json"
if os.path.exists(POPE_DIAG_CKPT):
    with open(POPE_DIAG_CKPT) as f:
        pope_diagnostics = json.load(f).get("diagnostics", [])
    _completed = set(d["sample_id"] for d in pope_diagnostics)
    _log.info("Loaded POPE diagnostics checkpoint: %d", len(_completed))
else:
    pope_diagnostics = []
    _completed = set()

pope_wp_results: dict[str, dict] = {}
pope_full_results: dict[str, dict] = {}
_run_start = time.time()

for split_name, samples in pope_splits.items():
    wp_preds = []
    full_preds = []

    for s in tqdm(samples, desc=f"POPE WP+Full {split_name}"):
        mllm_out = pope_mllm_outputs.get(s.sample_id, "")

        if s.sample_id in _completed:
            saved = next(d for d in pope_diagnostics if d["sample_id"] == s.sample_id)
            wp_preds.append(saved.get("wp_answer", mme_extract_yesno(mllm_out)))
            full_preds.append(saved.get("full_answer", mme_extract_yesno(mllm_out)))
            continue

        diag = {
            "sample_id": s.sample_id, "split": split_name,
            "image_path": s.image_path, "question": s.question,
            "ground_truth": s.label, "mllm_output": mllm_out,
            "raw_answer": mme_extract_yesno(mllm_out),
        }

        # Woodpecker
        try:
            result = wp_pipeline.process_single(
                image=s.image_path, ref_cap=mllm_out, image_id=Path(s.image_path).stem)
            if result.success:
                vkb_text = result.visual_knowledge_base.format()
                prompt = (f"Based on the given claims, please correct the description of the image.\n"
                          f"Description: {mllm_out}\nClaims: {vkb_text}\nCorrected description:")
                resp = wp_client.chat.completions.create(
                    model="gpt-5.4-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=512)
                wp_corrected = resp.choices[0].message.content or mllm_out
                diag["wp_vkb"] = vkb_text
            else:
                wp_corrected = mllm_out
            diag["wp_corrected"] = wp_corrected
            wp_answer = judge.extract_yesno(wp_corrected, s.question)
            diag["wp_answer"] = wp_answer
        except Exception as e:
            _log.error("POPE WP failed %s: %s", s.sample_id, e)
            wp_answer = mme_extract_yesno(mllm_out)
            diag["wp_answer"] = wp_answer

        # RelCheckFull
        try:
            full_corrected = full_system.correct(s.image_path, mllm_out)
            diag["full_corrected"] = full_corrected
            diag["full_kb"] = full_system.last_kb_text
            full_answer = judge.extract_yesno(full_corrected, s.question)
            diag["full_answer"] = full_answer
        except Exception as e:
            _log.error("POPE Full failed %s: %s", s.sample_id, e)
            full_answer = mme_extract_yesno(mllm_out)
            diag["full_answer"] = full_answer

        wp_preds.append(wp_answer)
        full_preds.append(full_answer)
        pope_diagnostics.append(diag)
        _completed.add(s.sample_id)

        if len(pope_diagnostics) % 50 == 0:
            with open(POPE_DIAG_CKPT, "w") as f:
                json.dump({"diagnostics": pope_diagnostics}, f)
            _log.info("POPE checkpoint: %d", len(pope_diagnostics))

    gts = [s.label for s in samples]
    pope_wp_results[split_name] = {
        "accuracy": accuracy_score(gts, wp_preds),
        "precision": precision_score(gts, wp_preds, pos_label="yes", zero_division=0),
        "recall": recall_score(gts, wp_preds, pos_label="yes", zero_division=0),
        "f1": f1_score(gts, wp_preds, pos_label="yes", zero_division=0),
    }
    pope_full_results[split_name] = {
        "accuracy": accuracy_score(gts, full_preds),
        "precision": precision_score(gts, full_preds, pos_label="yes", zero_division=0),
        "recall": recall_score(gts, full_preds, pos_label="yes", zero_division=0),
        "f1": f1_score(gts, full_preds, pos_label="yes", zero_division=0),
    }
    _log.info("POPE %s — Raw: %.3f, WP: %.3f, Full: %.3f (accuracy)",
              split_name,
              pope_raw_results[split_name]["accuracy"],
              pope_wp_results[split_name]["accuracy"],
              pope_full_results[split_name]["accuracy"])

# Final save
with open(POPE_DIAG_CKPT, "w") as f:
    json.dump({"diagnostics": pope_diagnostics}, f)
with open(f"{POPE_RESULTS_DIR}/pope_diagnostics.json", "w") as f:
    json.dump(pope_diagnostics, f, indent=2)
_log.info("POPE done in %.1f min. Saved %d diagnostics.", (time.time()-_run_start)/60, len(pope_diagnostics))

# %%
# ── POPE Cell 6: Results table ─────────────────────────────
import pandas as pd

rows = []
for split_name in ["random", "popular", "adversarial"]:
    if split_name in pope_raw_results:
        rows.append({"split": split_name, "system": "RawMLLM", **pope_raw_results[split_name]})
        rows.append({"split": split_name, "system": "Woodpecker", **pope_wp_results[split_name]})
        rows.append({"split": split_name, "system": "RelCheckFull", **pope_full_results[split_name]})

df = pd.DataFrame(rows)

print("\n=== POPE Accuracy (LLaVA v1) ===")
pivot_acc = df.pivot(index="system", columns="split", values="accuracy")
pivot_acc = pivot_acc.reindex(["RawMLLM", "Woodpecker", "RelCheckFull"])
pivot_acc = pivot_acc[["random", "popular", "adversarial"]]
print(pivot_acc.to_markdown())

print("\n=== POPE F1 ===")
pivot_f1 = df.pivot(index="system", columns="split", values="f1")
pivot_f1 = pivot_f1.reindex(["RawMLLM", "Woodpecker", "RelCheckFull"])
pivot_f1 = pivot_f1[["random", "popular", "adversarial"]]
print(pivot_f1.to_markdown())

print("\n=== Detailed Results ===")
for split_name in ["random", "popular", "adversarial"]:
    sub_df = df[df["split"] == split_name][["system", "accuracy", "precision", "recall", "f1"]]
    print(f"\n{split_name.upper()}:")
    print(sub_df.to_markdown(index=False))

pivot_acc.to_csv(f"{POPE_RESULTS_DIR}/pope_accuracy.csv")
pivot_f1.to_csv(f"{POPE_RESULTS_DIR}/pope_f1.csv")
df.to_csv(f"{POPE_RESULTS_DIR}/pope_detailed.csv", index=False)
_log.info("POPE results saved to %s", POPE_RESULTS_DIR)
