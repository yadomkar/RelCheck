# ============================================================
# RelCheck v3 — Smoke Test 2: MME × LLaVA 1.5 7B × 3 systems
# ============================================================
# Runs all 4 MME hallucination subtasks (existence, count, position, color)
# with RawMLLM, Woodpecker, and RelCheckFull.
#
# Uses LLaVA-1.5-7B from HuggingFace (llava-hf/llava-1.5-7b-hf).
# No special repo clone needed — standard transformers API.
#
# Copy-paste each cell into Colab. Works on T4 or A100.
from __future__ import annotations

# %% [markdown]
# # Smoke Test 2: MME × 3 systems
# Runs existence, count, position, color on LLaVA 1.5 7B.
# Compares RawMLLM vs Woodpecker vs RelCheckFull.

# %%
# ── Cell 1: Setup ───────────────────────────────────────────
import logging, os, sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
_log = logging.getLogger("smoke_test_2")

# Fix HuggingFace download hanging
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_XET"] = "1"

!pip install -q openai>=1.0 pydantic>=2.0 tqdm pandas scikit-learn \
    Pillow tenacity groundingdino-py supervision==0.6.0 addict yapf \
    tabulate python-Levenshtein spacy
!pip install transformers==4.45.2 tokenizers==0.20.1 -q
!python -m spacy download en_core_web_md -q

from google.colab import drive
drive.mount("/content/drive")

REPO_DIR = "/content/RelCheck"
if not os.path.exists(os.path.join(REPO_DIR, ".git")):
    os.system(f"git clone https://github.com/yadomkar/RelCheck.git {REPO_DIR}")
else:
    os.system(f"cd {REPO_DIR} && git pull myfork main -q 2>/dev/null || git pull -q")
sys.path.insert(0, REPO_DIR)

_log.info("Setup complete.")

# %%
# ── Cell 2: Config ──────────────────────────────────────────
OPENAI_API_KEY = ""  # <-- paste your key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

DRIVE_BASE = "/content/drive/MyDrive/RelCheck_Data"
MME_DATA_DIR = f"{DRIVE_BASE}/mme"
RESULTS_DIR = f"{DRIVE_BASE}/eval_harness/smoke_test_2/results"
CACHE_DIR = f"{DRIVE_BASE}/eval_harness/smoke_test_2/cache"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# GroundingDINO
GDINO_DIR = "/content/groundingdino_weights"
GDINO_CHECKPOINT = f"{GDINO_DIR}/groundingdino_swint_ogc.pth"
os.makedirs(GDINO_DIR, exist_ok=True)
if not os.path.exists(GDINO_CHECKPOINT):
    os.system(f"wget -q -O {GDINO_CHECKPOINT} https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth")

import groundingdino
GDINO_CONFIG = os.path.join(os.path.dirname(groundingdino.__file__), "config", "GroundingDINO_SwinT_OGC.py")

# RelTR
ENABLE_RELTR = True
RELTR_CHECKPOINT = f"{DRIVE_BASE}/checkpoint0149.pth"
if ENABLE_RELTR:
    RELTR_DIR = "/content/RelTR"
    if not os.path.exists(RELTR_DIR):
        os.system(f"git clone https://github.com/yrcong/RelTR.git {RELTR_DIR}")
    # Download RelTR checkpoint if not on Drive
    if not os.path.exists(RELTR_CHECKPOINT):
        _log.info("Downloading RelTR checkpoint via gdown...")
        os.system("pip install -q gdown")
        os.system(f"gdown https://drive.google.com/uc?id=1F_B4v6oqKpXKdD9YGz2qGZFsGQFDL5JY -O {RELTR_CHECKPOINT}")
        if os.path.exists(RELTR_CHECKPOINT):
            _log.info("RelTR checkpoint downloaded (%.1f MB)", os.path.getsize(RELTR_CHECKPOINT)/1e6)
        else:
            _log.warning("RelTR download failed — disabling SCENE layer")
            ENABLE_RELTR = False

import relcheck_v3.reltr.config as reltr_cfg
reltr_cfg.ENABLE_RELTR = ENABLE_RELTR
reltr_cfg.RELTR_CHECKPOINT_PATH = RELTR_CHECKPOINT

MLLM_MODEL_ID = "llava-hf/llava-1.5-7b-hf"
_log.info("Config complete.")

# %%
# ── Cell 2b: Download MME data (new account — no cached data) ──
# Downloads MME from HuggingFace lmms-lab/MME dataset and converts
# to the directory layout expected by MMELoader:
#   {subtask}/{subtask}.txt  (tab-separated: image_name\tquestion\tanswer)
#   {subtask}/images/        (image files)
#
# Skip this cell if you already have MME data on Drive.
import os

# Check if all 4 subtask txt files exist
_all_exist = all(
    os.path.exists(f"{MME_DATA_DIR}/{st}/{st}.txt")
    for st in ["existence", "count", "position", "color"]
)
if not _all_exist:
    _log.info("Downloading MME data from HuggingFace...")
    !pip install datasets -q
    from datasets import load_dataset

    for subtask in ["existence", "count", "position", "color"]:
        _log.info("Downloading MME-%s...", subtask)
        ds = load_dataset("lmms-lab/MME", subtask, split="test")
        subtask_dir = f"{MME_DATA_DIR}/{subtask}"
        images_dir = f"{subtask_dir}/images"
        os.makedirs(images_dir, exist_ok=True)

        txt_lines = []
        for i, row in enumerate(ds):
            # Save image
            img = row["image"]
            img_name = row.get("image_name", f"{i}.png")
            if not img_name.endswith((".png", ".jpg", ".jpeg")):
                img_name = f"{img_name}.png"
            img_path = f"{images_dir}/{img_name}"
            if not os.path.exists(img_path):
                img.save(img_path)

            # Collect question/answer for the subtask txt file
            question = row["question"]
            answer = row["answer"]
            txt_lines.append(f"{img_name}\t{question}\t{answer}")

        # Write the single subtask.txt file
        txt_path = f"{subtask_dir}/{subtask}.txt"
        with open(txt_path, "w") as f:
            f.write("\n".join(txt_lines) + "\n")

        _log.info("MME-%s: %d samples saved to %s", subtask, len(ds), txt_path)
else:
    _log.info("MME data already exists at %s", MME_DATA_DIR)

# %%
# ── Cell 3: Load MME samples ───────────────────────────────
from relcheck_v3.benchmarks.mme import MMELoader

all_mme = list(MMELoader().iter_samples(MME_DATA_DIR))
subtasks = {}
for s in all_mme:
    subtasks.setdefault(s.split, []).append(s)

for name, samples in sorted(subtasks.items()):
    _log.info("MME %s: %d samples", name, len(samples))

# Cap samples per subtask for smoke test speed
MAX_PER_SUBTASK = 30
for name in subtasks:
    subtasks[name] = subtasks[name][:MAX_PER_SUBTASK]
all_mme = [s for samples in subtasks.values() for s in samples]
_log.info("Capped to %d total samples (%d per subtask)", len(all_mme), MAX_PER_SUBTASK)

# %%
# ── Cell 4: MLLM inference (LLaVA 1.5 7B) ─────────────────
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration

_log.info("Loading LLaVA 1.5 7B from %s", MLLM_MODEL_ID)
processor = AutoProcessor.from_pretrained(MLLM_MODEL_ID)
model = LlavaForConditionalGeneration.from_pretrained(
    MLLM_MODEL_ID, torch_dtype=torch.float16, device_map="auto",
)
_log.info("LLaVA 1.5 7B loaded.")

# Run inference on ALL subtasks
mllm_outputs: dict[str, str] = {}
for s in tqdm(all_mme, desc="MLLM inference"):
    img = Image.open(s.image_path).convert("RGB")
    conversation = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": s.question},
        ]},
    ]
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=text_prompt, images=img, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    # LLaVA 1.5 HF returns input+output — slice off input tokens
    generated = out[:, inputs["input_ids"].shape[-1]:]
    mllm_outputs[s.sample_id] = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

non_empty = sum(1 for v in mllm_outputs.values() if v)
_log.info("MLLM done: %d/%d non-empty", non_empty, len(mllm_outputs))

# Free LLaVA from GPU
del model, processor
import gc; gc.collect(); torch.cuda.empty_cache()
_log.info("LLaVA 1.5 freed. GPU ready for correction.")

# %%
# ── Cell 5: RawMLLM baseline ───────────────────────────────
from relcheck_v3.eval.harness_metrics import mme_extract_yesno, mme_metrics

raw_results: dict[str, dict] = {}
for subtask_name, samples in subtasks.items():
    preds = []
    for s in samples:
        output = mllm_outputs[s.sample_id]
        preds.append({
            "image_name": s.metadata.get("image_name", s.sample_id),
            "question": s.question,
            "predicted": mme_extract_yesno(output),
            "ground_truth": s.label.lower(),
            "subtask": subtask_name,
        })
    raw_results[subtask_name] = mme_metrics(preds)

_log.info("RawMLLM metrics computed.")

# %%
# ── Cell 6: Woodpecker + RelCheckFull correction ───────────
from pathlib import Path
from relcheck_v3.claim_generation.config import ClaimGenConfig
from relcheck_v3.claim_generation.pipeline import ClaimGenerationPipeline
from relcheck_v3.systems.relcheck_full import RelCheckFull
from relcheck_v3.eval.answer_extractor import AnswerExtractor
from relcheck_v3.mllm.cache import InferenceCache
import openai

resolved_key = os.environ.get("OPENAI_API_KEY", OPENAI_API_KEY)

# Woodpecker pipeline (gpt-5.4-mini for claim gen, simple correction prompt)
wp_pipeline = ClaimGenerationPipeline(ClaimGenConfig(
    openai_api_key=resolved_key,
    gpt_model_id="gpt-5.4-mini",
    detector_config=GDINO_CONFIG,
    detector_model_path=GDINO_CHECKPOINT,
))
wp_client = openai.OpenAI(api_key=resolved_key)

# RelCheckFull (3-layer KB + GPT-5.4 thinking correction)
full_system = RelCheckFull(
    openai_api_key=resolved_key,
    corrector_model="gpt-5.4",
    gdino_config=GDINO_CONFIG,
    gdino_checkpoint=GDINO_CHECKPOINT,
    reltr_checkpoint=RELTR_CHECKPOINT,
    cache_dir=f"{CACHE_DIR}/systems/full/",
)

# Answer extractor (LLM judge for yes/no)
judge = AnswerExtractor(
    openai_api_key=resolved_key,
    model="gpt-5.4-mini",
    cache=InferenceCache(Path(f"{CACHE_DIR}/answers/")),
)

_log.info("Correction systems ready.")

# Run both systems on all subtasks
wp_results: dict[str, dict] = {}
full_results: dict[str, dict] = {}
sample_diagnostics: list[dict] = []

for subtask_name, samples in subtasks.items():
    wp_preds = []
    full_preds = []

    for s in tqdm(samples, desc=f"WP+Full {subtask_name}"):
        mllm_out = mllm_outputs[s.sample_id]
        diag = {
            "sample_id": s.sample_id, "subtask": subtask_name,
            "image_path": s.image_path, "question": s.question,
            "ground_truth": s.label.lower(), "mllm_output": mllm_out,
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
                diag["wp_vkb"] = "(claim gen failed)"
            diag["wp_corrected"] = wp_corrected
            wp_answer = judge.extract_yesno(wp_corrected, s.question)
            diag["wp_answer"] = wp_answer
        except Exception as e:
            _log.error("WP failed %s: %s", s.sample_id, e)
            wp_answer = mme_extract_yesno(mllm_out)
            diag["wp_corrected"] = mllm_out
            diag["wp_answer"] = wp_answer
            diag["wp_error"] = str(e)

        # RelCheckFull
        try:
            full_corrected = full_system.correct(s.image_path, mllm_out)
            diag["full_corrected"] = full_corrected
            diag["full_kb"] = full_system.last_kb_text
            full_answer = judge.extract_yesno(full_corrected, s.question)
            diag["full_answer"] = full_answer
        except Exception as e:
            _log.error("Full failed %s: %s", s.sample_id, e)
            full_answer = mme_extract_yesno(mllm_out)
            diag["full_corrected"] = mllm_out
            diag["full_answer"] = full_answer
            diag["full_error"] = str(e)

        sample_diagnostics.append(diag)

        wp_preds.append({
            "image_name": s.metadata.get("image_name", s.sample_id),
            "question": s.question, "predicted": wp_answer,
            "ground_truth": s.label.lower(), "subtask": subtask_name,
        })
        full_preds.append({
            "image_name": s.metadata.get("image_name", s.sample_id),
            "question": s.question, "predicted": full_answer,
            "ground_truth": s.label.lower(), "subtask": subtask_name,
        })

    wp_results[subtask_name] = mme_metrics(wp_preds)
    full_results[subtask_name] = mme_metrics(full_preds)
    _log.info("%s done — WP: %.0f, Full: %.0f",
              subtask_name, wp_results[subtask_name]["score"],
              full_results[subtask_name]["score"])

# %%
# ── Cell 6b: Per-sample diagnostics ────────────────────────
import json

diag_path = f"{RESULTS_DIR}/sample_diagnostics.json"
with open(diag_path, "w") as f:
    json.dump(sample_diagnostics, f, indent=2)
_log.info("Saved %d sample diagnostics to %s", len(sample_diagnostics), diag_path)

print("=" * 80)
print("SAMPLES WHERE WOODPECKER CORRECT BUT RELCHECK WRONG")
print("=" * 80)
for d in sample_diagnostics:
    gt = d["ground_truth"]
    wp_ok = d.get("wp_answer") == gt
    full_ok = d.get("full_answer") == gt
    if wp_ok and not full_ok:
        print(f"\n{'─'*60}")
        print(f"[{d['subtask']}] {d['sample_id']}")
        print(f"  Question:       {d['question']}")
        print(f"  Ground truth:   {gt}")
        print(f"  MLLM output:    {d['mllm_output']}")
        print(f"  Raw answer:     {d['raw_answer']}")
        print(f"  WP corrected:   {d.get('wp_corrected', 'N/A')}")
        print(f"  WP answer:      {d.get('wp_answer')} ✓")
        print(f"  Full corrected: {d.get('full_corrected', 'N/A')}")
        print(f"  Full answer:    {d.get('full_answer')} ✗")
        if d.get("full_kb"):
            print(f"  Full KB:\n{d['full_kb']}")

print("\n" + "=" * 80)
print("SAMPLES WHERE RELCHECK CORRECT BUT WOODPECKER WRONG")
print("=" * 80)
for d in sample_diagnostics:
    gt = d["ground_truth"]
    wp_ok = d.get("wp_answer") == gt
    full_ok = d.get("full_answer") == gt
    if full_ok and not wp_ok:
        print(f"\n{'─'*60}")
        print(f"[{d['subtask']}] {d['sample_id']}")
        print(f"  Question:       {d['question']}")
        print(f"  Ground truth:   {gt}")
        print(f"  MLLM output:    {d['mllm_output']}")
        print(f"  WP corrected:   {d.get('wp_corrected', 'N/A')}")
        print(f"  WP answer:      {d.get('wp_answer')} ✗")
        print(f"  Full corrected: {d.get('full_corrected', 'N/A')}")
        print(f"  Full answer:    {d.get('full_answer')} ✓")

# %%
# ── Cell 7: Results table ──────────────────────────────────
import pandas as pd

rows = []
for subtask_name in ["existence", "count", "position", "color"]:
    if subtask_name in raw_results:
        rows.append({"subtask": subtask_name, "system": "RawMLLM", **raw_results[subtask_name]})
        rows.append({"subtask": subtask_name, "system": "Woodpecker", **wp_results[subtask_name]})
        rows.append({"subtask": subtask_name, "system": "RelCheckFull", **full_results[subtask_name]})

df = pd.DataFrame(rows)

# Pivot for thesis-ready table
pivot = df.pivot(index="system", columns="subtask", values="score")
pivot = pivot.reindex(["RawMLLM", "Woodpecker", "RelCheckFull"])
pivot = pivot[["existence", "count", "position", "color"]]

print("\n=== MME Hallucination Subtask Scores (LLaVA 1.5 7B) ===")
print(pivot.to_markdown())

# Per-subtask detail
print("\n=== Detailed Results ===")
for subtask_name in ["existence", "count", "position", "color"]:
    sub_df = df[df["subtask"] == subtask_name][["system", "accuracy", "accuracy_plus", "score"]]
    print(f"\n{subtask_name.upper()}:")
    print(sub_df.to_markdown(index=False))

# Total MME hallucination score (sum of all 4 subtasks)
print("\n=== Total MME Hallucination Score ===")
for system in ["RawMLLM", "Woodpecker", "RelCheckFull"]:
    total = sum(
        raw_results[st]["score"] if system == "RawMLLM"
        else wp_results[st]["score"] if system == "Woodpecker"
        else full_results[st]["score"]
        for st in ["existence", "count", "position", "color"]
        if st in raw_results
    )
    print(f"  {system}: {total:.1f}")

# Save to Drive
pivot.to_csv(f"{RESULTS_DIR}/mme_smoke_test_2_llava15.csv")
_log.info("Results saved to %s", RESULTS_DIR)
