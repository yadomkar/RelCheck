# ============================================================
# RelCheck v3 — Smoke Test: MME × LLaVA v1 × 3 systems
# ============================================================
# Runs all 4 MME hallucination subtasks (existence, count, position, color)
# with RawMLLM, Woodpecker, and RelCheckFull.
#
# Uses pre-downloaded LLaVA v1 weights from /content/llava_weights/
# (or Drive fallback). Model swapping between LLaVA and GroundingDINO
# is handled automatically.
#
# Copy-paste each cell into Colab. Requires A100 GPU.
from __future__ import annotations

# %% [markdown]
# # Smoke Test: MME × 3 systems
# Runs existence, count, position, color on LLaVA v1 13B.
# Compares RawMLLM vs Woodpecker vs RelCheckFull.

# %%
# ── Cell 1: Setup ───────────────────────────────────────────
import logging, os, sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
_log = logging.getLogger("smoke_test")

# Fix HuggingFace download hanging
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_XET"] = "1"

!pip install -q openai>=1.0 pydantic>=2.0 tqdm pandas scikit-learn \
    Pillow tenacity groundingdino-py supervision==0.6.0 addict yapf \
    tabulate python-Levenshtein spacy
!pip install transformers==4.31.0 huggingface_hub==0.25.2 --force-reinstall -q
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

# Install LLaVA package
LLAVA_DIR = "/content/LLaVA"
if not os.path.exists(LLAVA_DIR):
    os.system(f"git clone https://github.com/haotian-liu/LLaVA.git {LLAVA_DIR}")
os.system(f"pip install -e {LLAVA_DIR} --no-deps -q")
sys.path.insert(0, LLAVA_DIR)

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

# LLaVA weights — local first, Drive fallback
LLAVA_WEIGHTS = "/content/llava_weights"
LLAVA_WEIGHTS_DRIVE = f"{DRIVE_BASE}/weights/llava-v1-13b"
if not os.path.exists(f"{LLAVA_WEIGHTS}/config.json"):
    if os.path.exists(f"{LLAVA_WEIGHTS_DRIVE}/config.json"):
        _log.info("Copying LLaVA weights from Drive...")
        os.makedirs(LLAVA_WEIGHTS, exist_ok=True)
        os.system(f"cp {LLAVA_WEIGHTS_DRIVE}/* {LLAVA_WEIGHTS}/")
    else:
        _log.error("LLaVA weights not found! Download them first.")

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

import relcheck_v3.reltr.config as reltr_cfg
reltr_cfg.ENABLE_RELTR = ENABLE_RELTR
reltr_cfg.RELTR_CHECKPOINT_PATH = RELTR_CHECKPOINT

MLLM_MODEL_ID = "liuhaotian/llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3"
_log.info("Config complete.")

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
# ── Cell 4: MLLM inference (LLaVA v1) ──────────────────────
import torch
from PIL import Image
from tqdm import tqdm
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, process_images
from transformers import AutoTokenizer

_log.info("Loading LLaVA v1 from %s", LLAVA_WEIGHTS)
tokenizer = AutoTokenizer.from_pretrained(LLAVA_WEIGHTS, use_fast=False)
model = LlavaLlamaForCausalLM.from_pretrained(
    LLAVA_WEIGHTS, torch_dtype=torch.float16, low_cpu_mem_usage=True
).cuda()
vision_tower = model.get_vision_tower()
if not vision_tower.is_loaded:
    vision_tower.load_model()
    vision_tower.to(device="cuda", dtype=torch.float16)
_log.info("LLaVA v1 loaded.")

# Run inference on ALL subtasks
mllm_outputs: dict[str, str] = {}
for s in tqdm(all_mme, desc="MLLM inference"):
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + s.question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    img = Image.open(s.image_path).convert("RGB")
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
    img_tensor = process_images([img], vision_tower.image_processor, model.config).to(dtype=torch.float16, device="cuda")
    with torch.inference_mode():
        out = model.generate(input_ids, images=img_tensor, max_new_tokens=50, do_sample=False)
    mllm_outputs[s.sample_id] = tokenizer.decode(out[0], skip_special_tokens=True).strip()

non_empty = sum(1 for v in mllm_outputs.values() if v)
_log.info("MLLM done: %d/%d non-empty", non_empty, len(mllm_outputs))

# Free LLaVA from GPU
del model, tokenizer, vision_tower
import gc; gc.collect(); torch.cuda.empty_cache()
_log.info("LLaVA freed. GPU ready for correction.")

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
# Per-sample diagnostics for inspection
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
# Inspect KB, corrections, and answer flips for each sample.
# Filter by subtask or look at mismatches only.

import json

# Save full diagnostics to Drive for later analysis
diag_path = f"{RESULTS_DIR}/sample_diagnostics.json"
with open(diag_path, "w") as f:
    json.dump(sample_diagnostics, f, indent=2)
_log.info("Saved %d sample diagnostics to %s", len(sample_diagnostics), diag_path)

# Show samples where RelCheckFull and Woodpecker disagree, or where
# RelCheckFull got it wrong but Woodpecker got it right
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

print("\n=== MME Hallucination Subtask Scores ===")
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
pivot.to_csv(f"{RESULTS_DIR}/mme_smoke_test.csv")
_log.info("Results saved to %s", RESULTS_DIR)
