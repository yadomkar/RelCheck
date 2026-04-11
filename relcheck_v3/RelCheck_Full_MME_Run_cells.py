# ============================================================
# RelCheck v3 — FULL MME Run: LLaVA v1 × 3 systems
# ============================================================
# Final production run. All 4 MME hallucination subtasks, no cap.
# Full logging to Drive, per-sample diagnostics, checkpointing.
#
# Estimated time: ~1.5 hours on A100
#   Cell 4 (LLaVA inference): ~20 min
#   Cell 6 (WP+Full correction): ~60 min
#
# Copy-paste each cell into Colab. Requires A100 GPU.
from __future__ import annotations

# %% [markdown]
# # Full MME Run: LLaVA v1 13B × 3 systems
# All samples, no cap. Final production run.

# %%
# ── Cell 1: Setup ───────────────────────────────────────────
import logging, os, sys, time

# Log to both console AND file on Drive
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_XET"] = "1"

!pip install -q openai>=1.0 pydantic>=2.0 tqdm pandas scikit-learn \
    Pillow tenacity groundingdino-py supervision==0.6.0 addict yapf \
    tabulate python-Levenshtein spacy
!pip install transformers==4.31.0 --no-deps -q
!pip install huggingface_hub==0.25.2 --no-deps -q
!python -m spacy download en_core_web_md -q
import transformers; print("transformers:", transformers.__version__)

from google.colab import drive
drive.mount("/content/drive")

# Paths
DRIVE_BASE = "/content/drive/MyDrive/RelCheck_Data"
RESULTS_DIR = f"{DRIVE_BASE}/eval_harness/full_mme/results"
CACHE_DIR = f"{DRIVE_BASE}/eval_harness/full_mme/cache"
LOG_DIR = f"{DRIVE_BASE}/eval_harness/full_mme/logs"
MME_DATA_DIR = f"{DRIVE_BASE}/mme"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Setup logging — console + file
_log_file = f"{LOG_DIR}/full_mme_run_{time.strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_log_file),
    ],
)
_log = logging.getLogger("full_mme")
_log.info("Log file: %s", _log_file)

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

# LLaVA weights — local first, Drive fallback
LLAVA_WEIGHTS = "/content/llava_weights"
LLAVA_WEIGHTS_DRIVE = f"{DRIVE_BASE}/weights/llava-v1-13b"
if not os.path.exists(f"{LLAVA_WEIGHTS}/config.json"):
    if os.path.exists(f"{LLAVA_WEIGHTS_DRIVE}/config.json"):
        _log.info("Copying LLaVA weights from Drive...")
        os.makedirs(LLAVA_WEIGHTS, exist_ok=True)
        os.system(f"cp {LLAVA_WEIGHTS_DRIVE}/* {LLAVA_WEIGHTS}/")
    else:
        _log.error("LLaVA weights not found!")

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
_log.info("Config complete. RESULTS_DIR=%s", RESULTS_DIR)

# %%
# ── Cell 3: Load ALL MME samples (no cap) ──────────────────
from relcheck_v3.benchmarks.mme import MMELoader

all_mme = list(MMELoader().iter_samples(MME_DATA_DIR))
subtasks = {}
for s in all_mme:
    subtasks.setdefault(s.split, []).append(s)

for name, samples in sorted(subtasks.items()):
    _log.info("MME %s: %d samples", name, len(samples))

# NO CAP — full run
all_mme = [s for samples in subtasks.values() for s in samples]
_log.info("Full MME: %d total samples across %d subtasks", len(all_mme), len(subtasks))

# %%
# ── Cell 4: MLLM inference (LLaVA v1) + checkpoint ─────────
import json, torch
from PIL import Image
from tqdm import tqdm
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, process_images
from transformers import AutoTokenizer

# Checkpoint: resume from saved MLLM outputs if available
MLLM_CKPT = f"{CACHE_DIR}/mllm_outputs.json"
if os.path.exists(MLLM_CKPT):
    with open(MLLM_CKPT) as f:
        mllm_outputs = json.load(f)
    _log.info("Loaded MLLM checkpoint: %d outputs", len(mllm_outputs))
else:
    mllm_outputs = {}

# Find samples that still need inference
remaining = [s for s in all_mme if s.sample_id not in mllm_outputs]
_log.info("MLLM inference: %d remaining / %d total", len(remaining), len(all_mme))

if remaining:
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

    for i, s in enumerate(tqdm(remaining, desc="MLLM inference")):
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

        # Checkpoint every 50 samples
        if (i + 1) % 50 == 0:
            with open(MLLM_CKPT, "w") as f:
                json.dump(mllm_outputs, f)
            _log.info("MLLM checkpoint saved (%d/%d)", i + 1, len(remaining))

    # Final save
    with open(MLLM_CKPT, "w") as f:
        json.dump(mllm_outputs, f)
    _log.info("MLLM outputs saved: %d total", len(mllm_outputs))

    # Free LLaVA
    del model, tokenizer, vision_tower
    import gc; gc.collect(); torch.cuda.empty_cache()
    _log.info("LLaVA freed.")
else:
    _log.info("All MLLM outputs cached — skipping inference.")

non_empty = sum(1 for v in mllm_outputs.values() if v)
_log.info("MLLM done: %d/%d non-empty", non_empty, len(mllm_outputs))

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
    _log.info("RawMLLM %s: score=%.1f", subtask_name, raw_results[subtask_name]["score"])

_log.info("RawMLLM total: %.1f", sum(r["score"] for r in raw_results.values()))

# %%
# ── Cell 6: Woodpecker + RelCheckFull (with checkpointing) ─
from pathlib import Path
from relcheck_v3.claim_generation.config import ClaimGenConfig
from relcheck_v3.claim_generation.pipeline import ClaimGenerationPipeline
from relcheck_v3.systems.relcheck_full import RelCheckFull
from relcheck_v3.eval.answer_extractor import AnswerExtractor
from relcheck_v3.mllm.cache import InferenceCache
import openai

resolved_key = os.environ.get("OPENAI_API_KEY", OPENAI_API_KEY)

# Woodpecker pipeline
wp_pipeline = ClaimGenerationPipeline(ClaimGenConfig(
    openai_api_key=resolved_key,
    gpt_model_id="gpt-5.4-mini",
    detector_config=GDINO_CONFIG,
    detector_model_path=GDINO_CHECKPOINT,
))
wp_client = openai.OpenAI(api_key=resolved_key)

# RelCheckFull
full_system = RelCheckFull(
    openai_api_key=resolved_key,
    corrector_model="gpt-5.4",
    gdino_config=GDINO_CONFIG,
    gdino_checkpoint=GDINO_CHECKPOINT,
    reltr_checkpoint=RELTR_CHECKPOINT,
    cache_dir=f"{CACHE_DIR}/systems/full/",
)

# Answer extractor
judge = AnswerExtractor(
    openai_api_key=resolved_key,
    model="gpt-5.4-mini",
    cache=InferenceCache(Path(f"{CACHE_DIR}/answers/")),
)

_log.info("Correction systems ready.")

# Load checkpoint if exists
DIAG_CKPT = f"{CACHE_DIR}/diagnostics_checkpoint.json"
if os.path.exists(DIAG_CKPT):
    with open(DIAG_CKPT) as f:
        _ckpt_data = json.load(f)
    sample_diagnostics = _ckpt_data.get("diagnostics", [])
    _completed_ids = set(d["sample_id"] for d in sample_diagnostics)
    _log.info("Loaded diagnostics checkpoint: %d completed", len(_completed_ids))
else:
    sample_diagnostics = []
    _completed_ids = set()

wp_results: dict[str, dict] = {}
full_results: dict[str, dict] = {}
_run_start = time.time()

for subtask_name, samples in subtasks.items():
    wp_preds = []
    full_preds = []
    subtask_start = time.time()

    for s in tqdm(samples, desc=f"WP+Full {subtask_name}"):
        mllm_out = mllm_outputs[s.sample_id]

        # Check if already completed
        if s.sample_id in _completed_ids:
            # Recover predictions from saved diagnostics
            saved = next(d for d in sample_diagnostics if d["sample_id"] == s.sample_id)
            wp_preds.append(saved.get("wp_answer", mme_extract_yesno(mllm_out)))
            full_preds.append(saved.get("full_answer", mme_extract_yesno(mllm_out)))
            continue

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
                diag["wp_vkb"] = f"(failed: {result.error_message})"
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

        wp_preds.append(wp_answer)
        full_preds.append(full_answer)
        sample_diagnostics.append(diag)
        _completed_ids.add(s.sample_id)

        # Checkpoint every 25 new samples
        if len(sample_diagnostics) % 25 == 0:
            with open(DIAG_CKPT, "w") as f:
                json.dump({"diagnostics": sample_diagnostics}, f)
            _log.info("Diagnostics checkpoint: %d samples", len(sample_diagnostics))

    wp_results[subtask_name] = mme_metrics([
        {"image_name": s.metadata.get("image_name", s.sample_id),
         "question": s.question, "predicted": p,
         "ground_truth": s.label.lower(), "subtask": subtask_name}
        for s, p in zip(samples, wp_preds)
    ])
    full_results[subtask_name] = mme_metrics([
        {"image_name": s.metadata.get("image_name", s.sample_id),
         "question": s.question, "predicted": p,
         "ground_truth": s.label.lower(), "subtask": subtask_name}
        for s, p in zip(samples, full_preds)
    ])

    elapsed = time.time() - subtask_start
    _log.info("%s done in %.1fs — Raw: %.0f, WP: %.0f, Full: %.0f",
              subtask_name, elapsed,
              raw_results[subtask_name]["score"],
              wp_results[subtask_name]["score"],
              full_results[subtask_name]["score"])

# Final checkpoint
with open(DIAG_CKPT, "w") as f:
    json.dump({"diagnostics": sample_diagnostics}, f)

total_elapsed = time.time() - _run_start
_log.info("Cell 6 total: %.1f min", total_elapsed / 60)

# Save full diagnostics to Drive
diag_path = f"{RESULTS_DIR}/full_mme_diagnostics.json"
with open(diag_path, "w") as f:
    json.dump(sample_diagnostics, f, indent=2)
_log.info("Saved %d diagnostics to %s", len(sample_diagnostics), diag_path)

# %%
# ── Cell 7: Results table + save ───────────────────────────
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

print("\n=== FULL MME Hallucination Subtask Scores (LLaVA v1) ===")
print(pivot.to_markdown())

# Per-subtask detail
print("\n=== Detailed Results ===")
for subtask_name in ["existence", "count", "position", "color"]:
    sub_df = df[df["subtask"] == subtask_name][["system", "accuracy", "accuracy_plus", "score"]]
    print(f"\n{subtask_name.upper()}:")
    print(sub_df.to_markdown(index=False))

# Total
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

# Save everything
pivot.to_csv(f"{RESULTS_DIR}/full_mme_scores.csv")
df.to_csv(f"{RESULTS_DIR}/full_mme_detailed.csv", index=False)
_log.info("Results saved to %s", RESULTS_DIR)

# %%
# ── Cell 8: Disagreement analysis ──────────────────────────
print("=" * 80)
print("WOODPECKER CORRECT BUT RELCHECK WRONG")
print("=" * 80)
_wp_wins = 0
for d in sample_diagnostics:
    gt = d["ground_truth"]
    wp_ok = d.get("wp_answer") == gt
    full_ok = d.get("full_answer") == gt
    if wp_ok and not full_ok:
        _wp_wins += 1
        print(f"\n{'─'*60}")
        print(f"[{d['subtask']}] {d['sample_id']}")
        print(f"  Q: {d['question']}")
        print(f"  GT: {gt}  |  Raw: {d['raw_answer']}  |  WP: {d.get('wp_answer')} ✓  |  Full: {d.get('full_answer')} ✗")
        print(f"  MLLM:          {d['mllm_output']}")
        print(f"  WP corrected:  {d.get('wp_corrected', 'N/A')}")
        print(f"  Full corrected: {d.get('full_corrected', 'N/A')}")
        if d.get("full_kb"):
            print(f"  KB:\n{d['full_kb']}")

print(f"\nTotal WP-wins: {_wp_wins}")

print("\n" + "=" * 80)
print("RELCHECK CORRECT BUT WOODPECKER WRONG")
print("=" * 80)
_full_wins = 0
for d in sample_diagnostics:
    gt = d["ground_truth"]
    wp_ok = d.get("wp_answer") == gt
    full_ok = d.get("full_answer") == gt
    if full_ok and not wp_ok:
        _full_wins += 1
        print(f"\n{'─'*60}")
        print(f"[{d['subtask']}] {d['sample_id']}")
        print(f"  Q: {d['question']}")
        print(f"  GT: {gt}  |  Raw: {d['raw_answer']}  |  WP: {d.get('wp_answer')} ✗  |  Full: {d.get('full_answer')} ✓")
        print(f"  MLLM:          {d['mllm_output']}")
        print(f"  Full corrected: {d.get('full_corrected', 'N/A')}")

print(f"\nTotal Full-wins: {_full_wins}")
print(f"\nNet advantage RelCheck: +{_full_wins - _wp_wins}")

# %%
# ── Cell 9: Summary stats ──────────────────────────────────
# Passthrough reasons, correction rates, per-subtask breakdown

_passthrough_reasons = {}
_corrected_count = 0
_total_count = len(sample_diagnostics)

for d in sample_diagnostics:
    if d.get("full_corrected") and d.get("full_corrected") != d.get("mllm_output"):
        _corrected_count += 1

# Per-subtask accuracy comparison
print("\n=== Per-Subtask Accuracy ===")
for st in ["existence", "count", "position", "color"]:
    st_diags = [d for d in sample_diagnostics if d["subtask"] == st]
    if not st_diags:
        continue
    raw_correct = sum(1 for d in st_diags if d["raw_answer"] == d["ground_truth"])
    wp_correct = sum(1 for d in st_diags if d.get("wp_answer") == d["ground_truth"])
    full_correct = sum(1 for d in st_diags if d.get("full_answer") == d["ground_truth"])
    n = len(st_diags)
    print(f"  {st}: Raw={raw_correct}/{n} ({raw_correct/n*100:.1f}%)  "
          f"WP={wp_correct}/{n} ({wp_correct/n*100:.1f}%)  "
          f"Full={full_correct}/{n} ({full_correct/n*100:.1f}%)")

print(f"\nRelCheckFull correction rate: {_corrected_count}/{_total_count} "
      f"({_corrected_count/_total_count*100:.1f}%)")

# KB layer coverage
_has_claim = sum(1 for d in sample_diagnostics if d.get("full_kb") and "Count:" in d["full_kb"] and d["full_kb"].split("Count:")[1].split("Specific:")[0].strip())
_has_scene = sum(1 for d in sample_diagnostics if d.get("full_kb") and "(conf=" in d.get("full_kb", ""))
_has_geom = sum(1 for d in sample_diagnostics if d.get("full_kb") and "is to the" in d.get("full_kb", ""))
print(f"\nKB coverage: CLAIM={_has_claim}/{_total_count}  SCENE={_has_scene}/{_total_count}  GEOM={_has_geom}/{_total_count}")

_log.info("Full MME run complete. Results at %s", RESULTS_DIR)
