# ============================================================
# RelCheck v3 — AMBER Smoke Test (LLaVA 1.5 7B)
# ============================================================
# Standalone AMBER evaluation. Run on a clean Colab (no
# transformers==4.31.0 pin). Uses HuggingFace datasets library
# to download AMBER with pre-merged answer labels.
#
# Copy-paste each cell into Colab. Works on T4 or A100.
from __future__ import annotations

# %% [markdown]
# # AMBER Smoke Test: LLaVA 1.5 7B × 3 systems

# %%
# ── Cell 1: Setup ───────────────────────────────────────────
import logging, os, sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
_log = logging.getLogger("amber_smoke")

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_XET"] = "1"

!pip install -q openai>=1.0 pydantic>=2.0 tqdm pandas scikit-learn \
    Pillow tenacity groundingdino-py supervision==0.6.0 addict yapf \
    tabulate python-Levenshtein spacy datasets
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
AMBER_DIR = f"{DRIVE_BASE}/amber"
AMBER_DATA_DIR = f"{AMBER_DIR}/data"
AMBER_IMAGE_DIR = f"{AMBER_DIR}/images"
RESULTS_DIR = f"{DRIVE_BASE}/eval_harness/amber_smoke/results"
CACHE_DIR = f"{DRIVE_BASE}/eval_harness/amber_smoke/cache"
os.makedirs(AMBER_DATA_DIR, exist_ok=True)
os.makedirs(AMBER_IMAGE_DIR, exist_ok=True)
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
    if not os.path.exists(RELTR_CHECKPOINT):
        _log.info("Downloading RelTR checkpoint...")
        os.system("pip install -q gdown")
        os.system(f"gdown 1F_B4v6oqKpXKdD9YGz2qGZFsGQFDL5JY -O {RELTR_CHECKPOINT}")
        if not os.path.exists(RELTR_CHECKPOINT):
            _log.warning("RelTR download failed — disabling SCENE layer")
            ENABLE_RELTR = False

import relcheck_v3.reltr.config as reltr_cfg
reltr_cfg.ENABLE_RELTR = ENABLE_RELTR
reltr_cfg.RELTR_CHECKPOINT_PATH = RELTR_CHECKPOINT

MLLM_MODEL_ID = "llava-hf/llava-1.5-7b-hf"
_log.info("Config complete.")

# %%
# ── Cell 3: Download AMBER data from HuggingFace ───────────
# Uses datasets library — works in clean environment (no transformers pin)
import json
from datasets import load_dataset

_amber_jsons = [
    ("query_discriminative_existence", "query_discriminative-existence.json"),
    ("query_discriminative_attribute", "query_discriminative-attribute.json"),
    ("query_discriminative_relation", "query_discriminative-relation.json"),
]

_all_exist = all(os.path.exists(f"{AMBER_DATA_DIR}/{f}") for _, f in _amber_jsons)

if not _all_exist:
    _log.info("Downloading AMBER from HuggingFace...")
    for hf_split, json_filename in _amber_jsons:
        _log.info("Downloading AMBER %s...", hf_split)
        ds = load_dataset("visual-preference/AMBER", split=hf_split)

        entries = []
        for row in ds:
            entry_id = row["id"]
            img_name = f"{entry_id}.png"
            img_path = f"{AMBER_IMAGE_DIR}/{img_name}"
            if not os.path.exists(img_path):
                row["image"].save(img_path)

            entries.append({
                "id": entry_id,
                "image": img_name,
                "query": row["query"],
                "answer": str(row["answer"]).strip().lower(),
            })

        with open(f"{AMBER_DATA_DIR}/{json_filename}", "w") as f:
            json.dump(entries, f)

        yes_n = sum(1 for e in entries if e["answer"] == "yes")
        _log.info("  %s: %d samples (yes=%d, no=%d)", json_filename, len(entries), yes_n, len(entries)-yes_n)

    # Also download generative (AMBERLoader requires it to exist)
    _log.info("Downloading AMBER generative split...")
    ds_g = load_dataset("visual-preference/AMBER", split="query_generative")
    gen_entries = []
    for row in ds_g:
        entry_id = row["id"]
        img_name = f"{entry_id}.png"
        img_path = f"{AMBER_IMAGE_DIR}/{img_name}"
        if not os.path.exists(img_path):
            row["image"].save(img_path)
        gen_entries.append({"id": entry_id, "image": img_name})
    with open(f"{AMBER_DATA_DIR}/query_generative.json", "w") as f:
        json.dump(gen_entries, f)
    _log.info("  generative: %d samples", len(gen_entries))
else:
    _log.info("AMBER data already exists at %s", AMBER_DATA_DIR)

# Verify
with open(f"{AMBER_DATA_DIR}/{_amber_jsons[0][1]}") as f:
    _sample = json.load(f)[:3]
for s in _sample:
    _log.info("Sample: id=%s image=%s answer=%s query=%s...", s["id"], s["image"], s["answer"], s["query"][:50])

# %%
# ── Cell 4: Load AMBER samples + cap ───────────────────────
from relcheck_v3.benchmarks.amber import AMBERLoader

all_amber = [
    s for s in AMBERLoader().iter_samples(AMBER_DATA_DIR, AMBER_IMAGE_DIR)
    if s.split != "g"
]

amber_subtasks = {}
for s in all_amber:
    amber_subtasks.setdefault(s.split, []).append(s)

for name, samples in sorted(amber_subtasks.items()):
    yes_n = sum(1 for s in samples if s.label == "yes")
    _log.info("AMBER %s: %d samples (yes=%d, no=%d)", name, len(samples), yes_n, len(samples)-yes_n)

AMBER_MAX_PER_SUBTASK = 30
for name in amber_subtasks:
    amber_subtasks[name] = amber_subtasks[name][:AMBER_MAX_PER_SUBTASK]
all_amber = [s for samples in amber_subtasks.values() for s in samples]
_log.info("Capped to %d total AMBER samples (%d per subtask)", len(all_amber), AMBER_MAX_PER_SUBTASK)

# %%
# ── Cell 5: MLLM inference (LLaVA 1.5 7B) ─────────────────
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

amber_mllm_outputs: dict[str, str] = {}
for s in tqdm(all_amber, desc="AMBER MLLM inference"):
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
    generated = out[:, inputs["input_ids"].shape[-1]:]
    amber_mllm_outputs[s.sample_id] = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

non_empty = sum(1 for v in amber_mllm_outputs.values() if v)
_log.info("AMBER MLLM done: %d/%d non-empty", non_empty, len(amber_mllm_outputs))

del model, processor
import gc; gc.collect(); torch.cuda.empty_cache()
_log.info("LLaVA 1.5 freed.")

# %%
# ── Cell 6: RawMLLM baseline ───────────────────────────────
from relcheck_v3.eval.harness_metrics import amber_discriminative_metrics

def _extract_yesno(text: str) -> str:
    t = text.strip().lower()
    if t.startswith("yes"):
        return "yes"
    if t.startswith("no"):
        return "no"
    if "yes" in t:
        return "yes"
    return "no"

amber_raw_results: dict[str, dict] = {}
for subtask_name, samples in amber_subtasks.items():
    preds = [_extract_yesno(amber_mllm_outputs[s.sample_id]) for s in samples]
    gts = [s.label for s in samples]
    amber_raw_results[subtask_name] = amber_discriminative_metrics(preds, gts)

_log.info("AMBER RawMLLM metrics computed.")

# %%
# ── Cell 7: Woodpecker + RelCheckFull correction ───────────
from pathlib import Path
from relcheck_v3.claim_generation.config import ClaimGenConfig
from relcheck_v3.claim_generation.pipeline import ClaimGenerationPipeline
from relcheck_v3.systems.relcheck_full import RelCheckFull
from relcheck_v3.eval.answer_extractor import AnswerExtractor
from relcheck_v3.mllm.cache import InferenceCache
import openai

resolved_key = os.environ.get("OPENAI_API_KEY", OPENAI_API_KEY)

wp_pipeline = ClaimGenerationPipeline(ClaimGenConfig(
    openai_api_key=resolved_key,
    gpt_model_id="gpt-5.4-mini",
    detector_config=GDINO_CONFIG,
    detector_model_path=GDINO_CHECKPOINT,
))
wp_client = openai.OpenAI(api_key=resolved_key)

full_system = RelCheckFull(
    openai_api_key=resolved_key,
    corrector_model="gpt-5.4",
    gdino_config=GDINO_CONFIG,
    gdino_checkpoint=GDINO_CHECKPOINT,
    reltr_checkpoint=RELTR_CHECKPOINT,
    cache_dir=f"{CACHE_DIR}/systems/full/",
)

judge = AnswerExtractor(
    openai_api_key=resolved_key,
    model="gpt-5.4-mini",
    cache=InferenceCache(Path(f"{CACHE_DIR}/answers/")),
)

_log.info("Correction systems ready.")

amber_wp_results: dict[str, dict] = {}
amber_full_results: dict[str, dict] = {}
amber_diagnostics: list[dict] = []

for subtask_name, samples in amber_subtasks.items():
    wp_preds = []
    full_preds = []

    for s in tqdm(samples, desc=f"AMBER WP+Full {subtask_name}"):
        mllm_out = amber_mllm_outputs[s.sample_id]
        diag = {
            "sample_id": s.sample_id, "subtask": subtask_name,
            "image_path": s.image_path, "question": s.question,
            "ground_truth": s.label, "mllm_output": mllm_out,
            "raw_answer": _extract_yesno(mllm_out),
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
            wp_preds.append(wp_answer)
        except Exception as e:
            _log.error("AMBER WP failed %s: %s", s.sample_id, e)
            wp_answer = _extract_yesno(mllm_out)
            diag["wp_corrected"] = mllm_out
            diag["wp_answer"] = wp_answer
            wp_preds.append(wp_answer)

        # RelCheckFull
        try:
            full_corrected = full_system.correct(s.image_path, mllm_out)
            diag["full_corrected"] = full_corrected
            diag["full_kb"] = full_system.last_kb_text
            full_answer = judge.extract_yesno(full_corrected, s.question)
            diag["full_answer"] = full_answer
            full_preds.append(full_answer)
        except Exception as e:
            _log.error("AMBER Full failed %s: %s", s.sample_id, e)
            full_answer = _extract_yesno(mllm_out)
            diag["full_corrected"] = mllm_out
            diag["full_answer"] = full_answer
            full_preds.append(full_answer)

        amber_diagnostics.append(diag)

    gts = [s.label for s in samples]
    amber_wp_results[subtask_name] = amber_discriminative_metrics(wp_preds, gts)
    amber_full_results[subtask_name] = amber_discriminative_metrics(full_preds, gts)
    _log.info("AMBER %s done — WP F1: %.3f, Full F1: %.3f",
              subtask_name,
              amber_wp_results[subtask_name]["f1"],
              amber_full_results[subtask_name]["f1"])

import json as _json
with open(f"{RESULTS_DIR}/amber_diagnostics.json", "w") as f:
    _json.dump(amber_diagnostics, f, indent=2)
_log.info("Saved %d AMBER diagnostics", len(amber_diagnostics))

# %%
# ── Cell 8: Results table ──────────────────────────────────
import pandas as pd

split_names = {"de": "Existence", "da": "Attribute", "dr": "Relation"}

rows = []
for subtask_code in ["de", "da", "dr"]:
    if subtask_code in amber_raw_results:
        name = split_names[subtask_code]
        rows.append({"subtask": name, "system": "RawMLLM", **amber_raw_results[subtask_code]})
        rows.append({"subtask": name, "system": "Woodpecker", **amber_wp_results[subtask_code]})
        rows.append({"subtask": name, "system": "RelCheckFull", **amber_full_results[subtask_code]})

df = pd.DataFrame(rows)

print("\n=== AMBER Discriminative F1 (LLaVA 1.5 7B) ===")
pivot_f1 = df.pivot(index="system", columns="subtask", values="f1")
pivot_f1 = pivot_f1.reindex(["RawMLLM", "Woodpecker", "RelCheckFull"])
pivot_f1 = pivot_f1[["Existence", "Attribute", "Relation"]]
print(pivot_f1.to_markdown())

print("\n=== AMBER Discriminative Accuracy ===")
pivot_acc = df.pivot(index="system", columns="subtask", values="accuracy")
pivot_acc = pivot_acc.reindex(["RawMLLM", "Woodpecker", "RelCheckFull"])
pivot_acc = pivot_acc[["Existence", "Attribute", "Relation"]]
print(pivot_acc.to_markdown())

print("\n=== Detailed Results ===")
for subtask_code, name in split_names.items():
    sub_df = df[df["subtask"] == name][["system", "accuracy", "precision", "recall", "f1"]]
    print(f"\n{name.upper()}:")
    print(sub_df.to_markdown(index=False))

pivot_f1.to_csv(f"{RESULTS_DIR}/amber_f1.csv")
pivot_acc.to_csv(f"{RESULTS_DIR}/amber_acc.csv")
_log.info("AMBER results saved to %s", RESULTS_DIR)

# %%
# ── Cell 9: Disagreement analysis ──────────────────────────
print("=" * 80)
print("AMBER: WOODPECKER CORRECT BUT RELCHECK WRONG")
print("=" * 80)
for d in amber_diagnostics:
    gt = d["ground_truth"]
    wp_ok = d.get("wp_answer") == gt
    full_ok = d.get("full_answer") == gt
    if wp_ok and not full_ok:
        print(f"\n{'─'*60}")
        print(f"[{split_names.get(d['subtask'], d['subtask'])}] {d['sample_id']}")
        print(f"  Q: {d['question']}")
        print(f"  GT: {gt}  |  Raw: {d['raw_answer']}  |  WP: {d.get('wp_answer')} ✓  |  Full: {d.get('full_answer')} ✗")
        print(f"  MLLM:          {d['mllm_output']}")
        print(f"  WP corrected:  {d.get('wp_corrected', 'N/A')}")
        print(f"  Full corrected: {d.get('full_corrected', 'N/A')}")
        if d.get("full_kb"):
            print(f"  KB:\n{d['full_kb']}")

print("\n" + "=" * 80)
print("AMBER: RELCHECK CORRECT BUT WOODPECKER WRONG")
print("=" * 80)
for d in amber_diagnostics:
    gt = d["ground_truth"]
    wp_ok = d.get("wp_answer") == gt
    full_ok = d.get("full_answer") == gt
    if full_ok and not wp_ok:
        print(f"\n{'─'*60}")
        print(f"[{split_names.get(d['subtask'], d['subtask'])}] {d['sample_id']}")
        print(f"  Q: {d['question']}")
        print(f"  GT: {gt}  |  Raw: {d['raw_answer']}  |  WP: {d.get('wp_answer')} ✗  |  Full: {d.get('full_answer')} ✓")
        print(f"  MLLM:          {d['mllm_output']}")
        print(f"  Full corrected: {d.get('full_corrected', 'N/A')}")
