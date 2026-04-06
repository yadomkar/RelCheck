# ============================================================
# RelCheck v2 — COCO Controlled Evaluation with RelTR
# ============================================================
# Copy-paste each cell block into Google Colab.
# Cells are separated by: # ── CELL N ──
#
# All metric logic lives in relcheck_v2.coco_eval.
# This file is thin orchestration only.


# ── CELL 0 — Config ─────────────────────────────────────────
N_IMAGES = 200
SEED = 42
TOGETHER_API_KEY = ""  # <-- paste your key
SAVE_DIR = "/content/drive/MyDrive/RelCheck_Data/coco_reltr_eval"
COCO_DIR = "/content/coco_val2017"
CAPTIONER = "llava"
INJECTION_TYPES = ["spatial_flip", "entity_swap", "action_swap", "count_inflation"]


# ── CELL 1 — Setup ──────────────────────────────────────────
# !pip install together Pillow requests transformers rapidfuzz spacy \
#     tenacity json-repair pysbd nltk pycocotools -q
# !python -m spacy download en_core_web_sm -q

import os, sys, json, time, random
from pathlib import Path
from collections import Counter
from PIL import Image
from google.colab import drive

drive.mount("/content/drive")
os.makedirs(SAVE_DIR, exist_ok=True)
os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY

import nltk
for pkg in ("wordnet", "omw-1.4", "punkt", "punkt_tab"):
    nltk.download(pkg, quiet=True)

REPO_DIR = "/content/RelCheck"
if not os.path.exists(os.path.join(REPO_DIR, ".git")):
    os.system(f"git clone https://github.com/yadomkar/RelCheck.git {REPO_DIR}")
else:
    os.system(f"cd {REPO_DIR} && git pull -q")
sys.path.insert(0, REPO_DIR)

if not os.path.exists("/content/RelTR"):
    os.system("git clone https://github.com/yrcong/RelTR.git /content/RelTR")
sys.path.insert(0, "/content/RelTR")

# RelTR checkpoint
RELTR_CKPT = "/content/drive/MyDrive/RelCheck_Data/RelTR/ckpt/checkpoint0149.pth"
os.makedirs(os.path.dirname(RELTR_CKPT), exist_ok=True)
if not os.path.exists(RELTR_CKPT):
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"])
    subprocess.run([sys.executable, "-m", "gdown",
        "https://drive.google.com/uc?id=1F_B4v6oqKpXKdD9YGz2qGZFsGQFDL5JY",
        "-O", RELTR_CKPT])

# COCO val2017
if not os.path.exists(f"{COCO_DIR}/val2017"):
    os.makedirs(COCO_DIR, exist_ok=True)
    os.system(f"wget -q http://images.cocodataset.org/zips/val2017.zip -O {COCO_DIR}/val2017.zip")
    os.system(f"unzip -q {COCO_DIR}/val2017.zip -d {COCO_DIR}")
if not os.path.exists(f"{COCO_DIR}/annotations"):
    os.system(f"wget -q http://images.cocodataset.org/annotations/annotations_trainval2017.zip "
              f"-O {COCO_DIR}/annotations.zip")
    os.system(f"unzip -q {COCO_DIR}/annotations.zip -d {COCO_DIR}")

from relcheck_v2.api import init_client
from relcheck_v2.models import get_gdino, get_llava, DEVICE
from relcheck_v2.kb import build_visual_kb
from relcheck_v2.captioning import caption_image
from relcheck_v2.correction import correct_long_caption
from relcheck_v2.correction._metrics import MetricsCollector
from relcheck_v2.reltr import (
    coco_categories_covered, coco_has_reltr_predicate_coverage,
)
from relcheck_v2.spatial import compute_spatial_facts, parse_spatial_facts
from relcheck_v2.coco_eval import (
    compute_r_chair, aggregate_r_chair,
    hallucination_removed, collateral_damage,
    compute_bleu4, compute_meteor,
    build_ablation_table,
)
import relcheck_v2.config as cfg

init_client(TOGETHER_API_KEY)
get_gdino()
if CAPTIONER == "llava":
    get_llava()
print(f"Setup complete. Device: {DEVICE}")


# ── CELL 2 — Image Selection with RelTR Vocab Filter ────────
from pycocotools.coco import COCO

coco = COCO(f"{COCO_DIR}/annotations/instances_val2017.json")
coco_caps = COCO(f"{COCO_DIR}/annotations/captions_val2017.json")

all_img_ids = sorted(coco.getImgIds())
random.seed(SEED)
random.shuffle(all_img_ids)

selected: list[int] = []
fail_reasons: Counter = Counter()

for img_id in all_img_ids:
    if len(selected) >= N_IMAGES:
        break

    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    if len(anns) < 2:
        fail_reasons["<2 objects"] += 1
        continue

    cat_ids = list({a["category_id"] for a in anns})
    cats = [coco.loadCats([c])[0]["name"] for c in cat_ids]
    if not coco_categories_covered(cats):
        fail_reasons["object_coverage"] += 1
        continue

    img_info = coco.loadImgs(img_id)[0]
    w, h = img_info["width"], img_info["height"]
    det_tuples = []
    for a in anns:
        bx, by, bw, bh = a["bbox"]
        cat_name = coco.loadCats([a["category_id"]])[0]["name"]
        det_tuples.append((cat_name, 1.0, [bx / w, by / h, (bx + bw) / w, (by + bh) / h]))

    spatial_facts = compute_spatial_facts(det_tuples)
    spatial_rels = parse_spatial_facts(spatial_facts)

    # Predicate coverage is a soft filter — log but don't reject.
    # RelTR's value is in semantic relations (riding, holding, etc.)
    # which can't be derived from COCO bbox geometry alone.
    if not coco_has_reltr_predicate_coverage(spatial_rels):
        fail_reasons["no_predicate_overlap"] += 1
        # Still select — object coverage is the hard requirement

    selected.append(img_id)

print(f"Selected {len(selected)}/{N_IMAGES} images")
print(f"Filter stats: {dict(fail_reasons)}")
with open(f"{SAVE_DIR}/selected_images.json", "w") as f:
    json.dump(selected, f)


# ── CELL 3 — Caption Generation ─────────────────────────────
CAPTIONS_PATH = f"{SAVE_DIR}/captions.json"
captions: dict[str, str] = {}
if os.path.exists(CAPTIONS_PATH):
    with open(CAPTIONS_PATH) as f:
        captions = json.load(f)
    print(f"Loaded {len(captions)} cached captions")

pil_images: dict[str, Image.Image] = {}
new = 0
for img_id in selected:
    img_info = coco.loadImgs(img_id)[0]
    pil = Image.open(f"{COCO_DIR}/val2017/{img_info['file_name']}").convert("RGB")
    pil_images[str(img_id)] = pil

    if str(img_id) not in captions:
        cap = caption_image(pil, captioner=CAPTIONER)
        if cap:
            captions[str(img_id)] = cap
            new += 1
            if new % 10 == 0:
                with open(CAPTIONS_PATH, "w") as f:
                    json.dump(captions, f, indent=2)
                print(f"  Generated {new} captions...")

if new > 0:
    with open(CAPTIONS_PATH, "w") as f:
        json.dump(captions, f, indent=2)
print(f"{new} new, {len(captions)} total captions")


# ── CELL 4 — Hallucination Injection ────────────────────────
INJECT_PATH = f"{SAVE_DIR}/injected.json"
injected_data: dict[str, dict] = {}

if os.path.exists(INJECT_PATH):
    with open(INJECT_PATH) as f:
        injected_data = json.load(f)
    print(f"Loaded {len(injected_data)} cached injections")
else:
    random.seed(SEED)
    _FAKE_ACTIONS = ["flying over", "swimming in", "climbing on", "dancing with"]

    for img_id_int in selected:
        img_id = str(img_id_int)
        cap = captions.get(img_id)
        if not cap:
            continue

        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id_int))
        img_info = coco.loadImgs(img_id_int)[0]
        w, h = img_info["width"], img_info["height"]

        bbox_lookup: dict[str, list[list[float]]] = {}
        for a in anns:
            cat = coco.loadCats([a["category_id"]])[0]["name"]
            bx, by, bw, bh = a["bbox"]
            bbox_lookup.setdefault(cat, []).append(
                [bx / w, by / h, (bx + bw) / w, (by + bh) / h]
            )

        cats = list(bbox_lookup.keys())
        if len(cats) < 2:
            continue

        inj_type = random.choice(INJECTION_TYPES)

        if inj_type == "spatial_flip":
            s, o = random.sample(cats, 2)
            stmt = f"The {s} is to the right of the {o}."
        elif inj_type == "entity_swap":
            real, fake = random.sample(cats, 2)
            stmt = f"A {fake} is visible near the {real}."
        elif inj_type == "action_swap":
            ent = random.choice(cats)
            other = random.choice(cats)
            stmt = f"The {ent} is {random.choice(_FAKE_ACTIONS)} the {other}."
        elif inj_type == "count_inflation":
            ent = random.choice(cats)
            inflated = len(bbox_lookup[ent]) + random.randint(3, 7)
            stmt = f"There are {inflated} {ent}s in the image."
        else:
            continue

        sep = " " if cap.rstrip().endswith(".") else ". "
        injected_data[img_id] = {
            "original_caption": cap,
            "corrupted_caption": cap.rstrip() + sep + stmt,
            "injected_statement": stmt,
            "injection_type": inj_type,
            "coco_annotations": [
                {"category_name": coco.loadCats([a["category_id"]])[0]["name"],
                 "bbox": a["bbox"]}
                for a in anns
            ],
            "image_width": w,
            "image_height": h,
        }

    with open(INJECT_PATH, "w") as f:
        json.dump(injected_data, f, indent=2)

print(f"Injected: {len(injected_data)} images")
print(f"By type: {dict(Counter(d['injection_type'] for d in injected_data.values()))}")


# ── CELL 5 — KB Construction ────────────────────────────────
KB_PATH = f"{SAVE_DIR}/knowledge_bases.json"

if os.path.exists(KB_PATH):
    with open(KB_PATH) as f:
        knowledge_bases = json.load(f)
    print(f"Loaded {len(knowledge_bases)} cached KBs")
else:
    cfg.ENABLE_RELTR = True
    knowledge_bases = {}
    for idx, (img_id, inj) in enumerate(injected_data.items()):
        pil = pil_images.get(img_id)
        if pil is None:
            continue
        t0 = time.time()
        kb = build_visual_kb(pil, inj["original_caption"], max_detections=20)
        knowledge_bases[img_id] = kb.to_dict()
        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{len(injected_data)}] ({time.time()-t0:.1f}s) "
                  f"sg={len(kb.scene_graph)} triples")
            with open(KB_PATH, "w") as f:
                json.dump(knowledge_bases, f)
    with open(KB_PATH, "w") as f:
        json.dump(knowledge_bases, f)
    cfg.ENABLE_RELTR = False
    print(f"Built KBs for {len(knowledge_bases)} images")

sg_counts = [len(kb.get("scene_graph", [])) for kb in knowledge_bases.values()]
print(f"Scene graph triples: mean={sum(sg_counts)/max(len(sg_counts),1):.1f}, "
      f"total={sum(sg_counts)}")


# ── CELL 6 — Baseline Run (ENABLE_RELTR=False) ──────────────
BASELINE_PATH = f"{SAVE_DIR}/corrected_baseline.json"
cfg.ENABLE_RELTR = False
mc_baseline = MetricsCollector()

if os.path.exists(BASELINE_PATH):
    with open(BASELINE_PATH) as f:
        baseline_data = json.load(f)
    print(f"Loaded {len(baseline_data)} cached baseline corrections")
else:
    baseline_data = {}
    for idx, (img_id, inj) in enumerate(injected_data.items()):
        pil = pil_images.get(img_id)
        if pil is None:
            continue
        kb = dict(knowledge_bases.get(img_id, {}))
        kb["scene_graph"] = []  # strip scene graph for baseline
        result = correct_long_caption(
            img_id, inj["corrupted_caption"], kb,
            pil_image=pil, metrics=mc_baseline,
        )
        baseline_data[img_id] = {
            "corrected": result.corrected,
            "errors": [e.triple.claim for e in result.errors],
            "edit_rate": result.edit_rate,
            "status": result.status,
        }
        if (idx + 1) % 20 == 0:
            print(f"  Baseline [{idx+1}/{len(injected_data)}]")
            with open(BASELINE_PATH, "w") as f:
                json.dump(baseline_data, f, indent=2)
    with open(BASELINE_PATH, "w") as f:
        json.dump(baseline_data, f, indent=2)
    mc_baseline.save(f"{SAVE_DIR}/path_logs_baseline.json")
    n_mod = sum(1 for d in baseline_data.values() if d["status"] == "modified")
    print(f"Baseline done. {n_mod}/{len(baseline_data)} modified.")


# ── CELL 7 — RelTR Run (ENABLE_RELTR=True) ──────────────────
RELTR_PATH = f"{SAVE_DIR}/corrected_reltr.json"
cfg.ENABLE_RELTR = True
mc_reltr = MetricsCollector()

if os.path.exists(RELTR_PATH):
    with open(RELTR_PATH) as f:
        reltr_data = json.load(f)
    print(f"Loaded {len(reltr_data)} cached RelTR corrections")
    cfg.ENABLE_RELTR = False
else:
    reltr_data = {}
    for idx, (img_id, inj) in enumerate(injected_data.items()):
        pil = pil_images.get(img_id)
        if pil is None:
            continue
        kb = knowledge_bases.get(img_id, {})
        result = correct_long_caption(
            img_id, inj["corrupted_caption"], kb,
            pil_image=pil, metrics=mc_reltr,
        )
        reltr_data[img_id] = {
            "corrected": result.corrected,
            "errors": [e.triple.claim for e in result.errors],
            "edit_rate": result.edit_rate,
            "status": result.status,
        }
        if (idx + 1) % 20 == 0:
            print(f"  RelTR [{idx+1}/{len(injected_data)}]")
            with open(RELTR_PATH, "w") as f:
                json.dump(reltr_data, f, indent=2)
    with open(RELTR_PATH, "w") as f:
        json.dump(reltr_data, f, indent=2)
    mc_reltr.save(f"{SAVE_DIR}/path_logs_reltr.json")
    cfg.ENABLE_RELTR = False
    n_mod = sum(1 for d in reltr_data.values() if d["status"] == "modified")
    print(f"RelTR done. {n_mod}/{len(reltr_data)} modified.")


# ── CELL 8 — R-CHAIR Computation ────────────────────────────
r_chair_results: dict[str, list[dict]] = {
    "original": [], "corrupted": [], "baseline": [], "reltr": [],
}

for img_id, inj in injected_data.items():
    anns = inj["coco_annotations"]
    w, h = inj["image_width"], inj["image_height"]

    r_chair_results["original"].append(compute_r_chair(inj["original_caption"], anns, w, h))
    r_chair_results["corrupted"].append(compute_r_chair(inj["corrupted_caption"], anns, w, h))
    if img_id in baseline_data:
        r_chair_results["baseline"].append(
            compute_r_chair(baseline_data[img_id]["corrected"], anns, w, h))
    if img_id in reltr_data:
        r_chair_results["reltr"].append(
            compute_r_chair(reltr_data[img_id]["corrected"], anns, w, h))

for version, results in r_chair_results.items():
    agg = aggregate_r_chair(results)
    print(f"{version:>12}: R-CHAIR_i={agg['r_chair_i']:.3f}  "
          f"R-CHAIR_s={agg['r_chair_s']:.3f}  (n={agg['n']})")

with open(f"{SAVE_DIR}/r_chair_results.json", "w") as f:
    json.dump(r_chair_results, f, indent=2)


# ── CELL 9 — Per-Image Metrics ───────────────────────────────
metrics_per_image: list[dict] = []

for img_id, inj in injected_data.items():
    row = {"img_id": img_id, "injection_type": inj["injection_type"]}
    orig = inj["original_caption"]
    stmt = inj["injected_statement"]

    for run_name, run_data in [("baseline", baseline_data), ("reltr", reltr_data)]:
        if img_id not in run_data:
            continue
        corrected = run_data[img_id]["corrected"]
        row[f"removal_{run_name}"] = hallucination_removed(
            inj["corrupted_caption"], corrected, stmt)
        row[f"collateral_{run_name}"] = collateral_damage(orig, corrected)
        row[f"bleu4_{run_name}"] = compute_bleu4(orig, corrected)
        row[f"meteor_{run_name}"] = compute_meteor(orig, corrected)

    metrics_per_image.append(row)

# CLIPScore (optional)
try:
    import torch
    from transformers import CLIPProcessor, CLIPModel

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    for row in metrics_per_image:
        pil = pil_images.get(row["img_id"])
        if pil is None:
            continue
        for run_name, run_data in [("baseline", baseline_data), ("reltr", reltr_data)]:
            if row["img_id"] not in run_data:
                continue
            inputs = clip_proc(
                text=[run_data[row["img_id"]]["corrected"]], images=pil,
                return_tensors="pt", padding=True,
            ).to(DEVICE)
            with torch.no_grad():
                outputs = clip_model(**inputs)
            row[f"clipscore_{run_name}"] = outputs.logits_per_image.item()
    print("CLIPScore computed.")
except Exception as e:
    print(f"CLIPScore skipped: {e}")

with open(f"{SAVE_DIR}/metrics_per_image.json", "w") as f:
    json.dump(metrics_per_image, f, indent=2)
print(f"Computed metrics for {len(metrics_per_image)} images")


# ── CELL 10 — LLM Judge ─────────────────────────────────────
from relcheck_v2.api import llm_call

JUDGE_PATH = f"{SAVE_DIR}/llm_judge.json"
judge_results: dict[str, dict[str, str]] = {}

if os.path.exists(JUDGE_PATH):
    with open(JUDGE_PATH) as f:
        judge_results = json.load(f)
    print(f"Loaded cached judge results")
else:
    for run_name, run_data in [("baseline", baseline_data), ("reltr", reltr_data)]:
        judge_results[run_name] = {}
        for idx, (img_id, inj) in enumerate(injected_data.items()):
            if img_id not in run_data:
                continue
            corrupted = inj["corrupted_caption"]
            corrected = run_data[img_id]["corrected"]
            if corrupted == corrected:
                judge_results[run_name][img_id] = "tie"
                continue

            prompt = (
                f"Compare these two captions for factual accuracy.\n"
                f'Caption A (corrupted): "{corrupted}"\n'
                f'Caption B (corrected): "{corrected}"\n\n'
                f"Which is more factually accurate? Answer exactly: A_WINS, B_WINS, or TIE"
            )
            raw = llm_call([{"role": "user", "content": prompt}],
                           max_tokens=20, temperature=0.0)
            if raw and "B_WINS" in raw.upper():
                judge_results[run_name][img_id] = "corrected_wins"
            elif raw and "A_WINS" in raw.upper():
                judge_results[run_name][img_id] = "corrupted_wins"
            else:
                judge_results[run_name][img_id] = "tie"

            if (idx + 1) % 20 == 0:
                print(f"  Judge [{run_name}] {idx+1}/{len(injected_data)}")
                with open(JUDGE_PATH, "w") as f:
                    json.dump(judge_results, f, indent=2)

    with open(JUDGE_PATH, "w") as f:
        json.dump(judge_results, f, indent=2)

for run_name in ("baseline", "reltr"):
    jr = judge_results.get(run_name, {})
    wins = sum(1 for v in jr.values() if v == "corrected_wins")
    print(f"{run_name}: corrected wins {wins}/{len(jr)} ({100*wins/max(len(jr),1):.1f}%)")


# ── CELL 11 — Ablation Table ────────────────────────────────
import pandas as pd

rows = build_ablation_table(
    metrics_per_image=metrics_per_image,
    r_chair_by_run=r_chair_results,
    judge_by_run=judge_results,
    injection_types=INJECTION_TYPES,
)

ablation_df = pd.DataFrame(rows)
print(ablation_df.to_string(index=False, float_format="%.3f"))
ablation_df.to_csv(f"{SAVE_DIR}/ablation_table.csv", index=False)
print(f"\nSaved to {SAVE_DIR}/ablation_table.csv")


# ── CELL 12 — Save All Results ──────────────────────────────
final_results = {
    "config": {"n_images": N_IMAGES, "seed": SEED, "captioner": CAPTIONER},
    "r_chair": {
        v: aggregate_r_chair(r)
        for v, r in r_chair_results.items() if r
    },
    "judge_summary": {
        run: {
            "corrected_wins": sum(1 for v in jr.values() if v == "corrected_wins"),
            "corrupted_wins": sum(1 for v in jr.values() if v == "corrupted_wins"),
            "ties": sum(1 for v in jr.values() if v == "tie"),
            "total": len(jr),
        }
        for run, jr in judge_results.items()
    },
}

with open(f"{SAVE_DIR}/final_results.json", "w") as f:
    json.dump(final_results, f, indent=2)

print("All results saved to:", SAVE_DIR)
print(json.dumps(final_results, indent=2))
