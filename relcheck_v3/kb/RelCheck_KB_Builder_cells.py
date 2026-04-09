# ============================================================
# RelCheck v3 — Knowledge Base Builder
# ============================================================
# Copy-paste each cell block into Google Colab.
# Cells are separated by: # ── CELL N ──
#
# Builds the full 3-layer Knowledge Base (CLAIM + GEOM + SCENE)
# from claim generation output. Run AFTER the claim generation
# pipeline (RelCheck_Claim_Gen_cells.py).
#
# CLAIM and GEOM require no extra GPU models beyond claim gen.
# SCENE (RelTR) is optional and gated by ENABLE_RELTR.


# ── CELL 0 — Config ─────────────────────────────────────────
# Paths from claim generation (should match RelCheck_Claim_Gen_cells.py)
CLAIM_GEN_DIR = "/content/drive/MyDrive/RelCheck_Data/claim_generation"
CLAIM_GEN_JSONL = f"{CLAIM_GEN_DIR}/output.jsonl"

# Image directory (same as claim gen)
IMAGE_DIR = "/content/coco_val2014/val2014"

# Output directory for KB results
KB_SAVE_DIR = "/content/drive/MyDrive/RelCheck_Data/kb"

# RelTR toggle — set True to enable scene graph layer
ENABLE_RELTR = False
RELTR_CHECKPOINT = "/content/drive/MyDrive/RelCheck_Data/checkpoint0149.pth"

MAX_SAMPLES = None  # None = all


# ── CELL 1 — Setup ──────────────────────────────────────────
import os, sys, json

from google.colab import drive
drive.mount("/content/drive")
os.makedirs(KB_SAVE_DIR, exist_ok=True)

REPO_DIR = "/content/RelCheck"
if not os.path.exists(os.path.join(REPO_DIR, ".git")):
    os.system(f"git clone https://github.com/yadomkar/RelCheck.git {REPO_DIR}")
else:
    os.system(f"cd {REPO_DIR} && git pull -q")
sys.path.insert(0, REPO_DIR)

# Apply RelTR config
import relcheck_v3.reltr.config as reltr_cfg
reltr_cfg.ENABLE_RELTR = ENABLE_RELTR
reltr_cfg.RELTR_CHECKPOINT_PATH = RELTR_CHECKPOINT

if ENABLE_RELTR:
    # Clone RelTR repo for model code
    RELTR_DIR = "/content/RelTR"
    if not os.path.exists(RELTR_DIR):
        os.system(f"git clone https://github.com/yrcong/RelTR.git {RELTR_DIR}")
    print(f"RelTR enabled — checkpoint: {RELTR_CHECKPOINT}")
else:
    print("RelTR disabled — SCENE layer will be empty")

print("Setup complete.")


# ── CELL 2 — Load Claim Generation Output ───────────────────
from relcheck_v3.claim_generation.models import (
    SampleResult, ObjectAnswer, VisualKnowledgeBase,
)

claim_results: list[SampleResult] = []

with open(CLAIM_GEN_JSONL, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        try:
            result = SampleResult.model_validate(rec)
            if result.success:
                claim_results.append(result)
        except Exception as e:
            print(f"Skipping malformed record: {e}")

print(f"Loaded {len(claim_results)} successful claim generation results")
if claim_results:
    print(f"  First: {claim_results[0].image_id} — {claim_results[0].ref_cap[:60]}...")

if MAX_SAMPLES is not None:
    claim_results = claim_results[:MAX_SAMPLES]
    print(f"  Capped to {MAX_SAMPLES} samples")


# ── CELL 3 — Build Knowledge Bases ──────────────────────────
import time
from PIL import Image
from tqdm import tqdm
from relcheck_v3.kb import build_kb, KnowledgeBase

kb_results: list[tuple[SampleResult, KnowledgeBase]] = []
errors = 0

t0 = time.time()
for r in tqdm(claim_results, desc="Building KBs"):
    try:
        # Load image for RelTR (only needed when enabled)
        pil_image = None
        if ENABLE_RELTR:
            img_path = os.path.join(IMAGE_DIR, f"COCO_val2014_{int(r.image_id):012d}.jpg")
            if os.path.exists(img_path):
                pil_image = Image.open(img_path).convert("RGB")

        kb = build_kb(
            vkb=r.visual_knowledge_base,
            object_answers=r.object_answers,
            image=pil_image,
        )
        kb_results.append((r, kb))
    except Exception as e:
        errors += 1
        print(f"[{r.image_id}] KB build failed: {e}")

elapsed = time.time() - t0
print(f"\nBuilt {len(kb_results)} KBs in {elapsed:.1f}s ({errors} errors)")


# ── CELL 4 — Save KB Output ─────────────────────────────────
import pandas as pd

# Save full KB text per sample as JSONL
kb_jsonl_path = f"{KB_SAVE_DIR}/kb_output.jsonl"
with open(kb_jsonl_path, "w") as f:
    for r, kb in kb_results:
        rec = {
            "image_id": r.image_id,
            "ref_cap": r.ref_cap,
            "kb_text": kb.format(),
            "n_spatial_facts": len(kb.spatial_facts),
            "n_scene_triples": len(kb.scene_graph),
            "spatial_facts": kb.spatial_facts,
            "scene_graph": kb.scene_graph,
        }
        f.write(json.dumps(rec) + "\n")

print(f"Saved {len(kb_results)} KB records to {kb_jsonl_path}")

# Summary CSV
rows = []
for r, kb in kb_results:
    rows.append({
        "image_id": r.image_id,
        "ref_cap": r.ref_cap,
        "n_count_claims": len(kb.claims.count_claims),
        "n_specific_claims": len(kb.claims.specific_claims),
        "n_overall_claims": len(kb.claims.overall_claims),
        "n_spatial_facts": len(kb.spatial_facts),
        "n_scene_triples": len(kb.scene_graph),
        "kb_text": kb.format(),
    })

df = pd.DataFrame(rows)
csv_path = f"{KB_SAVE_DIR}/kb_summary.csv"
df.to_csv(csv_path, index=False)
print(f"Saved summary CSV to {csv_path}")
print(f"\n{df.describe()}")


# ── CELL 5 — Display KB Examples ────────────────────────────
import matplotlib.pyplot as plt

show_n = min(5, len(kb_results))

for r, kb in kb_results[:show_n]:
    print(f"{'='*60}")
    print(f"Image ID: {r.image_id}")
    print(f"Ref-Cap:  {r.ref_cap}")
    print()
    print(kb.format())

    # Show image
    img_path = os.path.join(IMAGE_DIR, f"COCO_val2014_{int(r.image_id):012d}.jpg")
    if os.path.exists(img_path):
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.imshow(Image.open(img_path))
        ax.set_title(f"ID: {r.image_id}")
        ax.axis("off")
        plt.tight_layout()
        plt.show()
    print()


# ── CELL 6 — KB Statistics ──────────────────────────────────
if kb_results:
    spatial_counts = [len(kb.spatial_facts) for _, kb in kb_results]
    scene_counts = [len(kb.scene_graph) for _, kb in kb_results]
    claim_counts = [
        len(kb.claims.count_claims) + len(kb.claims.specific_claims) + len(kb.claims.overall_claims)
        for _, kb in kb_results
    ]

    print(f"KB Statistics ({len(kb_results)} samples):")
    print(f"  Claims per sample:  avg={sum(claim_counts)/len(claim_counts):.1f}, "
          f"min={min(claim_counts)}, max={max(claim_counts)}")
    print(f"  Spatial facts:      avg={sum(spatial_counts)/len(spatial_counts):.1f}, "
          f"min={min(spatial_counts)}, max={max(spatial_counts)}")
    print(f"  Scene triples:      avg={sum(scene_counts)/len(scene_counts):.1f}, "
          f"min={min(scene_counts)}, max={max(scene_counts)}")
