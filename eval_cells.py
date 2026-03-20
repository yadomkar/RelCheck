"""
RelCheck — New Evaluation Cells for Colab
==========================================
Copy each cell block into RelCheck_Master.ipynb.
Run AFTER Section 6 (full pipeline) completes.

Cell 0: Shared helper — compute_rpope_metrics()
Cell A: LLM-Judge R-POPE (caption-grounded) — the critical metric
Cell B: CLIPScore delta
Cell C: Filtered R-POPE (corrected subset only)
Cell D: LLM Direct-Correction Baseline (B3) — FAST pivot test
Cell E: Quick head-to-head: B3 vs RelCheck via LLM-judge (50 images)
"""


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 0 — Shared helper: compute_rpope_metrics()
# ═══════════════════════════════════════════════════════════════════════════════
# Run this cell FIRST — Cells A, C, D, and E all depend on it.
# ═══════════════════════════════════════════════════════════════════════════════

def compute_rpope_metrics(preds: list[dict]) -> dict:
    """
    Compute standard R-POPE metrics from a list of {pred, gt} dicts.

    Args:
        preds: List of dicts with 'pred' and 'gt' keys, each 'yes' or 'no'.

    Returns:
        Dict with accuracy, precision, recall, f1, yes_ratio, tp, tn, fp, fn, total.
    """
    tp = sum(1 for p in preds if p['pred'] == 'yes' and p['gt'] == 'yes')
    tn = sum(1 for p in preds if p['pred'] == 'no'  and p['gt'] == 'no')
    fp = sum(1 for p in preds if p['pred'] == 'yes' and p['gt'] == 'no')
    fn = sum(1 for p in preds if p['pred'] == 'no'  and p['gt'] == 'yes')
    total = len(preds)

    accuracy  = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    yes_ratio = (tp + fp) / total if total > 0 else 0.0

    return {
        'accuracy':  round(accuracy, 4),
        'precision': round(precision, 4),
        'recall':    round(recall, 4),
        'f1':        round(f1, 4),
        'yes_ratio': round(yes_ratio, 4),
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'total': total,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CELL A — LLM-Judge R-POPE (caption-grounded evaluation)
# ═══════════════════════════════════════════════════════════════════════════════
# This is the KEY evaluation that shows RelCheck's value.
# Uses Llama-3.3-70B as a TEXT-ONLY judge — no image, only caption.
# If the corrected caption is more accurate, answers match R-Bench GT better.
#
# Runs on all three conditions: B1 (no correction), B2 (self-refine), RelCheck
# ═══════════════════════════════════════════════════════════════════════════════

import os
import json
import time
import pandas as pd
from tqdm import tqdm
import together

# --- Config ---
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY", "")
LLM_JUDGE_MODEL  = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
REPO_DIR         = "/content/RelCheck"
EVAL_DIR         = f"{REPO_DIR}/eval"

client = together.Together(api_key=TOGETHER_API_KEY)

# --- Load existing results CSVs ---
B1_CSV = f"{EVAL_DIR}/baseline_no_correction.csv"
B2_CSV = f"{EVAL_DIR}/baseline_self_refine.csv"
RC_CSV = f"{EVAL_DIR}/relcheck_results.csv"

df_b1 = pd.read_csv(B1_CSV)
df_b2 = pd.read_csv(B2_CSV)
df_rc = pd.read_csv(RC_CSV)

print(f"Loaded B1: {len(df_b1)} rows, B2: {len(df_b2)} rows, RC: {len(df_rc)} rows")

# --- LLM Judge Function ---
LLM_JUDGE_SYSTEM = """You are a factual judge. Given an image caption and a yes/no question, \
answer based ONLY on what the caption states. Do NOT use any external knowledge.

If the caption explicitly or implicitly supports the answer "yes", respond "yes".
If the caption contradicts the claim or does not mention it, respond "no".

Respond with ONLY "yes" or "no" — nothing else."""

LLM_JUDGE_USER = """Caption: "{caption}"

Question: {question}

Based solely on what the caption says, answer yes or no:"""


def llm_judge(caption: str, question: str, max_retries: int = 3) -> str:
    """
    Ask Llama-3.3-70B to answer a yes/no question using ONLY the caption.
    Returns 'yes' or 'no' (lowercase).
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=LLM_JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": LLM_JUDGE_SYSTEM},
                    {"role": "user",   "content": LLM_JUDGE_USER.format(
                        caption=caption, question=question
                    )},
                ],
                max_tokens=5,
                temperature=0.0,
            )
            answer = response.choices[0].message.content.strip().lower()
            # Normalize to yes/no
            if "yes" in answer:
                return "yes"
            elif "no" in answer:
                return "no"
            else:
                return "no"  # default if LLM gives garbage
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # exponential backoff
                continue
            print(f"  [LLM Judge] Failed after {max_retries} attempts: {e}")
            return "no"  # conservative default


# --- Run LLM Judge on all three conditions ---

def run_llm_judge_on_df(df, caption_col, label):
    """
    Run LLM-judge R-POPE on a DataFrame.

    Args:
        df: DataFrame with columns [image_id, question, gt, {caption_col}]
        caption_col: Name of the column containing the caption to judge
        label: Label for this condition (e.g., "B1", "B2", "RelCheck")

    Returns:
        (rows, preds) — list of detailed rows and list of {pred, gt} dicts
    """
    rows  = []
    preds = []
    errors = 0

    print(f"\n{'='*60}")
    print(f"  LLM-Judge R-POPE — {label}")
    print(f"  Caption column: {caption_col}")
    print(f"  Total questions: {len(df)}")
    print(f"{'='*60}")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"LLM-Judge {label}"):
        caption  = str(row[caption_col])
        question = str(row['question'])
        gt       = str(row['gt']).lower().strip()

        llm_pred = llm_judge(caption, question)

        rows.append({
            'image_id':       row.get('image_id', ''),
            'question':       question,
            'caption':        caption,
            'caption_source': label,
            'gt':             gt,
            'llm_pred':       llm_pred,
            'correct':        llm_pred == gt,
            'relation_type':  row.get('relation_type', ''),
        })
        preds.append({'pred': llm_pred, 'gt': gt})

    return rows, preds


# B1: No correction — use BLIP-2 caption
# The caption column name depends on your CSV. Check which column has the caption:
b1_caption_col = 'blip2_caption' if 'blip2_caption' in df_b1.columns else 'original_caption'
print(f"B1 caption column: {b1_caption_col}")
print(f"B1 columns: {list(df_b1.columns)}")

b1_rows, b1_preds = run_llm_judge_on_df(df_b1, b1_caption_col, "B1 (No Correction)")

# B2: Self-refine — use refined caption
b2_caption_col = 'refined_caption'
print(f"\nB2 caption column: {b2_caption_col}")
b2_rows, b2_preds = run_llm_judge_on_df(df_b2, b2_caption_col, "B2 (Self-Refine)")

# RelCheck: Use corrected caption
rc_caption_col = 'corrected_caption'
print(f"\nRC caption column: {rc_caption_col}")
rc_rows, rc_preds = run_llm_judge_on_df(df_rc, rc_caption_col, "RelCheck")


# --- Compute and display metrics ---

def print_metrics(metrics, label):
    print(f"\n{'='*55}")
    print(f"  {label} — R-POPE  [LLM-Judge evaluator]")
    print(f"{'='*55}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:14s}: {v:.4f}")
        else:
            print(f"  {k:14s}: {v}")


b1_metrics = compute_rpope_metrics(b1_preds)
b2_metrics = compute_rpope_metrics(b2_preds)
rc_metrics = compute_rpope_metrics(rc_preds)

print_metrics(b1_metrics, "B1 (No Correction)")
print_metrics(b2_metrics, "B2 (Self-Refine)")
print_metrics(rc_metrics, "RelCheck")


# --- Comparison table ---

print(f"\n{'='*70}")
print("  LLM-Judge R-POPE — COMPARISON TABLE")
print(f"{'='*70}")
print(f"  {'Metric':<14} {'B1 (No Corr)':>14} {'B2 (Self-Ref)':>14} {'RelCheck':>14}")
print(f"  {'-'*56}")
for m in ['accuracy', 'precision', 'recall', 'f1', 'yes_ratio']:
    v1 = b1_metrics[m]
    v2 = b2_metrics[m]
    v3 = rc_metrics[m]
    # Bold the best
    best = max(v1, v2, v3) if m != 'yes_ratio' else None
    print(f"  {m:<14} {v1:>14.4f} {v2:>14.4f} {v3:>14.4f}", end="")
    if best and v3 == best:
        print("  ★")
    else:
        print()
print(f"{'='*70}")

# --- Improvement deltas ---
print(f"\n  RelCheck vs B1:  accuracy Δ = {rc_metrics['accuracy'] - b1_metrics['accuracy']:+.4f}")
print(f"  RelCheck vs B2:  accuracy Δ = {rc_metrics['accuracy'] - b2_metrics['accuracy']:+.4f}")


# --- Save all results ---

df_llm_b1 = pd.DataFrame(b1_rows)
df_llm_b2 = pd.DataFrame(b2_rows)
df_llm_rc = pd.DataFrame(rc_rows)

# Combined CSV
df_llm_all = pd.concat([df_llm_b1, df_llm_b2, df_llm_rc], ignore_index=True)
df_llm_all.to_csv(f"{EVAL_DIR}/r_pope_llm_judge.csv", index=False)

# Individual CSVs
df_llm_b1.to_csv(f"{EVAL_DIR}/r_pope_llm_judge_b1.csv", index=False)
df_llm_b2.to_csv(f"{EVAL_DIR}/r_pope_llm_judge_b2.csv", index=False)
df_llm_rc.to_csv(f"{EVAL_DIR}/r_pope_llm_judge_rc.csv", index=False)

# Summary JSON
llm_judge_summary = {
    "B1_no_correction": b1_metrics,
    "B2_self_refine":   b2_metrics,
    "RelCheck":         rc_metrics,
    "delta_vs_b1":      rc_metrics['accuracy'] - b1_metrics['accuracy'],
    "delta_vs_b2":      rc_metrics['accuracy'] - b2_metrics['accuracy'],
}
with open(f"{EVAL_DIR}/r_pope_llm_judge_summary.json", "w") as f:
    json.dump(llm_judge_summary, f, indent=2)

print(f"\n✅ LLM-Judge results saved to {EVAL_DIR}/r_pope_llm_judge*.csv")
print(f"✅ Summary saved to {EVAL_DIR}/r_pope_llm_judge_summary.json")

# --- Per-relation-type breakdown ---
print(f"\n{'='*70}")
print("  LLM-Judge R-POPE — PER RELATION TYPE (RelCheck)")
print(f"{'='*70}")
for rtype in df_llm_rc['relation_type'].unique():
    subset = df_llm_rc[df_llm_rc['relation_type'] == rtype]
    n = len(subset)
    acc = subset['correct'].mean()
    # Compare to B1
    b1_sub = df_llm_b1[df_llm_b1['relation_type'] == rtype]
    b1_acc = b1_sub['correct'].mean() if len(b1_sub) > 0 else 0
    delta = acc - b1_acc
    print(f"  {rtype:<20} n={n:>4}  RC acc={acc:.4f}  B1 acc={b1_acc:.4f}  Δ={delta:+.4f}")
print(f"{'='*70}")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL B — CLIPScore Delta
# ═══════════════════════════════════════════════════════════════════════════════
# Measures whether corrected captions align better with images than originals.
# Reference-free metric — uses CLIP embeddings (no ground-truth caption needed).
# ═══════════════════════════════════════════════════════════════════════════════

# !pip install -q open_clip_torch   # uncomment if not installed

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

# Load CLIP model
try:
    import open_clip
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='laion2b_s34b_b79k'
    )
    clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
    clip_model = clip_model.eval().cuda()
    CLIP_AVAILABLE = True
    print("[CLIPScore] Loaded ViT-B-32 (LAION-2B)")
except ImportError:
    print("[CLIPScore] open_clip not installed. Run: pip install open_clip_torch")
    CLIP_AVAILABLE = False

if CLIP_AVAILABLE:
    def clip_score(image_path: str, caption: str) -> float:
        """Compute CLIPScore(image, caption) — cosine similarity in CLIP space."""
        image = clip_preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).cuda()
        text  = clip_tokenizer([caption]).cuda()

        with torch.no_grad():
            img_feat  = clip_model.encode_image(image)
            txt_feat  = clip_model.encode_text(text)

            # Normalize
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

            score = (img_feat @ txt_feat.T).item()
        return score

    # Load RelCheck results
    df_rc = pd.read_csv(f"{EVAL_DIR}/relcheck_results.csv")

    # We need image paths — reconstruct from image_id
    IMAGE_DIR = "/content/RelCheck_Data/images"  # Google Drive cached images
    # Fallback: try to load from eval_dataset if available
    try:
        with open(f"{EVAL_DIR}/rbench_subset.json") as f:
            eval_dataset = json.load(f)
        id_to_path = {item['image_id']: item['image_path'] for item in eval_dataset}
    except FileNotFoundError:
        # Build mapping from image_id → path
        id_to_path = {}
        for img_file in os.listdir(IMAGE_DIR):
            img_id = os.path.splitext(img_file)[0]
            id_to_path[img_id] = os.path.join(IMAGE_DIR, img_file)

    clip_rows = []
    errors = 0

    print(f"\nComputing CLIPScore for {len(df_rc)} images...")

    # Deduplicate: one score per image (not per question)
    unique_images = df_rc.drop_duplicates(subset=['image_id'])

    for _, row in tqdm(unique_images.iterrows(), total=len(unique_images), desc="CLIPScore"):
        image_id = row['image_id']
        original = str(row.get('original_caption', row.get('blip2_caption', '')))
        corrected = str(row.get('corrected_caption', original))
        image_path = id_to_path.get(image_id)

        if not image_path or not os.path.exists(str(image_path)):
            errors += 1
            continue

        try:
            score_orig = clip_score(image_path, original)
            score_corr = clip_score(image_path, corrected)

            clip_rows.append({
                'image_id':            image_id,
                'original_caption':    original,
                'corrected_caption':   corrected,
                'clipscore_original':  round(score_orig, 4),
                'clipscore_corrected': round(score_corr, 4),
                'clipscore_delta':     round(score_corr - score_orig, 4),
                'was_corrected':       original.strip() != corrected.strip(),
                'edit_rate':           row.get('edit_rate', 0.0),
            })
        except Exception as e:
            errors += 1
            continue

    df_clip = pd.DataFrame(clip_rows)
    df_clip.to_csv(f"{EVAL_DIR}/clipscore_results.csv", index=False)

    # --- Summary ---
    corrected_only = df_clip[df_clip['was_corrected'] == True]
    uncorrected    = df_clip[df_clip['was_corrected'] == False]

    print(f"\n{'='*60}")
    print("  CLIPScore Results")
    print(f"{'='*60}")
    print(f"  Total images:     {len(df_clip)}")
    print(f"  Corrected images: {len(corrected_only)}")
    print(f"  Errors:           {errors}")

    print(f"\n  --- ALL IMAGES ---")
    print(f"  Avg CLIPScore (original):  {df_clip['clipscore_original'].mean():.4f}")
    print(f"  Avg CLIPScore (corrected): {df_clip['clipscore_corrected'].mean():.4f}")
    print(f"  Avg delta:                 {df_clip['clipscore_delta'].mean():+.4f}")

    if len(corrected_only) > 0:
        print(f"\n  --- CORRECTED IMAGES ONLY ---")
        print(f"  Avg CLIPScore (original):  {corrected_only['clipscore_original'].mean():.4f}")
        print(f"  Avg CLIPScore (corrected): {corrected_only['clipscore_corrected'].mean():.4f}")
        print(f"  Avg delta:                 {corrected_only['clipscore_delta'].mean():+.4f}")
        print(f"  Images improved:           {(corrected_only['clipscore_delta'] > 0).sum()}/{len(corrected_only)}")
        print(f"  Images degraded:           {(corrected_only['clipscore_delta'] < 0).sum()}/{len(corrected_only)}")

    print(f"\n{'='*60}")
    print(f"✅ CLIPScore results saved to {EVAL_DIR}/clipscore_results.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL C — Filtered R-POPE (corrected images only)
# ═══════════════════════════════════════════════════════════════════════════════
# The full R-POPE is diluted by ~480 images where RelCheck made no changes.
# This cell re-computes metrics on ONLY the images that were actually corrected,
# showing the true effect size.
# ═══════════════════════════════════════════════════════════════════════════════

# Load RelCheck results
df_rc_full = pd.read_csv(f"{EVAL_DIR}/relcheck_results.csv")

# Find corrected images: where original_caption != corrected_caption
corrected_image_ids = set(
    df_rc_full[
        df_rc_full['original_caption'].fillna('') != df_rc_full['corrected_caption'].fillna('')
    ]['image_id'].unique()
)

print(f"Total images: {df_rc_full['image_id'].nunique()}")
print(f"Corrected images: {len(corrected_image_ids)}")
print(f"Uncorrected images: {df_rc_full['image_id'].nunique() - len(corrected_image_ids)}")

# --- Filter all three CSVs to corrected images only ---

df_b1_full = pd.read_csv(B1_CSV)
df_b2_full = pd.read_csv(B2_CSV)

df_b1_filt = df_b1_full[df_b1_full['image_id'].isin(corrected_image_ids)]
df_b2_filt = df_b2_full[df_b2_full['image_id'].isin(corrected_image_ids)]
df_rc_filt = df_rc_full[df_rc_full['image_id'].isin(corrected_image_ids)]

print(f"\nFiltered questions — B1: {len(df_b1_filt)}, B2: {len(df_b2_filt)}, RC: {len(df_rc_filt)}")

# --- VQA-based R-POPE on filtered subset ---

b1_filt_preds = [{'pred': r['pred'], 'gt': r['gt']} for _, r in df_b1_filt.iterrows()]
b2_filt_preds = [{'pred': r['pred'], 'gt': r['gt']} for _, r in df_b2_filt.iterrows()]
rc_filt_preds = [{'pred': r['pred'], 'gt': r['gt']} for _, r in df_rc_filt.iterrows()]

b1_filt_m = compute_rpope_metrics(b1_filt_preds)
b2_filt_m = compute_rpope_metrics(b2_filt_preds)
rc_filt_m = compute_rpope_metrics(rc_filt_preds)

print(f"\n{'='*70}")
print("  FILTERED R-POPE (VQA) — Corrected Images Only")
print(f"{'='*70}")
print(f"  {'Metric':<14} {'B1 (No Corr)':>14} {'B2 (Self-Ref)':>14} {'RelCheck':>14}")
print(f"  {'-'*56}")
for m in ['accuracy', 'precision', 'recall', 'f1', 'yes_ratio']:
    print(f"  {m:<14} {b1_filt_m[m]:>14.4f} {b2_filt_m[m]:>14.4f} {rc_filt_m[m]:>14.4f}")
print(f"{'='*70}")
print(f"  (Based on {len(corrected_image_ids)} corrected images, "
      f"{len(rc_filt_preds)} questions)")

# --- LLM-Judge on filtered subset (if LLM-judge CSVs exist) ---

llm_judge_csv = f"{EVAL_DIR}/r_pope_llm_judge.csv"
if os.path.exists(llm_judge_csv):
    df_llm = pd.read_csv(llm_judge_csv)

    for source_label in ['B1 (No Correction)', 'B2 (Self-Refine)', 'RelCheck']:
        sub = df_llm[df_llm['caption_source'] == source_label]
        sub_filt = sub[sub['image_id'].isin(corrected_image_ids)]
        if len(sub_filt) > 0:
            preds = [{'pred': r['llm_pred'], 'gt': r['gt']} for _, r in sub_filt.iterrows()]
            m = compute_rpope_metrics(preds)
            print(f"\n  LLM-Judge Filtered — {source_label}: acc={m['accuracy']:.4f}, f1={m['f1']:.4f}")

# Save filtered results
filt_summary = {
    "n_corrected_images": len(corrected_image_ids),
    "n_questions": len(rc_filt_preds),
    "B1_filtered": b1_filt_m,
    "B2_filtered": b2_filt_m,
    "RelCheck_filtered": rc_filt_m,
}
with open(f"{EVAL_DIR}/filtered_rpope_summary.json", "w") as f:
    json.dump(filt_summary, f, indent=2)

print(f"\n✅ Filtered R-POPE saved to {EVAL_DIR}/filtered_rpope_summary.json")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL D — LLM Direct-Correction Baseline (B3)
# ═══════════════════════════════════════════════════════════════════════════════
# This is the CRITICAL ablation: give Llama-3.3-70B the BLIP-2 caption and
# ask it to fix any incorrect relationships — NO structured pipeline, NO
# OWLv2 detection, NO VQA verification. Just brute-force LLM correction.
#
# If RelCheck beats this, it proves the structured detect-then-correct pipeline
# adds value over simply throwing an LLM at the problem.
#
# ★ RUN THIS FIRST on 50 images to quickly test the pivot question ★
# Then expand to full 600 if results look promising for RelCheck.
#
# DOES NOT REQUIRE re-running the pipeline — uses existing B1 captions.
# ═══════════════════════════════════════════════════════════════════════════════

import os
import json
import time
import pandas as pd
from tqdm import tqdm
import together

# --- Config ---
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY", "")
LLM_MODEL        = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
REPO_DIR         = "/content/RelCheck"
EVAL_DIR         = f"{REPO_DIR}/eval"

client_d = together.Together(api_key=TOGETHER_API_KEY)

# --- Load B1 (original BLIP-2 captions) ---
B1_CSV = f"{EVAL_DIR}/baseline_no_correction.csv"
df_b1_d = pd.read_csv(B1_CSV)

# Get unique image→caption mapping
b1_caption_col_d = 'blip2_caption' if 'blip2_caption' in df_b1_d.columns else 'original_caption'
unique_imgs = df_b1_d.drop_duplicates(subset=['image_id'])[['image_id', b1_caption_col_d]].copy()
unique_imgs.rename(columns={b1_caption_col_d: 'original_caption'}, inplace=True)

print(f"Total unique images: {len(unique_imgs)}")

# --- LLM Direct-Correction Prompt ---
B3_SYSTEM_PROMPT = """You are a precise image caption editor. You will be given an image caption \
that may contain incorrect relationship descriptions (e.g., wrong spatial relations like \
"on" instead of "next to", wrong actions like "riding" instead of "standing near", or wrong \
attributes connecting objects).

Your job:
1. Identify any relationships between objects that seem likely to be incorrect or implausible.
2. Fix ONLY those relationships with the most plausible alternative.
3. Keep all other words, objects, and descriptions exactly the same.
4. If the caption seems correct, return it unchanged.
5. Output ONLY the corrected caption — no explanation, no quotes, nothing else."""

B3_USER_TEMPLATE = """Caption: "{caption}"

Review this caption for any incorrect or implausible relationships between objects. \
Fix only the incorrect relationships and return the corrected caption:"""


def llm_direct_correct(caption: str, max_retries: int = 3) -> str:
    """
    Ask Llama-3.3-70B to directly correct any relational errors in a caption.
    No structured pipeline — just brute-force LLM correction.
    """
    for attempt in range(max_retries):
        try:
            response = client_d.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": B3_SYSTEM_PROMPT},
                    {"role": "user",   "content": B3_USER_TEMPLATE.format(caption=caption)},
                ],
                max_tokens=200,
                temperature=0.2,
            )
            corrected = response.choices[0].message.content.strip()
            # Strip accidental quotes
            corrected = corrected.strip('"').strip("'").strip()
            return corrected
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            print(f"  [B3] Failed after {max_retries} attempts: {e}")
            return caption  # return original on failure


# --- Run on first N images (set N=50 for fast pivot test, N=len for full run) ---
N_IMAGES = 50  # ← Change to len(unique_imgs) for full 600-image run

subset = unique_imgs.head(N_IMAGES).copy()
print(f"\nRunning LLM direct-correction on {len(subset)} images...")

b3_results = []
for _, row in tqdm(subset.iterrows(), total=len(subset), desc="B3 Direct LLM"):
    original = str(row['original_caption'])
    corrected = llm_direct_correct(original)

    # Compute edit rate
    if original.strip() == corrected.strip():
        edit_rate = 0.0
    else:
        # Simple character-level edit rate
        m_len, n_len = len(original), len(corrected)
        dp = list(range(n_len + 1))
        for i in range(1, m_len + 1):
            prev = dp[:]
            dp[0] = i
            for j in range(1, n_len + 1):
                if original[i-1] == corrected[j-1]:
                    dp[j] = prev[j-1]
                else:
                    dp[j] = 1 + min(prev[j], dp[j-1], prev[j-1])
        edit_rate = dp[n_len] / m_len if m_len > 0 else 0.0

    b3_results.append({
        'image_id':           row['image_id'],
        'original_caption':   original,
        'b3_corrected':       corrected,
        'was_changed':        original.strip() != corrected.strip(),
        'edit_rate':          round(edit_rate, 4),
    })

df_b3 = pd.DataFrame(b3_results)
df_b3.to_csv(f"{EVAL_DIR}/baseline_llm_direct.csv", index=False)
import shutil as _shutil; _shutil.copy2(f"{EVAL_DIR}/baseline_llm_direct.csv", DRIVE_EVAL_DIR); print("💾 Synced baseline_llm_direct.csv → Drive")

# --- Summary ---
n_changed = df_b3['was_changed'].sum()
avg_edit  = df_b3[df_b3['was_changed']]['edit_rate'].mean() if n_changed > 0 else 0.0

print(f"\n{'='*60}")
print(f"  B3 (LLM Direct Correction) — Summary")
print(f"{'='*60}")
print(f"  Images processed:  {len(df_b3)}")
print(f"  Images changed:    {n_changed}/{len(df_b3)} ({100*n_changed/len(df_b3):.1f}%)")
print(f"  Avg edit rate:     {avg_edit:.4f} (on changed images)")
print(f"{'='*60}")

# --- Show 5 examples ---
changed = df_b3[df_b3['was_changed']].head(5)
for _, row in changed.iterrows():
    print(f"\n  Image: {row['image_id']}")
    print(f"  ORIG: {row['original_caption']}")
    print(f"  B3:   {row['b3_corrected']}")

print(f"\n✅ B3 results saved to {EVAL_DIR}/baseline_llm_direct.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL E — Quick Head-to-Head: B3 vs RelCheck via LLM-Judge (50 images)
# ═══════════════════════════════════════════════════════════════════════════════
# Uses the same LLM-judge approach as Cell A, but runs on ONLY the 50-image
# subset from Cell D. Compares B1 (original), B3 (LLM direct), and RelCheck.
#
# This gives you a fast answer to the pivot question:
#   "Does the structured pipeline beat brute-force LLM?"
#
# ★ RUN IMMEDIATELY AFTER CELL D ★
# ═══════════════════════════════════════════════════════════════════════════════

import os
import json
import time
import pandas as pd
from tqdm import tqdm
import together

# --- Config ---
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY", "")
LLM_JUDGE_MODEL  = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
REPO_DIR         = "/content/RelCheck"
EVAL_DIR         = f"{REPO_DIR}/eval"

client_e = together.Together(api_key=TOGETHER_API_KEY)

# --- Load data ---
B3_CSV = f"{EVAL_DIR}/baseline_llm_direct.csv"
B1_CSV = f"{EVAL_DIR}/baseline_no_correction.csv"
RC_CSV = f"{EVAL_DIR}/relcheck_results.csv"

df_b3_e  = pd.read_csv(B3_CSV)
df_b1_e  = pd.read_csv(B1_CSV)
df_rc_e  = pd.read_csv(RC_CSV)

# Get the image IDs from B3 (the 50-image subset)
b3_image_ids = set(df_b3_e['image_id'].unique())

# Filter B1 and RC to only those images
df_b1_sub = df_b1_e[df_b1_e['image_id'].isin(b3_image_ids)].copy()
df_rc_sub = df_rc_e[df_rc_e['image_id'].isin(b3_image_ids)].copy()

# Build image_id → B3 corrected caption mapping
b3_caption_map = dict(zip(df_b3_e['image_id'], df_b3_e['b3_corrected']))

# Add B3 caption column to B1 DataFrame (B1 has the questions + GT)
df_b1_sub['b3_caption'] = df_b1_sub['image_id'].map(b3_caption_map)

print(f"B3 images: {len(b3_image_ids)}")
print(f"B1 questions (subset): {len(df_b1_sub)}")
print(f"RC questions (subset): {len(df_rc_sub)}")

# --- LLM Judge (reuse from Cell A) ---
LLM_JUDGE_SYSTEM_E = """You are a factual judge. Given an image caption and a yes/no question, \
answer based ONLY on what the caption states. Do NOT use any external knowledge.

If the caption explicitly or implicitly supports the answer "yes", respond "yes".
If the caption contradicts the claim or does not mention it, respond "no".

Respond with ONLY "yes" or "no" — nothing else."""

LLM_JUDGE_USER_E = """Caption: "{caption}"

Question: {question}

Based solely on what the caption says, answer yes or no:"""


def llm_judge_e(caption: str, question: str, max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            response = client_e.chat.completions.create(
                model=LLM_JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": LLM_JUDGE_SYSTEM_E},
                    {"role": "user",   "content": LLM_JUDGE_USER_E.format(
                        caption=caption, question=question
                    )},
                ],
                max_tokens=5,
                temperature=0.0,
            )
            answer = response.choices[0].message.content.strip().lower()
            if "yes" in answer:
                return "yes"
            elif "no" in answer:
                return "no"
            else:
                return "no"
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return "no"


# --- Run LLM-Judge on 3 conditions: B1, B3, RelCheck ---

def run_judge_subset(df, caption_col, label):
    preds = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Judge-{label}"):
        caption  = str(row[caption_col])
        question = str(row['question'])
        gt       = str(row['gt']).lower().strip()
        pred     = llm_judge_e(caption, question)
        preds.append({'pred': pred, 'gt': gt})
    return preds


b1_cap_col = 'blip2_caption' if 'blip2_caption' in df_b1_sub.columns else 'original_caption'

print("\n--- Running LLM-Judge on 50-image subset ---")
b1_preds_e = run_judge_subset(df_b1_sub, b1_cap_col, "B1")
b3_preds_e = run_judge_subset(df_b1_sub, 'b3_caption', "B3")

# For RelCheck, get the corrected_caption column
rc_cap_col = 'corrected_caption'
rc_preds_e = run_judge_subset(df_rc_sub, rc_cap_col, "RC")

# --- Compute metrics ---
b1_m = compute_rpope_metrics(b1_preds_e)
b3_m = compute_rpope_metrics(b3_preds_e)
rc_m = compute_rpope_metrics(rc_preds_e)

# --- Display comparison ---
print(f"\n{'='*75}")
print("  ★ PIVOT TEST: B1 vs B3 (LLM Direct) vs RelCheck — LLM-Judge R-POPE ★")
print(f"{'='*75}")
print(f"  {'Metric':<14} {'B1 (Original)':>14} {'B3 (LLM Direct)':>16} {'RelCheck':>14}")
print(f"  {'-'*60}")
for m in ['accuracy', 'precision', 'recall', 'f1', 'yes_ratio']:
    v1, v3, vr = b1_m[m], b3_m[m], rc_m[m]
    best = max(v1, v3, vr) if m != 'yes_ratio' else None
    marker = ""
    if best:
        if vr == best:
            marker = "  ★ RC wins"
        elif v3 == best:
            marker = "  ⚠ B3 wins"
        elif v1 == best:
            marker = "  — B1 wins"
    print(f"  {m:<14} {v1:>14.4f} {v3:>16.4f} {vr:>14.4f}{marker}")
print(f"{'='*75}")

print(f"\n  B3 vs B1:  accuracy Δ = {b3_m['accuracy'] - b1_m['accuracy']:+.4f}")
print(f"  RC vs B1:  accuracy Δ = {rc_m['accuracy'] - b1_m['accuracy']:+.4f}")
print(f"  RC vs B3:  accuracy Δ = {rc_m['accuracy'] - b3_m['accuracy']:+.4f}")

# --- Decision guidance ---
print(f"\n{'='*75}")
if rc_m['accuracy'] > b3_m['accuracy']:
    print("  ✅ RelCheck BEATS brute-force LLM → structured pipeline adds value!")
    print("     Proceed with full 600-image re-run using new code.")
elif rc_m['accuracy'] == b3_m['accuracy']:
    print("  ⚖️  RelCheck TIES brute-force LLM → pipeline may add interpretability value")
    print("     Consider: RelCheck provides traceable edits + triple-level analysis")
else:
    gap = b3_m['accuracy'] - rc_m['accuracy']
    print(f"  ⚠️  B3 beats RelCheck by {gap:.4f} accuracy")
    if gap < 0.03:
        print("     Gap is small — RelCheck still has interpretability advantage.")
        print("     Argue: RelCheck = comparable accuracy + explainability + no retraining.")
    else:
        print("     Gap is significant — consider pivoting:")
        print("     Option A: Use LLM direct-correction as a STAGE in RelCheck")
        print("               (detect with pipeline, correct with LLM direct)")
        print("     Option B: Add B3 as a baseline and focus on interpretability angle")
print(f"{'='*75}")

# Save results
h2h_summary = {
    "n_images": len(b3_image_ids),
    "n_questions_b1": len(b1_preds_e),
    "n_questions_rc": len(rc_preds_e),
    "B1_metrics": b1_m,
    "B3_llm_direct_metrics": b3_m,
    "RelCheck_metrics": rc_m,
    "delta_rc_vs_b3": round(rc_m['accuracy'] - b3_m['accuracy'], 4),
    "delta_b3_vs_b1": round(b3_m['accuracy'] - b1_m['accuracy'], 4),
    "delta_rc_vs_b1": round(rc_m['accuracy'] - b1_m['accuracy'], 4),
}
with open(f"{EVAL_DIR}/pivot_test_b3_vs_relcheck.json", "w") as f:
    json.dump(h2h_summary, f, indent=2)
import shutil as _shutil; _shutil.copy2(f"{EVAL_DIR}/pivot_test_b3_vs_relcheck.json", DRIVE_EVAL_DIR); print("💾 Synced pivot_test_b3_vs_relcheck.json → Drive")

print(f"\n✅ Pivot test saved to {EVAL_DIR}/pivot_test_b3_vs_relcheck.json")
