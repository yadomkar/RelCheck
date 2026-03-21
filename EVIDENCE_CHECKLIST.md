# RelCheck v2 — Master Evidence & Tables Checklist

**Purpose:** Every table, figure, and comparison needed for the CS298 report.
Run experiments in order. Check off as you compute each result.

---

## Tables

### Table 1: Main R-POPE Results (LLM-Judge) — PRIMARY METRIC
*Caption-grounded evaluation: Llama judges caption quality directly (no image).*

| Method | Accuracy | Precision | Recall | F1 | Yes Ratio |
|--------|----------|-----------|--------|----|-----------|
| B1: No Correction | | | | | |
| B2: Self-Refinement | | | | | |
| B3: LLM Direct Correction (no evidence) | | | | | |
| RelCheck v2 (structured KB + correction) | | | | | |

- [ ] B1 (reuse existing baseline_no_correction.csv)
- [ ] B2 (reuse existing self_refine.csv)
- [ ] B3 (run Llama correction with no evidence)
- [ ] RelCheck v2 (run full new pipeline)
- **Run on:** 600 images, mention-filtered R-Bench questions

---

### Table 2: R-POPE Results (VQA-based) — SECONDARY METRIC
*Shows downstream VQA stability — LLaVA answers from image, so corrections shouldn't hurt.*

| Method | Accuracy | Precision | Recall | F1 | Yes Ratio |
|--------|----------|-----------|--------|----|-----------|
| B1 | | | | | |
| B2 | | | | | |
| B3 | | | | | |
| RelCheck v2 | | | | | |

- [ ] Already have B1, B2 from previous runs
- [ ] B3 + RelCheck v2 need to be run
- **Expected:** All methods ≈ same (proves VQA metric is insensitive — that's a finding)

---

### Table 3: KB Source Ablation — KEY CONTRIBUTION
*Which grounding method works best for building the relational KB?*

| KB Source | LLM-Judge Acc | CLIPScore Δ | Corrections Made | Edit Rate |
|-----------|---------------|-------------|------------------|-----------|
| No KB (B3 baseline) | | | | |
| GroundingDINO spatial only | | | | |
| GroundingDINO + geometric rules | | | | |
| Full RelCheck v2 (GDINO + geometry + Llama NLI) | | | | |

- [ ] No KB (same as B3)
- [ ] GDINO spatial only (detect objects, no geometry — just "these objects exist")
- [ ] GDINO + geometry (spatial relations from bbox, no LLM comparison)
- [ ] Full pipeline (GDINO + geometry + Llama comparison + correction)
- **Run on:** 600 images
- **Story:** Each component adds value; full pipeline > any subset

---

### Table 4: Correction Method Ablation — KEY CONTRIBUTION
*How you present evidence to the corrector matters.*

| Correction Method | LLM-Judge Acc | BLEU-4 | Edit Rate | Grammatical? |
|-------------------|---------------|--------|-----------|--------------|
| B3: No evidence ("fix hallucinations") | | | | |
| KB dump: Full KB, unstructured | | | | |
| Structured: Only contradictions highlighted | | | | |
| Structured + verification loop | | | | |

- [ ] B3 (already have this)
- [ ] KB dump (give Llama full KB, ask to rewrite)
- [ ] Structured (give only contradictions + KB facts, ask to fix specific spans)
- [ ] Structured + loop (correct → re-verify → correct again if new issues)
- **Run on:** 600 images (or at least on corrected subset)
- **Story:** Targeted correction > blind correction; verification loop catches cascading errors

---

### Table 5: CLIPScore Comparison
*Automated image-caption alignment.*

| Method | Mean CLIPScore | Δ vs B1 |
|--------|----------------|---------|
| B1 | | — |
| B2 | | |
| B3 | | |
| RelCheck v2 | | |

- [ ] Compute CLIPScore for all 4 conditions
- **Run on:** 600 images

---

### Table 6: R-CHAIR (Manual Annotation) — 50 images
*Gold-standard hallucination rate.*

| Method | R-CHAIR_s (caption-level) | R-CHAIR_i (triple-level) |
|--------|---------------------------|--------------------------|
| B1 | | |
| B3 | | |
| RelCheck v2 | | |

- [ ] Manually annotate 50 images: for each caption, mark which relational claims are hallucinated
- [ ] Compute R-CHAIR_s = % captions with ≥1 hallucination
- [ ] Compute R-CHAIR_i = % triples that are hallucinated
- **Time:** ~1-2 hours

---

### Table 7: Filtered R-POPE (corrected images only)
*R-POPE on ONLY images where RelCheck made corrections — not diluted by unchanged images.*

| Method | Accuracy | Precision | Recall | F1 | N images |
|--------|----------|-----------|--------|----|----|
| B1 (on corrected subset) | | | | | |
| RelCheck v2 (on corrected subset) | | | | | |
| Delta | | | | | |

- [ ] Filter to images where edit_rate > 0
- [ ] Run LLM-judge R-POPE on just those images
- **Story:** When RelCheck acts, it helps — the full-dataset number is diluted

---

### Table 8: Per-Relation-Type Breakdown
*Different tools excel at different relation types.*

| Relation Type | N triples | Hallucination Rate (B1) | Hallucination Rate (RelCheck) | Δ |
|---------------|-----------|-------------------------|-------------------------------|---|
| Spatial (on, in, above, ...) | | | | |
| Action (holding, riding, ...) | | | | |
| Attribute (red, large, ...) | | | | |

- [ ] Categorize all triples by type
- [ ] Compute hallucination rates per type before/after correction
- **Story:** Geometric verification is strong for spatial; action relations remain harder (future work)

---

### Table 9: Pipeline Statistics

| Statistic | Value |
|-----------|-------|
| Total images processed | 600 |
| Images with ≥1 contradiction | |
| Images corrected | |
| Avg contradictions per image | |
| Avg edit rate (corrected images) | |
| Avg objects detected per image | |
| Avg spatial relations per image | |
| Mean pipeline runtime per image | |

- [ ] Compute all stats from pipeline run
- [ ] Add timing to pipeline loop

---

### Table 10: Threshold Sensitivity

| yes_ratio threshold | LLM-Judge Acc | Corrections | Edit Rate |
|---------------------|---------------|-------------|-----------|
| 0.50 (permissive) | | | |
| 0.55 | | | |
| 0.60 | | | |
| 0.65 (current) | | | |
| 0.70 (strict) | | | |

- [ ] Run pipeline 5x with different thresholds (only need to re-run comparison + correction, not detection)
- **Story:** Shows sensitivity curve, justifies threshold choice

---

## Figures

### Figure 1: Architecture Diagram
- [ ] 3-stage pipeline: Caption → GroundingDINO Detection → Geometric KB → Llama Comparison → Llama Correction
- Show data flow between stages
- Highlight what's novel (KB construction, structured comparison)

### Figure 2: Qualitative Examples (5-8 images)
- [ ] Before/after corrections with visual annotations
- Show: original caption, detected objects (with boxes), KB relations, contradictions found, corrected caption
- Pick examples that show different relation types

### Figure 3: Threshold Sensitivity Curve
- [ ] Plot: x = threshold, y = LLM-Judge accuracy
- Show correction count on secondary y-axis
- **Story:** Sweet spot where accuracy peaks

### Figure 4: Per-Relation-Type Bar Chart
- [ ] Grouped bars: spatial vs action vs attribute
- Before/after correction hallucination rates

### Figure 5: KB Source Ablation Bar Chart
- [ ] LLM-Judge accuracy for each KB source
- Shows incremental value of each component

### Figure 6: R-POPE VQA vs LLM-Judge Comparison
- [ ] Side-by-side bar chart showing VQA is flat while LLM-judge shows differences
- **Story:** Proves methodological contribution (R-POPE VQA is broken for this task)

---

## Key Numbers to Compute During Full Run

When running the 600-image experiment, make sure to save:

1. Per-image: image_id, original_caption, corrected_caption, n_objects_detected, n_relations_found, n_contradictions, n_corrections, edit_rate, runtime_seconds
2. Per-triple: image_id, subject, relation, object, relation_type, detected_in_image (bool), kb_spatial_relation (if any), llm_judgment (contradiction/supported/unverifiable), corrected (bool)
3. Per-question (R-POPE): image_id, question, gt_answer, b1_pred, b3_pred, relcheck_pred, mentioned_in_caption (bool)

Save as CSVs for easy analysis later.

---

### Table 11: Multi-Captioner Generalizability
*Proves relational hallucinations are model-agnostic, not a BLIP-2-specific problem.*

| Captioner | R-CHAIR_s (before) | R-CHAIR_s (after RelCheck) | LLM-Judge Acc (before) | LLM-Judge Acc (after) |
|-----------|--------------------|-----------------------------|------------------------|----------------------|
| BLIP-2 (flan-t5-xl) | | | | |
| InstructBLIP | | | | |

- [ ] Run InstructBLIP on 600 images (or 100 subset) to generate captions
- [ ] Run RelCheck v2 pipeline on InstructBLIP captions
- [ ] Compute LLM-Judge R-POPE for both captioners
- [ ] Cite R-Bench (Wu et al., ICML 2024) Table 1 showing hallucinations across models
- **Story:** Relational hallucinations are a model-agnostic problem; RelCheck improves caption quality regardless of captioner
- **Time:** ~3-4 hrs (mostly Colab runtime)
- **Priority:** Medium — not in minimum viable, but preempts the "just use a better captioner" criticism

---

## Evidence for Report Discussion Section

- [ ] 3-5 failure cases with analysis (why did RelCheck fail?)
- [ ] GroundingDINO detection failure examples (objects not found)
- [ ] Action relation limitation examples (geometry can't verify actions)
- [ ] Comparison to Woodpecker (object hallucination vs relational)
- [ ] Comparison to Reefknot (black-box vs internal model modification)
- [ ] Runtime comparison (is it practical?)

---

## Minimum Viable Evidence (if time is tight)

If running out of time, these are the MUST-HAVES:

1. ✅ Table 1 (LLM-Judge R-POPE — main metric)
2. ✅ Table 3 (KB source ablation — main contribution)
3. ✅ Table 4 (Correction method ablation — second contribution)
4. ✅ Table 9 (Pipeline stats)
5. ✅ Figure 1 (Architecture diagram)
6. ✅ Figure 2 (Qualitative examples)
7. ✅ 3 failure cases for discussion

Everything else strengthens the report but isn't strictly necessary to pass.
