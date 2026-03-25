# Memory — CS298 / RelCheck
**Last updated:** 2026-03-25 (Session 7)

---

## Me
**Siddhi Patil** — SJSU Master's student, CS298 (Spring 2026). Solo project, 3-week execution window. Email: siddhipatil506@gmail.com | Student ID: 018185102

---

## People

| Who | Role |
|-----|------|
| **Dr. Andreopoulos** | Advisor — Dr. William Andreopoulos (SJSU) |
| **Dr. Saxena** | Committee member — Dr. Navrati Saxena (SJSU) |
| **Pruthviraj** | Committee member — Pruthviraj Urankar; also did CS298 (PhotoProof, Spring 2025) |

---

## Project

| Name | What |
|------|------|
| **RelCheck** | CS298 master's project — training-free, post-hoc system that detects + corrects relational hallucinations in Multimodal LLM outputs |
| **CS297** | Prerequisite research methods course — lit survey + pipeline design already done |
| **CS298** | Master's project course — implementation + evaluation + report, due ~April 1, 2026 |

---

## Terms & Acronyms

| Term | Meaning |
|------|---------|
| **MLLM** | Multimodal Large Language Model (e.g., BLIP-2) |
| **Triple** | (subject, relation, object) — structured unit extracted from a caption |
| **R-POPE (VQA)** | Current eval: LLaVA(image + caption, question) → yes/no vs GT. Problem: LLaVA ignores caption, uses image directly |
| **R-POPE (NLI/LLM-judge)** | Llama-3.3-70B(caption only, question) → yes/no vs GT. Measures caption quality directly. ✅ Working — +5.8% on 100 images |
| **Enrichment** | Key insight from Session 6: BLIP-2's main problem is omissions (45%), not false claims (2%). Enrichment = fix errors + add verified missing KB facts in ≤3 sentences |
| **R-CHAIR_s** | % captions with ≥1 relational hallucination (caption-level) |
| **R-CHAIR_i** | % of total triples that are hallucinated (triple-level) |
| **CLIPScore** | Reference-free image-caption alignment metric (CLIP embedding similarity) |
| **R-Bench** | Benchmark dataset (Wu et al., ICML 2024) — 11,651 relational Q&A pairs on nocaps images |
| **BLIP-2** | Vision-language model used for captioning (blip2-flan-t5-xl) |
| **LLaVA-1.5-7B** | Legacy cross-model verifier — replaced by Llama-4-Maverick in Session 5 (too weak, 9-11/20 false positives) |
| **GroundingDINO** | IDEA-Research/grounding-dino-tiny — zero-shot object detector for spatial geometry verification (replaces OWLv2 in Session 5) |
| **Llama-4-Maverick** | meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 — VLM for action/attribute verification via Together.ai (replaces LLaVA-1.5-7B in Session 5) |
| **OWLv2** | google/owlv2-base-patch16-ensemble — legacy detector, replaced by GroundingDINO in Session 5 |
| **OWL-ViT** | google/owlvit-base-patch32 — legacy detector, replaced by OWLv2 in Session 2 |
| **Pix2Grp** | CVPR 2024 SGG model — attempted in Session 4, failed viability test |
| **Llama-3.3-70B-Instruct-Turbo** | LLM for triple extraction + correction via Together.ai (replaced Mistral-7B) |
| **Together.ai** | API provider for Llama-3.3-70B (paid credits) |
| **Woodpecker** | Related work — corrects object hallucinations but ignores relational structure |
| **POPE** | Prior benchmark for object hallucination (binary Yes/No); R-POPE extends this |
| **Edit Rate** | Levenshtein(before, after) / len(before) — measures how much correction changed caption |
| **BLEU-4** | Fluency metric — corrected caption vs. reference caption |
| **nocaps** | Image dataset used by R-Bench for evaluation |
| **self-refinement** | Baseline 2: prompt BLIP-2 to re-check its own description |
| **NLI** | Natural Language Inference — check if caption entails answer to a question |
| **VisMin** | NeurIPS 2024 paper (Awal et al.) — benchmark for fine-grained visual understanding; validates spatial relation difficulty; inspired multi-question VQA voting |

---

## RelCheck Pipeline (3 Stages) — Updated Session 2

1. **Triple Extractor** — Llama-3.3-70B via Together.ai (spaCy fallback)
   - "with" removed from SPATIAL_KEYWORDS → routes to VQA via OTHER type
   - INSTRUMENTAL_KEYWORDS added: {"with", "using", "via"}
2. **Relation Verifier** — type-aware: spatial (OWLv2 + geometry) vs. action/attribute (LLaVA-1.5-7B VQA)
   - **OWLv2** replaces OWL-ViT v1 (better open-vocab detection)
   - **Multi-question VQA voting**: 3 paraphrased questions per triple, averaged yes_ratios (inspired by VisMin)
   - **Dual query cleaning**: aggressive for OWLv2 (strips adjectives), light for VQA (keeps adjectives)
   - Unknown spatial relations now fall back to VQA instead of auto-approving
3. **Minimal Corrector** — Llama-3.3-70B edits only hallucinated spans; self-consistency guard
   - **Batch correction**: 2+ hallucinated triples corrected in single LLM call (avoids cascading)
   - Sequential fallback preserved if batch fails

---

## Key Files

| File | What |
|------|------|
| `RelCheck_Master.ipynb` | Master Colab notebook — run this for all experiments |
| `relcheck/triple_extractor.py` | Stage 1 — INSTRUMENTAL_KEYWORDS added Session 2 |
| `relcheck/relation_verifier.py` | Stage 2 — OWLv2 + multi-question voting + dual cleaning added Session 2 |
| `relcheck/corrector.py` | Stage 3 — batch correction added Session 2 |
| `relcheck/relcheck_pipeline.py` | End-to-end wiring — num_paraphrases param added Session 2 |
| `CLAUDE.md` | Project memory — update every session |
| `eval/` | CSVs from experiments (generated by Colab, synced to Google Drive) |
| `figures/` | Plots and architecture diagram (generated by Colab) |
| `images/` | 12 pilot images |

---

## Experimental Results (as of 2026-03-19, BEFORE Session 2 fixes)

### R-POPE (VQA-based, LLaVA evaluator) — 600 images, 1,618 questions

| Metric | B1 (No Correction) | B2 (Self-Refine) | RelCheck |
|--------|-------------------|------------------|----------|
| Accuracy | 74.23% | 75.40% | 75.34% |
| Precision | 70.25% | 72.01% | 71.94% |
| Recall | 88.76% | 87.35% | 87.35% |
| F1 | 78.43% | 78.94% | 78.90% |
| Yes ratio | 66.69% | 64.03% | 64.09% |

### RelCheck Pipeline Stats (pre-Session 2)
- Images corrected: 119/600 (19.8%)
- Avg edit rate: 3.84%
- Avg BLEU-4: 0.909

### Key Finding: R-POPE (VQA) doesn't capture RelCheck's value
- RelCheck ≈ Baseline 2 on R-POPE because LLaVA answers from the IMAGE, ignoring caption corrections
- The pipeline IS working (119 corrections made, edit rate > 0) but R-POPE (VQA) can't measure it
- Need LLM-judge evaluation to measure caption quality directly

**⚠ IMPORTANT: After Session 2 code changes, need to re-run full pipeline (Section 6) to get new numbers. Expect correction count to increase significantly (119 → 200+) due to fixes #1-2 and multi-question voting.**

---

## Session 2 Architecture Changes (2026-03-19)

### Design Flaws Fixed

1. **"with" routing bug** — `"with"` was in `SPATIAL_KEYWORDS`, so `(baby, with, toy box)` routed to OWL-ViT geometry. No geometric rule existed → always marked "supported" (silent false negative). Fixed: moved to `INSTRUMENTAL_KEYWORDS`, routes to VQA.

2. **Unknown spatial auto-approval** — `check_spatial_relation` returned `True` for unrecognized relations ("across", "along", "through", etc.). Fixed: returns `None` → triggers VQA fallback.

3. **VQA question over-cleaning** — `_clean_query` stripped adjectives from VQA questions ("an old man" → "man"). Harmful when multiple similar entities exist. Fixed: `_clean_query_for_vqa` strips only articles, keeps adjectives.

4. **Sequential correction cascading** — Multiple hallucinated triples were corrected one-at-a-time, feeding corrected caption into next LLM call → drift risk. Fixed: batch correction in single LLM call for 2+ triples, with sequential fallback.

### Optimizations Added

5. **Multi-question VQA voting** (inspired by VisMin, NeurIPS 2024) — For each triple, generate 2-3 paraphrased questions (active/passive voice, reversed spatial perspective, descriptive variants). Average yes_ratios for more robust decision. Shrinks uncertain zone → more confident decisions → more corrections. Controlled by `num_paraphrases` param (default=3, set to 1 for original behavior).

6. **OWLv2 upgrade** — Default detector changed from `google/owlvit-base-patch32` (2022) to `google/owlv2-base-patch16-ensemble` (better accuracy). Supports both via constructor param. Should reduce VQA-fallback rate for spatial triples.

### VisMin Paper Insights (NeurIPS 2024)
- **Key finding**: VLMs perform below random chance on spatial relations (validates RelCheck's type-aware routing)
- **Grounding DINO**: Used by VisMin for detection; OWLv2 is our compromise (similar improvement, easier API)
- **Multi-question approach**: Their automatic filtering uses multiple question types per sample — inspired our VQA voting
- **4-type taxonomy**: object, attribute, count, spatial. RelCheck covers spatial/action/attribute; count is acknowledged limitation/future work
- **NOT incorporating**: diffusion-based image editing, fine-tuning (RelCheck is training-free), AMT verification pipeline

---

## Critical Evaluation Problem + Solution

### Problem
R-POPE (VQA-based) uses LLaVA(image + caption) as evaluator. LLaVA is a strong vision model that answers from the image directly, so caption corrections don't change its answers. RelCheck ≈ B2 on this metric despite making real corrections.

### Solution: LLM-Judge R-POPE (caption-grounded)
Use Llama-3.3-70B as a TEXT-ONLY judge:
1. Give LLM the caption + R-Bench question (NO image)
2. LLM answers yes/no based solely on what the caption says
3. Compare to R-Bench ground truth (which encodes image reality via human annotation)
4. If corrected caption is more accurate → answers match GT better → higher accuracy

This directly measures caption quality because the LLM can ONLY use the caption.

### Full Evaluation Stack
1. **R-POPE (VQA)** — current results, keep as-is, shows downstream VQA stability
2. **R-POPE (LLM-judge)** — NEW, measures caption accuracy directly via NLI
3. **R-CHAIR** — manual annotation of 50 images, triple-level hallucination rate
4. **CLIPScore** — automated image-caption alignment delta
5. **Filtered R-POPE** — R-POPE on ONLY corrected images (not diluted by unchanged 481)

### Implementation Status
- R-POPE (VQA): ✅ Done (B1, B2, RelCheck all complete) — **needs re-run with new code**
- R-POPE (LLM-judge): ⬜ Need to write cell + run
- R-CHAIR: ⬜ Need manual annotation (~1 hr)
- CLIPScore: ⬜ Need to write cell + run (~30 min)
- Filtered R-POPE: ⬜ Easy — just filter existing results

---

## Bugs Fixed (Session 1)

1. **Root cause of ModuleNotFoundError**: Cell 3 (restore results) created `/content/RelCheck/eval/` before cell 6 (clone-repo). On fresh Colab runtime, this created a plain directory → clone-repo thought repo existed → git pull failed silently → relcheck/ never cloned. Fixed by: cell 3 guards on `os.path.exists(REPO_DIR/relcheck)`, clone-repo checks for `.git` dir, path guards remove stale non-git dirs.
2. **Broken widget metadata**: Colab saved `metadata.widgets` without required `state` key → GitHub showed "Invalid Notebook". Fixed by removing broken widget metadata.
3. **Colab overwrite**: Saving notebook in Colab auto-pushed "Created using Colab" commit that overwrote fixes. Re-applied all fixes after pulling.

---

## Status (as of 2026-03-23, Session 6)

### Session 6: Enrichment Approach + First R-POPE Improvement

**Problem solved:** R-POPE (VQA) showed 0% improvement because LLaVA answers from the image. R-POPE (LLM-judge) with Llama-3.3-70B measures caption quality directly — and correction-only still showed 0% because BLIP-2's errors are mainly omissions, not false claims.

**Key insight:** BLIP-2 baseline has 53.1% R-POPE accuracy with 45% omissions and only 2% false claims. Correction-only fixes the 2% but can't help the 45%. Enrichment addresses both: fix errors + add verified missing facts from KB.

**Enrichment pipeline (Cell 5 of RelCheck_Enriched_100.ipynb):**
1. Build Visual KB (GroundingDINO detections + Maverick VLM description)
2. Llama-3.3-70B single-call analysis: find errors + missing facts → produce improved caption in JSON
3. Guard: only use rewrite if errors or missing facts found
4. Safeguards: sentence count ≤4, LLM verification against KB (faithfulness + fluency + coherence), safe default on failure
5. Verification checks against KB (not original caption) — only FAILs for: KB contradiction, bad grammar, nonsensical repetition

**Results (100 images, 277 R-Bench questions):**

| Metric | BLIP-2 (Original) | RelCheck (Enriched) | Delta |
|--------|-------------------|-------------------|-------|
| R-POPE Accuracy | 147/277 (53.1%) | 163/277 (58.8%) | **+5.8%** |
| Images modified | — | 88/100 | — |
| Modified-only accuracy | 53.3% | 59.8% | +6.5% |
| Questions improved | — | 21 | — |
| Questions regressed | — | 5 | — |
| Net improvement | — | +16 | — |

**By relation type:** SPATIAL +12, ACTION +4, ATTRIBUTE +1, OTHER -1

**Bugs fixed this session:**
- Verification was comparing to original caption (rejected all enrichments) → changed to verify against KB
- Edit rate gate rejected everything (BLIP-2 captions ~30 chars, enriched ~150 chars) → removed gate, kept for reporting
- Every caption was being rewritten → added `if errors_found or missing_found:` guard
- Verification didn't check fluency → added grammar/coherence/repetition checks
- Verification API failure kept unverified rewrite → now keeps original

**Key files:**

| File | What |
|------|------|
| `RelCheck_Enriched_100.ipynb` | Main 100-image enrichment notebook (7 code cells) |
| `RelCheck_Screening.ipynb` | Screening notebook — finds R-Bench images with hallucinations |

**Remaining work:**
- Analyze 5 regressed questions (could push to +7%+)
- Scale to 600 images
- R-CHAIR evaluation
- Multi-model experiment (InternVL2)
- Report writing (~45-50 pages)

---

## Status (as of 2026-03-25, Session 7)

### Session 7: Full 600-Image Notebook + Bug Fixes

**Deliverable:** `RelCheck_600.ipynb` — comprehensive 16-cell notebook that produces ALL evidence for the CS298 report in a single automated run. Zero manual work required.

**Notebook structure (16 cells):**

| Cell | What | Output |
|------|------|--------|
| 0 | Markdown overview | — |
| 1 | Setup + constants + `llm_call()` + checkpoint helpers | — |
| 2 | Load GDINO + BLIP-2 models | GPU models |
| 3 | Load images + R-Bench data (random N_IMAGES sample) | Image dict |
| 4 | BLIP-2 captioning | `captions.json` checkpoint |
| 5 | KB construction (GDINO + Maverick) with per-image timing | `kb_results.json` checkpoint |
| 6 | Full RelCheck enrichment with per-image timing | `enriched.json` checkpoint |
| 7 | Ablation correctors (B2, B3, KB-obj, KB-geom, KB-dump) — post-hoc from saved KB | `ablation_captions.json` checkpoint |
| 8 | R-POPE LLM-Judge (all 7 methods) + McNemar's test | Tables 1, 3, 4, 7, 8 + `rpope_results.json` |
| 9 | CLIPScore via OpenCLIP | Table 5 |
| 10 | Pipeline stats + all CSVs | Table 9 + 5 CSV files |
| 11 | Qualitative examples with bbox visualizations + failure cases | Figure 2 + discussion material |
| 12 | Figures 4, 5, 6 (per-relation-type, KB ablation, correction method) | Bar charts |
| 13 | Automated R-CHAIR (50-image sample, VLM triple verification) | Table 6 |
| 14 | InstructBLIP multi-captioner (100-image subset) | Table 11 |
| 15 | Architecture diagram | Figure 1 |

**Key design decisions:**
- Every expensive cell saves checkpoints to Google Drive (every 50 images), loads from cache on restart
- All ablation variants (B2, B3, KB-obj, KB-geom, KB-dump) computed post-hoc from saved KB — no re-running detection
- McNemar's test for statistical significance (B1 vs RelCheck)
- R-CHAIR fully automated via VLM triple verification (no manual annotation)
- InstructBLIP comparison uses same image subset for fair comparison

**Bugs fixed this session:**
1. Cell 8: `methods_to_eval` cache check was broken — checked `m in rpope_raw[img_id]` but keys are `"method|||question"` format. Never detected cached results, always re-ran everything. Fixed with proper key format check.
2. Cell 8: Missing `.get()` safety for `entry["method"]` in aggregation loop — could crash on malformed JSON. Fixed to `entry.get("method")` with skip guard.
3. Cell 14: InstructBLIP comparison computed BLIP-2 stats on all 600 images but InstructBLIP only runs on 100. Unfair comparison. Fixed to compute both on same `ib_sample` subset.
4. Cell 13: R-CHAIR checkpoint too infrequent (every 20 → every 10 images).

**Run instructions:**
1. First: set `N_IMAGES = 50` in Cell 1 for validation run
2. If validation passes: set `N_IMAGES = 600` for full run
3. Estimated cost: ~$3.80 on Together.ai for 600 images

**Key files:**

| File | What |
|------|------|
| `RelCheck_600.ipynb` | Full 600-image notebook (16 cells) — run this for all evidence |
| `RelCheck_Enriched_100.ipynb` | Previous 100-image enrichment notebook (Session 6) |

---

## Previous Status (Session 5)

### Session 3 Recap
- eval_cells.py updated with Cell 0 (metrics helper), Cell D (B3 baseline), Cell E (pivot test)
- ALL ablation variants produce identical R-POPE (VQA) scores (~75.4%), confirming VQA insensitivity
- **Reefknot (ACL Findings 2025)**: Closest competitor — modifies internal model probabilities (not black-box), no corrected caption output

### Session 4 Recap: Pix2Grp Attempt
- Chose Pix2Grp (CVPR 2024) for scene graph grounding
- **FAILED viability test** — Pix2Grp did not produce useful scene graphs on test images
- SGG models in general have only 40-55% mRecall on action predicates
- Decision: abandon SGG-based approach entirely

### Session 5: Architecture Pivot — Type-Aware Verification (GroundingDINO + VLM)

**Why pivot from SGG:** Pix2Grp failed viability test. Even SOTA SGG models are unreliable on action predicates. Scene graph approach was too fragile for a 13-day deadline.

**New approach:** Replace both the failed LLaVA-1.5-7B verifier (Session 2-3) and the failed Pix2Grp SGG (Session 4) with a type-aware split: deterministic geometry for spatial relations + strong VLM for action/attribute relations.

**Key insight:** Different relation types need fundamentally different verification strategies. Spatial relations ARE geometric facts (verifiable from bounding boxes). Action relations need visual understanding (require a capable VLM, not LLaVA-7B).

### New RelCheck v2 Pipeline (5 Stages)

1. **BLIP-2** → generates caption (unchanged)
2. **Llama-3.3-70B** → extract (subject, relation, object) triples + classify type (SPATIAL/ACTION/ATTRIBUTE)
3. **GroundingDINO** (IDEA-Research/grounding-dino-tiny via HuggingFace) → detect objects + bboxes
   - **Spatial triples** → bbox geometry rules (deterministic: "on" = subject above object + horizontal overlap, etc.)
   - Failed detections → fall through to VLM
4. **Llama-4-Maverick VLM** (meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 via Together.ai) → verify action/attribute triples
   - Multi-question voting: 3 paraphrased yes/no questions, averaged yes_ratios
   - Cross-model verification (BLIP-2 generated caption, Maverick verifies — uncorrelated biases)
5. **Llama-3.3-70B Corrector** → structured evidence + minimal correction + fluency gate
   - Batch correction for 2+ hallucinations (from Session 2)
   - Structured evidence format (prevents literal evidence insertion bug from Session 2)

### Why This Architecture (Reviewer-Ready Arguments)

1. **Type-aware routing is the core contribution** — no prior work (Woodpecker, LURE, Reefknot) splits verification by relation type
2. **GroundingDINO spatial verification is deterministic** — zero hallucination risk from verifier, unlike VQA
3. **Llama-4-Maverick >> LLaVA-1.5-7B** — much stronger VLM for action verification (17B MoE vs 7B)
4. **Training-free + black-box** — works on any captioning model (vs Reefknot which needs model internals)
5. **"Just use a better captioner" objection** — plan to run RelCheck on InternVL2 captions too, showing improvement even on stronger models

### Corrector Safeguards (from Session 2-3 failure analysis)

1. **Fewer false positives** — GroundingDINO is deterministic; Maverick is much stronger than LLaVA
2. **Structured evidence format** — prevents literal evidence insertion into captions
3. **Fluency gate** — BLEU-4 threshold; reject corrections that destroy grammar
4. **Batch correction** — single LLM call for 2+ hallucinations (no cascading drift)
5. **Self-consistency check** — re-verify corrected triples (optional verification loop ablation)

### Research Contributions (4 total)
1. **Problem**: Relational hallucinations need specialized handling (vs. Woodpecker/LURE for objects)
2. **Detection**: Type-aware verification (geometry for spatial, VLM for action) outperforms uniform approaches
3. **Correction**: Structured evidence + fluency gate > blind correction
4. **Evaluation**: R-POPE (VQA) is insensitive; LLM-judge R-POPE measures caption quality directly

### Ablation Design (tests type-aware routing claim)

**Dimension 1 — Verification strategy:**
- B1: No correction (original BLIP-2)
- B2: Self-refinement (BLIP-2 re-checks itself)
- B3: Blind LLM correction (Llama corrects with no evidence)
- GroundingDINO geometry only (spatial verified, actions auto-approved)
- VLM only (all relations verified by Maverick, no geometry)
- **Full RelCheck: type-aware routing** (spatial → geometry, action → VLM)

**Dimension 2 — Correction method:**
- No evidence (B3)
- Raw evidence dump
- Structured evidence (targeted contradiction + facts)
- Structured + verification loop

### Key Files (Session 5)
| File | What |
|------|------|
| `RelCheck_Viability_Test_v2.ipynb` | 5-image viability test for GroundingDINO + Maverick approach |
| `RelCheck_Viability_Test.ipynb` | OLD — Pix2Grp approach (failed) |
| `EVIDENCE_CHECKLIST.md` | Master list of all 10 tables + 6 figures needed |

### Models Used (Session 5)
| Model | ID | Where | Purpose |
|-------|----|-------|---------|
| **BLIP-2** | Salesforce/blip2-flan-t5-xl | Colab GPU | Caption generation (target model) |
| **GroundingDINO** | IDEA-Research/grounding-dino-tiny | Colab GPU | Object detection for spatial geometry |
| **Llama-4-Maverick** | meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 | Together.ai API | VLM action/attribute verification |
| **Llama-3.3-70B** | meta-llama/Llama-3.3-70B-Instruct-Turbo | Together.ai API | Triple extraction + correction |

---

## Revised Plan — 9 Days Remaining (April 3 deadline)

| Day | Task | Time Est |
|-----|------|----------|
| **Day 1 (Mar 21)** | ✅ Viability test v2 (GroundingDINO + Maverick) — PASSED | Done |
| **Days 2-3 (Mar 22-23)** | ✅ Built enriched pipeline, first +5.8% R-POPE result on 100 images | Done |
| **Days 4-5 (Mar 24-25)** | ✅ Built RelCheck_600.ipynb (16 cells, all metrics + ablations + figures), fixed 4 bugs | Done |
| **Day 6 (Mar 26)** | Run RelCheck_600.ipynb: N_IMAGES=50 validation, then N_IMAGES=600 full run | 6 hrs |
| **Day 7 (Mar 27)** | Analyze results, fix any issues, re-run if needed | 4 hrs |
| **Days 8-11 (Mar 28-31)** | Report writing (~45-50 pages) | 15 hrs |
| **Day 12 (Apr 1-2)** | Code cleanup, polish, buffer | 4 hrs |

---

## Threshold Settings (current — unchanged)

| Parameter | Value | Effect |
|-----------|-------|--------|
| `yes_ratio ≥ 0.65` | Supported (not hallucinated) | High bar to confirm |
| `yes_ratio ∈ [0.50, 0.65)` | Uncertain → hallucinated=None → skipped | Wide uncertain zone |
| `yes_ratio < 0.50` | Hallucinated → corrected | Any "no" preference triggers |
| `CONFIDENCE_THRESHOLD = 0.45` | Min confidence for decision | Below → uncertain |
| `DETECTION_THRESHOLD = 0.15` | OWLv2 min score | Below → not detected |
| `num_paraphrases = 3` | Multi-question voting | NEW: 3 paraphrased questions per triple |

### Note on Multi-Question Voting + Thresholds
With 3 paraphrases averaged, yes_ratios should be less noisy. The [0.50, 0.65) uncertain zone may naturally shrink. If results still look weak after re-run, consider:
- Lower supported threshold: 0.65 → 0.55
- Lower CONFIDENCE_THRESHOLD: 0.45 → 0.35
- Expected effect: more images corrected, stronger signal

---

## Preferences
- Siddhi runs all code in **Google Colab** (not local) — T4 or A100 GPU (Colab Pro)
- Claude handles all code writing; Siddhi just runs the notebook
- Work is done via Cowork sessions — update CLAUDE.md at end of each session
- Only suggest API solutions confirmed working on Together.ai (Siddhi paid for credits)
- Keep laptop awake during runs: `caffeinate -i` in Terminal
- Don't waste time on long re-runs without knowing evaluation works first
- **Focus on getting a solid project first, report can come later** (stated Session 2)
