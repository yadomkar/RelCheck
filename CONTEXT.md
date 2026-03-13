# CS298 Project Context File
**Last Updated:** March 11, 2026
**Student:** Siddhi Patil (018185102) | siddhipatil506@gmail.com
**Project:** RelCheck — Relation-Aware Hallucination Detection and Correction for Multimodal Large Language Models
**Advisor:** Dr. William Andreopoulos | Committee: Dr. Navrati Saxena, Pruthviraj Urankar
**Deadline:** ~April 1, 2026 (3 weeks from start)

---

## Project Summary

RelCheck is a **training-free, post-hoc** system that detects and corrects **relational hallucinations** in Multimodal LLM (MLLM) outputs. Instead of rewriting whole captions, it:
1. Extracts structured **(subject, relation, object)** triples from a generated caption
2. Verifies each triple using external vision-language modules
3. Applies **minimal edits** only to hallucinated relations

**Key innovation:** First system to target relation-level hallucinations explicitly (prior work focuses on object-level). Novel contributions include R-POPE and R-CHAIR metrics.

---

## Comparison With Prior SJSU CS298 Projects

### vs. PhotoProof (Pruthviraj Urankar, Spring 2025 — your own committee member!)
PhotoProof built an iOS blockchain app for photo authentication using Ethereum + Merkle trees. Solid engineering but low research novelty — all components existed, the contribution was integration. It passed SJSU comfortably. **RelCheck is more ambitious and more research-original.** You will clear the SJSU bar if you execute.

### vs. GaLoRA (NeurIPS 2025 Workshop — 3 students)
GaLoRA was a genuine published paper combining GNNs + LoRA for node classification on text-attributed graphs. Clean math, 3 datasets, strong ablations. This is the ceiling for CS298. RelCheck's problem framing is equally strong and more timely, but GaLoRA had 3 students. **Solo, in 3 weeks, you cannot match GaLoRA's breadth — so instead focus on depth and clean execution of a scoped-down version.**

### Publishability Assessment
**Yes, publishable — but only with clean execution.** The research gap is real: R-Bench (ICML 2024) benchmarks relational hallucinations but provides no correction method. Woodpecker corrects object hallucinations but ignores relational structure. No prior work has defined R-POPE or R-CHAIR. A workshop paper at ACL/EMNLP/CVPR is realistic. Risks: (1) 3 weeks is tight, (2) ground truth labeling for R-CHAIR is non-trivial, (3) the correction LLM can introduce new errors.

---

## What Was Done in CS297 (Already Complete)
- [x] Comprehensive literature survey (16+ papers)
- [x] Problem taxonomy: spatial / action / attribute-based relational hallucinations
- [x] High-level RelCheck pipeline design
- [x] Evaluation framework design (R-POPE, R-CHAIR proposed)
- [x] BLIP-2 caption probe notebook (`relcheck_blip2_probe.ipynb`) on 12 pilot images
- [x] 12 pilot images in `images/` folder

---

## RelCheck Architecture (To Implement)

```
Input: Image + MLLM-generated caption
        ↓
[Stage 1] Triple Extractor
  - Parse caption → list of (subject, relation, object) triples
  - Tools: spaCy dependency parsing + LLM prompt fallback (Mistral-7B via Together.ai)
        ↓
[Stage 2] Relation Verifier (relation-type-aware)
  - Spatial relations (on/in/above/below/left/right):
      → OWL-ViT bounding boxes + geometric check (IoU, centroid comparison)
  - Action relations (holding/carrying/eating/riding):
      → BLIP-2 VQA probe: "Is [subject] [relation] [object]?" → Yes/No
  - Attribute relations (color/size/shape across entities):
      → BLIP-2 VQA probe with targeted attribute questions
        ↓
[Stage 3] Minimal Corrector
  - For each hallucinated triple: locate minimal span in caption
  - Use Mistral-7B (fully open-source, via Together.ai API) to replace only that span
  - Verify correction doesn't introduce new triples (self-consistency check)
  - Return corrected caption
        ↓
Output: Corrected caption + structured log: {triple, verified, hallucinated, correction}
```

---

## Scope (Realistic for 3 Weeks Solo)
- **One primary MLLM:** BLIP-2 (blip2-flan-t5-xl) — already set up
- **Evaluate on:** 200 images from R-Bench (not all 11,651 — clearly state this as a compute limitation)
- **Two baselines:** (1) No correction, (2) Self-refinement (ask BLIP-2 to re-check itself)
  - Note: Woodpecker is object-level only; cite its numbers from the paper instead of reimplementing
- **LLM for correction:** Mistral-7B via Together.ai (free tier) — fully open-source, reproducible

---

## Evaluation Strategy

### Datasets
- **R-Bench** (Wu et al., ICML 2024): 11,651 relational Yes/No questions on nocaps validation set
  - Paper: `2406.16449v4.pdf` | GitHub: https://github.com/mrwu-mac/R-Bench
  - USE: 200-image subset (image-level questions only for R-POPE)
- **Pilot set**: 12 images in `images/` folder — used for qualitative demo and development

### Metrics
| Metric | Description | How to Compute |
|--------|-------------|----------------|
| **R-POPE** | Binary relation Q&A accuracy | R-Bench Yes/No questions → accuracy, precision, recall, F1 |
| **R-CHAIR_s** | % captions with ≥1 relational hallucination | Manual labels on 50-image subset |
| **R-CHAIR_i** | % of total triples that are hallucinated | Triple-level annotation on 50-image subset |
| **Edit Rate** | How much caption changed | Levenshtein distance(before, after) / len(before) |
| **BLEU-4** | Fluency preservation | corrected caption vs. original reference caption |

### Baselines
1. **No correction** (raw BLIP-2 output) — lower bound
2. **Self-refinement** (prompt BLIP-2: "Look at this image again. Is your description accurate?")
3. **RelCheck (ours)**

### Ablation Study (one day, key for publishability)
- RelCheck **without** spatial verifier → shows spatial module contributes
- RelCheck **without** VQA verifier → shows action/attribute verification contributes
- RelCheck **without** correction step → detect-only, measures detection precision
- Full RelCheck → best performance

---

## Key Files
| File | Purpose |
|------|---------|
| `relcheck_main.py` | Main pipeline (currently empty — implement in Days 1–5) |
| `relcheck_blip2_probe.ipynb` | BLIP-2 captioning notebook (Colab, done) |
| `images/` | 12 pilot images (1.jpeg–12.webp) |
| `SiddhiPatil_CS297_Literature_Survey (1).pdf` | Literature survey |
| `Siddhi_Patil_CS298_Proposal.pdf` | Project proposal |
| `2406.16449v4.pdf` | R-Bench paper (ICML 2024) — key related work |
| `CONTEXT.md` | This file — update at end of every session |
| `figures/` | (To create) Architecture diagram + result figures |
| `eval/` | (To create) Evaluation scripts and results CSVs |

---

## REVISED 21-Day Task Plan

### WEEK 1: Build RelCheck (March 11–17)

| Day | Date | Task | Deliverable |
|-----|------|------|-------------|
| 1 | Mar 11 | Setup GitHub repo + env; scaffold `relcheck_main.py`; implement Triple Extractor (spaCy) | Working triple extractor on 5 test captions |
| 2 | Mar 12 | Implement VQA Verifier: BLIP-2 Yes/No probes for action + attribute relations | `verify_triple()` function returning True/False |
| 3 | Mar 13 | Implement Spatial Verifier: OWL-ViT bounding box extraction + geometric checks (IoU, centroid above/below/left/right) | Spatial verification for at least 3 relation types |
| 4 | Mar 14 | Implement Minimal Corrector using Mistral-7B via Together.ai; add self-consistency guard | End-to-end correction on single caption |
| 5 | Mar 15 | **Pilot demo:** Run full pipeline on 12 `images/` folder images; save before/after captions to CSV | `pilot_results.csv` with captions + corrections |
| 6 | Mar 16 | Draw architecture diagram (use draw.io or matplotlib); write module docstrings; clean up code | `figures/architecture.png` |
| 7 | Mar 17 | Push clean code to GitHub with README explaining setup + usage | Public GitHub repo |

### WEEK 2: Evaluate & Experiment (March 18–24)

| Day | Date | Task | Deliverable |
|-----|------|------|-------------|
| 8 | Mar 18 | Clone R-Bench repo; download 200-image subset from nocaps; adapt R-Bench questions to evaluation harness | `eval/rbench_subset.json` (200 images + questions) |
| 9 | Mar 19 | Implement R-POPE scoring script: run BLIP-2 on all 200 R-Bench images (no correction), record Yes/No answers vs. ground truth | `eval/baseline_no_correction.csv` + R-POPE score |
| 10 | Mar 20 | Run self-refinement baseline on 200 images; record results | `eval/baseline_self_refine.csv` + R-POPE score |
| 11 | Mar 21 | Run full RelCheck on 200 images; record results + edit rates | `eval/relcheck_results.csv` + all metrics |
| 12 | Mar 22 | Ablation study: run 3 ablated versions (no spatial, no VQA, detect-only) | `eval/ablation_results.csv` |
| 13 | Mar 23 | Manually annotate 50 images for R-CHAIR (relational ground truth labels); compute R-CHAIR_s and R-CHAIR_i | `eval/rchair_labels.csv` + R-CHAIR scores |
| 14 | Mar 24 | Compile ALL results into final tables and figures (matplotlib); select 5 qualitative before/after examples | `figures/results_table.png`, `figures/qualitative_examples.png` |

### WEEK 3: Write the Report (March 25–31)

| Day | Date | Task | Deliverable |
|-----|------|------|-------------|
| 15 | Mar 25 | Write Abstract (200 words) + Introduction (1.5 pages): problem, motivation, contributions | Draft intro section |
| 16 | Mar 26 | Write Related Work (2 pages): pull from lit survey — Woodpecker, POPE, R-Bench, SpatialVLM, VOLCANO | Draft related work section |
| 17 | Mar 27 | Write System Design (2–3 pages): describe all 3 stages with architecture diagram embedded | Draft system section |
| 18 | Mar 28 | Write Experimental Setup (dataset, baselines, metrics) + Results (tables with R-POPE, ablation) | Draft experiments + results section |
| 19 | Mar 29 | Write Analysis + Discussion: what works, what fails, failure mode taxonomy (from Day 13 annotation) | Draft analysis section |
| 20 | Mar 30 | Write Conclusion + Future Work + compile full References; assemble complete draft | Full draft PDF |
| 21 | Mar 31 | Final review: check all figures, fix formatting, proofread, submit | **FINAL SUBMISSION** |

---

## Session Log (Update Every Session!)

### Session 1 — March 11, 2026
- Read all source documents (lit survey, proposal, R-Bench paper, existing code)
- Compared with GaLoRA (NeurIPS 2025 Workshop) and PhotoProof (SJSU Spring 2025)
- Created/updated CONTEXT.md with revised plan
- Key decisions: scope to 200 R-Bench images, use OWL-ViT (not Grounding DINO), use Mistral-7B (not GPT-3.5), 2 baselines instead of 3, add GitHub day
- STATUS: Ready to start Day 1 coding tasks

### Session 2 — March 13, 2026
- Confirmed that all pipeline modules exist: triple_extractor.py, relation_verifier.py, corrector.py, relcheck_pipeline.py
- **Key context**: Siddhi has not yet run any code; implementation exists but is unverified
- **New workflow strategy**: Minimize manual work; Claude handles all code; user just runs Colab
- **Setup files created**:
  - `README.md` — project overview and quickstart
  - `requirements.txt` — all pip dependencies
  - `setup.sh` — one-command GitHub push script
  - `relcheck/__init__.py` — makes relcheck/ a proper Python package
  - `eval/` and `figures/` directories created
  - `RelCheck_Master.ipynb` — complete Colab notebook covering ALL experiments
- **RelCheck_Master.ipynb sections**:
  - Section 0: Setup (GPU check, pip install, GitHub clone, API key)
  - Section 1: Load Models (BLIP-2 in 8-bit, VQA monkey-patch to share model)
  - Section 2: Pilot demo on 12 images → pilot_results.csv
  - Section 3: R-Bench download (200-image subset, nocaps images)
  - Section 4: Baseline 1 (no correction) → R-POPE metrics
  - Section 5: Baseline 2 (self-refinement) → R-POPE metrics
  - Section 6: Full RelCheck → all metrics including BLEU-4, edit rate
  - Section 7: Ablation study (no_spatial, no_vqa, detect_only)
  - Section 8: R-CHAIR annotation helper + metric computation
  - Section 9: All figures (bar charts, ablation, by-relation, qualitative)
- **Next steps for Siddhi**:
  1. Create GitHub repo "RelCheck" (public, no README)
  2. Update GITHUB_USERNAME in setup.sh, run `bash setup.sh`
  3. Open RelCheck_Master.ipynb in Colab Pro (T4 or A100)
  4. Paste Together.ai API key in Section 0 Cell 4
  5. Update GITHUB_REPO_URL in Section 0 Cell 5
  6. Runtime → Run all
- STATUS: Colab notebook ready. Waiting on GitHub setup and first run.

---

## Publishable Originality Checklist
- [ ] RelCheck is the first **triple-level post-hoc** relational hallucination corrector
- [ ] R-POPE: novel metric extending POPE to relational queries — computed on R-Bench
- [ ] R-CHAIR_s + R-CHAIR_i: novel metrics for relational hallucination rate in captions
- [ ] Taxonomy of 3 relation hallucination types with empirical frequency from R-Bench subset
- [ ] Minimal edit strategy preserves fluency (quantified via BLEU-4 + edit rate)
- [ ] Ablation study proves each module contributes independently
- [ ] 5 qualitative before/after examples in paper

---

## Technical Stack
- **Python 3.10+**
- **spaCy** (en_core_web_sm) — dependency parsing for triple extraction
- **transformers** (HuggingFace) — BLIP-2 for captioning + VQA verification
- **OWL-ViT** (google/owlvit-base-patch32) — open-vocab object detection for bounding boxes
- **Together.ai API** (Mistral-7B-Instruct) — open-source LLM for correction step (free tier)
- **python-Levenshtein** — edit rate computation
- **nltk** — BLEU-4 scoring
- **pandas, matplotlib, seaborn** — results analysis + figures

---

## Important Research Gaps to Highlight in Report
1. R-Bench (Wu et al., ICML 2024) benchmarks relational hallucinations but provides **zero correction** — RelCheck fills this gap
2. Woodpecker corrects object hallucinations but has no relational structure (no (s,r,o) triples)
3. Spatial VLMs require retraining; RelCheck is plug-and-play post-hoc on any frozen MLLM
4. No prior work has defined or measured R-POPE or R-CHAIR systematically

---

## Notes / Decisions Made
- **OWL-ViT** chosen over Grounding DINO: same open-vocab detection, simpler HuggingFace install, no custom repo needed
- **Mistral-7B via Together.ai** for correction: fully open-source + reproducible (free tier: 5M tokens/month)
- **Scope: 200 R-Bench images** — clearly stated as compute limitation; note full-scale eval as future work
- **2 baselines** (no-correction + self-refinement) is sufficient; cite Woodpecker numbers from paper rather than reimplementing
- Architecture diagram to be created on Day 6 using draw.io (free, browser-based)
- GitHub repo to be made public on Day 7 for open-source deliverable requirement

