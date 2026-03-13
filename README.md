# RelCheck: Relation-Aware Hallucination Detection and Correction for Multimodal LLMs

**CS298 Master's Project — SJSU Spring 2026**
**Student:** Siddhi Patil | **Advisor:** Dr. William Andreopoulos

---

## Overview

RelCheck is a **training-free, post-hoc** system that detects and corrects **relational hallucinations** in Multimodal Large Language Model (MLLM) outputs. Rather than rewriting entire captions, RelCheck:

1. **Stage 1 — Triple Extractor**: Parses MLLM captions into structured `(subject, relation, object)` triples using spaCy dependency parsing
2. **Stage 2 — Relation Verifier**: Verifies each triple against the image using relation-type-aware strategies:
   - *Spatial relations* (`on`, `above`, `left of`, …) → OWL-ViT bounding box detection + geometric checks
   - *Action/Attribute relations* (`holding`, `riding`, `is red`, …) → BLIP-2 VQA binary probes
3. **Stage 3 — Minimal Corrector**: Applies minimal edits to hallucinated triples using Mistral-7B-Instruct (via Together.ai), with a self-consistency guard

**Key novelty:** First system to target relation-level hallucinations explicitly. Introduces R-POPE and R-CHAIR metrics for relational hallucination evaluation.

---

## Quickstart (Google Colab)

1. Open **`RelCheck_Master.ipynb`** in Google Colab (Runtime → T4 GPU or A100)
2. In **Cell 4**, paste your Together.ai API key
3. In **Cell 5**, update the GitHub repo URL with your username
4. Click **Runtime → Run all**

Everything else is automated: model downloads, R-Bench data, evaluation, figure generation.

---

## Repository Structure

```
RelCheck/
├── relcheck/                   # Core pipeline modules
│   ├── triple_extractor.py     # Stage 1: spaCy triple extraction
│   ├── relation_verifier.py    # Stage 2: BLIP-2 VQA + OWL-ViT spatial
│   ├── corrector.py            # Stage 3: Mistral-7B minimal correction
│   └── relcheck_pipeline.py    # End-to-end wiring + batch runner
├── eval/                       # Evaluation scripts and results (generated)
│   ├── pilot_results.csv
│   ├── baseline_no_correction.csv
│   ├── baseline_self_refine.csv
│   ├── relcheck_results.csv
│   └── ablation_results.csv
├── figures/                    # Generated figures (generated)
├── images/                     # 12 pilot images
├── RelCheck_Master.ipynb       # ← Master Colab notebook (run this)
├── requirements.txt
└── README.md
```

---

## Technical Stack

| Component | Library | Purpose |
|-----------|---------|---------|
| Triple extraction | spaCy `en_core_web_sm` | Dependency parsing → (s,r,o) triples |
| MLLM backbone | `Salesforce/blip2-flan-t5-xl` | Captioning + VQA verification |
| Spatial detection | `google/owlvit-base-patch32` | Open-vocab bounding box detection |
| LLM correction | Mistral-7B-Instruct (Together.ai) | Minimal caption edits |
| Evaluation dataset | R-Bench (Wu et al., ICML 2024) | Relational hallucination benchmark |

---

## Evaluation

RelCheck is evaluated on a 200-image subset of R-Bench (nocaps validation set):

| Metric | Description |
|--------|-------------|
| **R-POPE** | Accuracy / Precision / Recall / F1 on relational Yes/No questions |
| **R-CHAIR_s** | % captions with ≥1 relational hallucination |
| **R-CHAIR_i** | % of total triples that are hallucinated |
| **Edit Rate** | Levenshtein distance (before vs. after correction) / caption length |
| **BLEU-4** | Fluency preservation of corrected captions |

Baselines: (1) Raw BLIP-2 output, (2) BLIP-2 self-refinement.

---

## Citation

If you use this work, please cite:
```bibtex
@mastersthesis{patil2026relcheck,
  author  = {Siddhi Patil},
  title   = {RelCheck: Relation-Aware Hallucination Detection and Correction for Multimodal Large Language Models},
  school  = {San Jose State University},
  year    = {2026}
}
```

---

## Related Work

- **R-Bench** (Wu et al., ICML 2024): Relational hallucination benchmark — provides the evaluation data used in this project
- **Woodpecker** (Yin et al., 2023): Object-level hallucination correction — does not address relational structure
- **POPE** (Li et al., 2023): Binary object hallucination evaluation — R-POPE extends this to relations
