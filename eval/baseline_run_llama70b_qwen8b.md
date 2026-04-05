# Baseline Run — Llama-3.3-70B + Qwen3-VL-8B

**Date:** 2026-04-04
**Captioner:** LLaVA-1.5-7B
**LLM (extraction/correction):** meta-llama/Llama-3.3-70B-Instruct-Turbo
**VLM (VQA verification):** Qwen/Qwen3-VL-8B-Instruct
**Detection (bboxes):** GroundingDINO (IDEA-Research/grounding-dino-tiny)
**Images:** 20 (R-Bench subset, seed=42)
**Addendum:** enabled

## Per-Image Results

All 20/20 images modified. Total correction time: ~543s (~27s avg).

## Correction Path Summary

| Metric | Value |
|---|---|
| Triples per image (mean/median) | 7.2 / 7.0 |
| Spatial verdicts (C/I/U) | 42 / 38 / 0 |
| Action verdicts (C/I/U) | 10 / 6 / 0 |
| Attribute verdicts (C/I/U) | 10 / 4 / 0 |
| Top evidence: spatial | 80 |
| Top evidence: action_vqa | 30 |
| Guidance: REPLACE_WORD | 26 |
| Guidance: DELETE_SENTENCE | 22 |
| Guidance: SOFTEN | 0 |
| Batch acceptance rate | 100.0% |
| Fallback deletion rate | 0.0% |
| Post-verify revert rate | 0.0% |
| Addendum acceptance rate | 0.0% |

## KB Usage

| Metric | Value |
|---|---|
| Mean hard facts | 6.7 |
| Mean spatial facts | 30.6 |
| Mean visual desc len | 1034 |
| Mean detections | 13.5 |
| Spatial fact hit rate | 3.8% |
| Bbox coverage | 52.3% |
| KB-first correct rel | 27.1% |
| Addendum novelty | 13.4% |

## Path Effectiveness

| Source | Images | Modified |
|---|---|---|
| spatial | 14 | 100.0% |
| action_vqa | 4 | 100.0% |

## R-POPE Results (injected hallucinations, GT=no)

| Metric | Value |
|---|---|
| Images evaluated | 20 |
| Injection detected | 16/20 (80%) |
| Recoveries | 12/16 (75%) |
| Accuracy — Original | 90.0% |
| Accuracy — Corrupted | 10.0% (delta -80.0%) |
| Accuracy — Corrected | 65.0% (delta +55.0%) |

### By Relation Type

| Type | n | orig | corr | fixed | drops | rec |
|---|---|---|---|---|---|---|
| ACTION | 9 | 7/9 | 1/9 | 7/9 | 6 | 6 |
| ATTRIBUTE | 8 | 8/8 | 1/8 | 5/8 | 7 | 5 |
| SPATIAL | 3 | 3/3 | 0/3 | 1/3 | 3 | 1 |

## Supplemental (other R-Bench questions, same images)

| Metric | Value |
|---|---|
| Original | 46.9% |
| Corrupted | 40.6% (delta -6.2%) |
| Corrected | 53.1% (delta +12.5%) |
