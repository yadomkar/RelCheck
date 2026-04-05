# Run — ViTPose Fix + Geometry-Grounded VQA Prompting

**Date:** 2026-04-04
**Captioner:** LLaVA-1.5-7B
**LLM:** meta-llama/Llama-3.3-70B-Instruct-Turbo
**VLM:** Qwen/Qwen3-VL-8B-Instruct
**Detection:** GroundingDINO (grounding-dino-tiny)
**Changes:** Fixed ViTPose model ID (VitPoseForPoseEstimation), added geometry-grounded VQA prompting, richer bbox context hints

## Comparison vs Baseline

| Metric | Baseline | This Run | Delta |
|---|---|---|---|
| SPATIAL INCORRECT | 38 | 48 | +10 |
| ACTION INCORRECT | 6 | 8 | +2 |
| ATTRIBUTE INCORRECT | 4 | 6 | +2 |
| REPLACE_WORD guidance | 26 | 40 | +14 |
| KB-first correct rel | 27.1% | 37.1% | +10% |
| Keypoints loaded | 0 (0%) | 4 (12.1%) | +4 |
| Geo confirmed | 0 | 2 | +2 |
| Geo violated | 0 | 3 | +3 |
| Geo-VQA agreement | 0% | 60% | +60% |
| Corrected accuracy | 65.0% | 60.0% | -5% |
| Recovery rate | 75% | 69% | -6% |
| Supplemental delta | +12.5% | +15.6% | +3.1% |

## Analysis

Pipeline is more aggressive — finding more errors and making more corrections.
But some extra corrections are false positives, hurting injected-question accuracy.
Supplemental accuracy improved (+15.6% vs +12.5%), suggesting the extra corrections
help on non-injected questions (fixing real pre-existing hallucinations in LLaVA captions).

The geometry-grounded prompting may be biasing VQA toward rejection when geometry
says False, causing some correct relations to be flagged as errors.
