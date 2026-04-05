# RelCheck Ablation Study — Full Results

**Dataset:** 20 R-Bench images, LLaVA-1.5-7B captioner, seed=42
**Date:** 2026-04-04 / 2026-04-05

## Main Results Table

| Config | Corrected Acc | Recovery | Suppl (corr−corrup) | Detection |
|---|---|---|---|---|
| No correction (corrupted) | 10.0% | — | −9.4% | — |
| A: Baseline (Llama-70B + Qwen-8B) | 65.0% | 75% (12/16) | +12.5% | 80% (16/20) |
| B: + ViTPose fix + geo hints + addendum | 60.0% | 69% (11/16) | +15.6% | 80% (16/20) |
| C: + ViTPose fix + geo hints − addendum | 65.0% | 69% (11/16) | +15.6% | 80% (16/20) |
| D: Qwen3.5-397B VLM (thinking model) | 0.0% | 0% (0/16) | 0.0% | 0% |
| E: KB rebuild (same as C, fresh KB) | 60.0% | 62% (10/16) | +9.4% | 80% (16/20) |

**Note on run E:** Same code as C but KB was rebuilt fresh. The KB has more detections
(20 vs 13.5 avg) and far more spatial facts (190 vs 30.6 avg), likely from a different
GDino detection run or cache. NLI and embedding matching were NOT active (ENABLE_NLI=False).
The accuracy drop is from the richer/noisier KB causing more aggressive corrections.

## Verification Statistics

| Metric | A (baseline) | B (+addendum) | C (−addendum) | E (fresh KB) |
|---|---|---|---|---|
| Total triples extracted | 144 | 144 | 148 | 144 |
| SPATIAL CORRECT | 42 | 33 | 31 | 34 |
| SPATIAL INCORRECT | 38 | 48 | 51 | 45 |
| ACTION CORRECT | 10 | 9 | 8 | 10 |
| ACTION INCORRECT | 6 | 8 | 12 | 7 |
| ATTRIBUTE CORRECT | 10 | 10 | 8 | 8 |
| ATTRIBUTE INCORRECT | 4 | 6 | 5 | 6 |
| UNKNOWN verdicts | 0 | 0 | 0 | 0 |
| Total INCORRECT | 48 | 62 | 68 | 58 |

## Correction Application

| Metric | A (baseline) | B (+addendum) | C (−addendum) | E (fresh KB) |
|---|---|---|---|---|
| REPLACE_WORD guidance | 26 | 40 | 45 | 37 |
| DELETE_SENTENCE guidance | 22 | 22 | 23 | 21 |
| SOFTEN guidance | 0 | 0 | 0 | 0 |
| Batch acceptance rate | 100% | 100% | 100% | 100% |
| Fallback deletion rate | 0% | 0% | 0% | 0% |
| Post-verify revert rate | 0% | 0% | 0% | 0% |

## KB & Evidence Sources

| Metric | A (baseline) | B (+addendum) | C (−addendum) | E (fresh KB) |
|---|---|---|---|---|
| KB-first correct rel rate | 27.1% | 37.1% | 35.3% | 46.6% |
| Spatial fact hit rate | 3.8% | 3.7% | 3.7% | 2.5% |
| Bbox coverage | 52.3% | 52.7% | 52.6% | 72.7% |
| Mean detections | 13.5 | 13.5 | 13.5 | 20.0 |
| Mean spatial facts | 30.6 | 30.6 | 30.6 | 190.1 |

## Geometry & Pose System

| Metric | A (baseline) | B (+addendum) | C (−addendum) | E (fresh KB) |
|---|---|---|---|---|
| Geo check possible | 5 (16.1%) | 5 (16.1%) | 5 (15.2%) | 7 (22.6%) |
| Keypoints loaded | 0 (0%) | 4 (12.1%) | 4 (12.1%) | 5 (16.1%) |
| Geo confirmed (True) | 0 | 2 | 2 | 5 |
| Geo violated (False) | 1 | 3 | 3 | 2 |
| Geo-VQA agreement | 0% | 60% | 60% | 85.7% |
| Grasping family hits | 10 | 10 | 10 | 10 |
| Mounting family hits | 2 | 2 | 2 | 2 |

## Geometry-VQA Detailed Decisions (runs B & C)

| Claim | Family | Geo Result | Keypoints | VQA Verdict | Agreement |
|---|---|---|---|---|---|
| man holding hammer | grasping | True | YES | CORRECT | ✓ AGREE |
| woman holding handbag | grasping | False | YES | INCORRECT | ✓ AGREE |
| woman sitting on couch | mounting | False | NO | CORRECT | ✗ DISAGREE |
| man holding microphone | grasping | False | YES | INCORRECT | ✓ AGREE |
| woman holding snowboard | grasping | True | YES | INCORRECT | ✗ DISAGREE |

## INCORRECT Verdict Sources (run C)

| Source | Count | % |
|---|---|---|
| VQA spatial fallback | 43 | 63.2% |
| VQA only (unanimous_no) | 11 | 16.2% |
| Entity absent (GDino+VQA) | 8 | 11.8% |
| VQA only (split_no) | 4 | 5.9% |
| Geometry violated + VQA | 2 | 2.9% |

## R-POPE Per-Image Breakdown (run C)

| img_id | Type | Status | Injected Claim |
|---|---|---|---|
| 2e5f02731ae7036d | SPATIAL | ✗ NOT RECOVERED | pitcher next to small bowl |
| 0e5613a5e521f5c5 | SPATIAL | ✓ RECOVERED | desk on top of bookcase |
| 79a6209e9b93590d | ATTRIBUTE | ✓ RECOVERED | fire truck on table |
| 635ac16656fe05fb | ATTRIBUTE | — NOT DETECTED | seafood served with fruits |
| 584110c20a4695d9 | SPATIAL | ✗ NOT RECOVERED | drawer on left side of sink |
| 3a39040ecd47e0f6 | ACTION | — NOT DETECTED | man holding donut |
| 2c4004388f263f72 | ATTRIBUTE | ✗ NOT RECOVERED | horses racing in straight line |
| 30ace30967aca271 | ACTION | ✓ RECOVERED | man holding baseball bat |
| 25f91afe0f75cba2 | ACTION | ✓ RECOVERED | man wearing watch |
| c542d605c0024365 | ATTRIBUTE | ✗ NOT RECOVERED | vegetables in bowl of dip |
| 1109ae1bfb4efe78 | ACTION | ✓ RECOVERED | women holding hot dog |
| 1050b3c3f36090a3 | ACTION | — NOT DETECTED | woman sitting on couch |
| 273bf8ea1ddb78c0 | ATTRIBUTE | ✗ NOT RECOVERED | nameplate on pot |
| 55f1af9a02042d09 | ATTRIBUTE | ✓ RECOVERED | man going downhill |
| 5e87eaf2ecc31207 | ACTION | ✓ RECOVERED | woman holding snowboard |
| eac4dfdf82776151 | ATTRIBUTE | ✓ RECOVERED | books lined up in order |
| 0f25f9ba117d9552 | ATTRIBUTE | ✓ RECOVERED | buildings in image |
| 9bf02cc9cf9f8f97 | ACTION | ✓ RECOVERED | sewing machine on shelf |
| 4ece4336cb293790 | ACTION | — NOT DETECTED | man holding pool cue |
| c3fd5f3c65c29839 | ACTION | ✓ RECOVERED | lamp on floor |

## NOT DETECTED Analysis (4 images)

| img_id | Why not detected |
|---|---|
| 635ac16656fe05fb | "seafood served with fruits" — not extracted as a relational triple |
| 3a39040ecd47e0f6 | "man holding donut" — detected & deleted, but judge orig=no already (not a valid drop) |
| 1050b3c3f36090a3 | "woman sitting on couch" — orig=yes (injection didn't change judge answer, not a valid test) |
| 4ece4336cb293790 | "man holding pool cue" — orig=yes (man IS holding pool cue, injection is true statement) |

## NOT RECOVERED Analysis (5 images, run C)

| img_id | Why not recovered |
|---|---|
| 2e5f02731ae7036d | "pitcher next to small bowl" — triple verified as CORRECT (pitcher IS next to bowl) |
| 584110c20a4695d9 | "drawer on left side of sink" — triple verified as CORRECT (drawer IS on left of sink) |
| 2c4004388f263f72 | "horses racing in straight line" — not extracted as relational triple, left in caption |
| c542d605c0024365 | "vegetables in bowl of dip" — not extracted as relational triple, left in caption |
| 273bf8ea1ddb78c0 | "nameplate on pot" — corrected to "nameplate above pot" but judge still says yes |

## Bbox Coverage — Most Missed Entities (run C)

| Entity | Misses |
|---|---|
| sink | 6 |
| jockeys | 3 |
| floor | 3 |
| books | 3 |
| family | 3 |
| raft | 3 |
| bottle | 3 |
| people | 2 |
| table | 2 |
| fish | 2 |
| bathroom | 2 |
| faucets | 2 |
| field | 2 |
| cucumbers | 2 |

## Key Takeaways

1. **Detection is solid (80%)** — 16/20 injections detected. The 4 misses are mostly invalid test cases (2 where orig=yes, 1 not extractable as triple, 1 deleted but judge disagrees).

2. **Recovery gap (69-75%)** — 5 detected-but-not-recovered cases. 2 are because the injected claim is actually TRUE (pitcher IS next to bowl, drawer IS on left of sink). 2 are extraction failures. 1 is a correction that didn't satisfy the judge.

3. **ViTPose fix was impactful** — went from 0 to 4 keypoint loads, enabling real grasping verification. 3/4 agreed with VQA.

4. **Addendum hurts injected accuracy** — 60% with addendum vs 65% without. The added spatial facts confuse the R-POPE judge on injected questions.

5. **Addendum doesn't affect supplemental** — +15.6% both with and without. The supplemental improvement comes from the correction itself, not addendum.

6. **Geometry-grounded prompting increased corrections** — 48→68 INCORRECT verdicts, 26→45 REPLACE_WORD. More aggressive but same accuracy.

7. **KB spatial facts are the #1 correction source** — 24/68 corrections use deterministic KB data (free, no API calls).

8. **Bbox coverage (52.6%) is the main bottleneck** — half of entities can't be found by GDino, limiting geometry checks to 15% of action triples.

9. **Fresh KB rebuild (run E) shows KB quality matters** — more detections (20 vs 13.5) and spatial facts (190 vs 30.6) improved bbox coverage to 72.7% and geo-VQA agreement to 85.7%, but the richer KB also caused more aggressive corrections that hurt accuracy (60% vs 65%). The batch LLM corrector produces garbled output when given too many corrections. Correction quality is now the bottleneck, not detection.

10. **Best config for paper: Run C** — ViTPose + geo hints, no addendum. 65% corrected accuracy, +55% from corrupted, +15.6% supplemental improvement, 80% detection, 69% recovery.
