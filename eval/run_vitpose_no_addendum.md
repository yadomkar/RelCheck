# Run — ViTPose Fix + Geometry Hints + No Addendum

**Date:** 2026-04-04
**Captioner:** LLaVA-1.5-7B
**LLM:** meta-llama/Llama-3.3-70B-Instruct-Turbo
**VLM:** Qwen/Qwen3-VL-8B-Instruct
**Changes:** ViTPose fix, geometry-grounded VQA prompting, include_addendum=False

## Ablation Table (all runs)

| Config | Corrected Acc | Recovery | Supplemental | INCORRECT found |
|---|---|---|---|---|
| Baseline (Llama+Qwen8B) | 65.0% | 75% (12/16) | +12.5% | 48 |
| + ViTPose + geo hints + addendum | 60.0% | 69% (11/16) | +15.6% | 62 |
| + ViTPose + geo hints − addendum | 65.0% | 69% (11/16) | +15.6% | 68 |
| Qwen3.5-397B VLM (failed) | 0.0% | 0% | 0% | 0 |

## Key Findings

1. Addendum was NOT helping injected-question accuracy — removing it brought
   corrected accuracy back from 60% to 65% (matching baseline)
2. Addendum was NOT hurting supplemental accuracy — still +15.6% without it
3. The supplemental improvement (+15.6% vs +12.5%) comes from the geometry-grounded
   prompting and ViTPose, not from addendum
4. Recovery dropped from 75% to 69% — the geometry hints are causing 1 extra
   false positive (flagging a correct relation as wrong)
5. More INCORRECT verdicts found (68 vs 48) — pipeline is more aggressive,
   catching more real errors in the pre-existing LLaVA hallucinations

## Per-Image R-POPE Detail

| img_id | type | status | issue |
|---|---|---|---|
| 2e5f02731ae7036d | SPATIAL | NOT RECOVERED | "pitcher next to bowl" still in corrected caption |
| 0e5613a5e521f5c5 | SPATIAL | RECOVERED | "desk on bookcase" → "desk near bookcase" |
| 79a6209e9b93590d | ATTRIBUTE | RECOVERED | "fire truck on table" deleted |
| 635ac16656fe05fb | ATTRIBUTE | NOT DETECTED | "seafood served with fruits" — not extracted as triple |
| 584110c20a4695d9 | SPATIAL | NOT RECOVERED | "drawer on left side of sink" still present |
| 3a39040ecd47e0f6 | ACTION | NOT DETECTED | "holding donut" deleted but judge still says yes |
| 2c4004388f263f72 | ATTRIBUTE | NOT RECOVERED | "horses racing in straight line" still present |
| 30ace30967aca271 | ACTION | RECOVERED | "holding baseball bat" deleted |
| 25f91afe0f75cba2 | ACTION | RECOVERED | "wearing watch" deleted |
| c542d605c0024365 | ATTRIBUTE | NOT RECOVERED | "vegetables in bowl of dip" still present |
| 1109ae1bfb4efe78 | ACTION | RECOVERED | "holding hot dog" deleted |
| 1050b3c3f36090a3 | ACTION | NOT DETECTED | orig=yes (not a valid injection) |
| 273bf8ea1ddb78c0 | ATTRIBUTE | NOT RECOVERED | "nameplate on pot" → "nameplate above pot" |
| 55f1af9a02042d09 | ATTRIBUTE | RECOVERED | "going downhill" deleted |
| 5e87eaf2ecc31207 | ACTION | RECOVERED | "holding snowboard" → "near snowboard" |
| eac4dfdf82776151 | ATTRIBUTE | RECOVERED | "books lined up in order" deleted |
| 0f25f9ba117d9552 | ATTRIBUTE | RECOVERED | "buildings in image" deleted |
| 9bf02cc9cf9f8f97 | ACTION | RECOVERED | "sewing machine on shelf" deleted |
| 4ece4336cb293790 | ACTION | NOT DETECTED | orig=yes (not a valid injection) |
| c3fd5f3c65c29839 | ACTION | RECOVERED | "lamp on floor" deleted |

## Correction Quality Issues Observed

- "two bowls right the table" — garbled spatial replacement
- "plants on floor" → "plants below the floor" — nonsensical correction
- "woman kneeling is above a handbag" — wrong spatial replacement
- "table is situated left a brown book" — garbled output
- Some corrections replace spatial words with raw prepositions without articles
