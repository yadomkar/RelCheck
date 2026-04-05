# Run — NLI + Embedding Entity Matching

**Date:** 2026-04-05
**Changes:** NLI KB verification + embedding-based entity matching in find_best_bbox_from_kb

## Key Metrics vs Previous Best (run C: no addendum)

| Metric | Run C | This Run | Delta |
|---|---|---|---|
| Bbox coverage | 52.6% | 72.7% | +20.1% ✅ |
| Geo check possible | 15.2% (5/33) | 22.6% (7/31) | +7.4% ✅ |
| Keypoints loaded | 12.1% (4/33) | 16.1% (5/31) | +4% ✅ |
| Geo-VQA agreement | 60% | 85.7% | +25.7% ✅ |
| Geo confirmed | 2 | 5 | +3 ✅ |
| KB-first correct rel | 35.3% | 46.6% | +11.3% ✅ |
| Mean spatial facts | 30.6 | 190.1 | +159.5 (more detections) |
| Mean detections | 13.5 | 20.0 | +6.5 |
| Corrected accuracy | 65.0% | 60.0% | -5% ❌ |
| Recovery | 69% (11/16) | 62% (10/16) | -7% ❌ |
| Supplemental | +15.6% | +9.4% | -6.2% ❌ |

## What improved
- Bbox coverage +20% — embedding matcher finding entities that word match missed
- Geometry checks firing more often (22.6% vs 15.2%)
- Geo-VQA agreement much higher (85.7% vs 60%) — better bbox matches = better geometry
- KB-first correct rel rate up to 46.6% — more KB spatial facts available

## What got worse
- Corrected accuracy dropped 5%
- Recovery dropped 7%
- Supplemental dropped 6.2%
- Mean spatial facts exploded to 190 (from 30.6) — more detections = more spatial fact pairs

## Likely cause
More detections (20 vs 13.5) means more spatial facts (190 vs 30.6), which means
more triples get flagged as INCORRECT by the spatial fact matching. The pipeline is
being too aggressive — correcting things that were actually correct in the original
caption. The garbled corrections visible in the R-POPE output confirm this:
- "two bowls right the table" — garbled
- "carrots scattered left the plate" — garbled
- "The man in the image is going downhill" — NOT RECOVERED, still present
- "buildings in the image" — NOT RECOVERED, still present

## R-POPE Per-Image

| img_id | Type | Status |
|---|---|---|
| 2e5f02731ae7036d | SPATIAL | ✗ NOT RECOVERED |
| 0e5613a5e521f5c5 | SPATIAL | ✓ RECOVERED |
| 79a6209e9b93590d | ATTRIBUTE | ✓ RECOVERED |
| 635ac16656fe05fb | ATTRIBUTE | — NOT DETECTED |
| 584110c20a4695d9 | SPATIAL | ✓ RECOVERED (NEW!) |
| 3a39040ecd47e0f6 | ACTION | — NOT DETECTED |
| 2c4004388f263f72 | ATTRIBUTE | ✗ NOT RECOVERED |
| 30ace30967aca271 | ACTION | ✓ RECOVERED |
| 25f91afe0f75cba2 | ACTION | ✓ RECOVERED |
| c542d605c0024365 | ATTRIBUTE | ✗ NOT RECOVERED |
| 1109ae1bfb4efe78 | ACTION | ✓ RECOVERED |
| 1050b3c3f36090a3 | ACTION | — NOT DETECTED |
| 273bf8ea1ddb78c0 | ATTRIBUTE | ✗ NOT RECOVERED |
| 55f1af9a02042d09 | ATTRIBUTE | ✗ NOT RECOVERED (was RECOVERED before!) |
| 5e87eaf2ecc31207 | ACTION | ✓ RECOVERED |
| eac4dfdf82776151 | ATTRIBUTE | ✓ RECOVERED |
| 0f25f9ba117d9552 | ATTRIBUTE | ✗ NOT RECOVERED (was RECOVERED before!) |
| 9bf02cc9cf9f8f97 | ACTION | ✓ RECOVERED |
| 4ece4336cb293790 | ACTION | — NOT DETECTED |
| c3fd5f3c65c29839 | ACTION | ✓ RECOVERED |

## Regressions (images that were RECOVERED before but aren't now)
- 55f1af9a02042d09: "man going downhill" — correction changed "holding a microphone" to something garbled, and "going downhill" is still present
- 0f25f9ba117d9552: "buildings in image" — correction changed spatial relations but left "buildings" in caption

## New recovery
- 584110c20a4695d9: "drawer on left side of sink" — NOW RECOVERED (was NOT RECOVERED before). The better bbox coverage helped here.
