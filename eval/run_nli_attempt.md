# Run — NLI Attempt (NLI did NOT fire)

**Date:** 2026-04-05
**Changes:** NLI code deployed but enable_nli defaulted to False (bug — not wired to config)
**Evidence:** All 113 verifications have nli_verdict=None
**Fix applied:** Changed `enable_nli: bool = False` → `enable_nli: bool = ENABLE_NLI` in surgical/__init__.py

## Results (without NLI actually firing)

| Metric | Value |
|---|---|
| Corrected accuracy | 70.0% (best so far) |
| Recovery | 75% (12/16) |
| Supplemental | +9.4% |
| Detection | 80% (16/20) |

## Why 70% (up from 65%)

The +5% comes from exactly ONE image: `584110c20a4695d9` (sink/drawer).
- Run C: "drawer on left side of sink" verified as CORRECT → NOT RECOVERED
- This run: corrected to "drawer is below the sink" → RECOVERED
- Cause: richer KB (190 spatial facts vs 30.6) provided a different spatial relation

ACTION (7/9) and ATTRIBUTE (5/8) results are identical to run C.
SPATIAL went from 1/3 → 2/3 — that one image accounts for the entire improvement.

## NLI was NOT active
The `enable_nli` parameter in `correct_long_caption()` defaulted to `False`.
The config had `ENABLE_NLI = True` but the function didn't read from config.
Fixed by importing ENABLE_NLI and using it as the default value.

## Key Numbers
- SPATIAL: 34C / 46I / 0U
- ACTION: 11C / 10I / 0U
- ATTRIBUTE: 7C / 5I / 0U
- Total INCORRECT: 61
- REPLACE_WORD: 40, DELETE_SENTENCE: 21
- Bbox coverage: 73.0%
- Geo-VQA agreement: 85.7%
- KB-first correct rel: 44.3%
- Correction sources: spatial_kb=27, none(DELETE)=17, vlm_query=10, action_3stage=7

## Garbled corrections still present
- "two bowls right the table"
- "carrots scattered left the plate"
- "below the floor"
- "is incorrect, instead:"
