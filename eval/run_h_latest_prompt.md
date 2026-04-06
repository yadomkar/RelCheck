# Run H: Clean KB + Correction Cap + Garble Detection + Anti-Garble Prompt

**Date:** 2026-04-05
**Config:** Clean KB (batch_size=4 + compound splitting), NLI enabled, SKIP_KB_GEOMETRY=True, MAX_CORRECTIONS_PER_BATCH=5, expanded has_garble(), anti-garble prompt rules 9-10
**What's NOT active:** NLI_USE_VISUAL_DESCRIPTION is still ON (5 visual_description evidence items found)

## Which Fixes Are Active

| Fix | Active? | Evidence |
|---|---|---|
| Clean KB (compound label splitting) | ✓ | 0/344 compound labels |
| Correction cap (MAX=5) | ✓ | Max 5 corrections in any image (was 10 in Run G) |
| Expanded garble detection | ✓ | 5/18 garble detected (was 0/18 in all prior runs) |
| Anti-garble prompt rules | ✓ | New prompt with rules 9-10 |
| NLI_USE_VISUAL_DESCRIPTION=False | ✗ NOT ACTIVE | 5 visual_description evidence items still present |

## Key Metrics

| Metric | Run F (dirty KB) | Run G (clean KB) | Run H (all fixes) |
|---|---|---|---|
| Corrected Accuracy | 72.2% | 61.1% | 72.2% |
| Recovery Rate | 79% (11/14) | 64% (9/14) | 71% (10/14) |
| Supplemental (corr−corrup) | +6.7% | +6.7% | +0.0% |
| Batch acceptance | 94.4% | 94.4% | 66.7% |
| Fallback deletion | 5.6% | 5.6% | 33.3% |
| Garble detected | 0/18 | 0/18 | 5/18 |
| Max corrections/image | 9 | 10 | 5 |

## Impact of Each Fix

### Correction Cap (MAX=5): WORKING, MIXED IMPACT

The cap is active. Images that had 7-10 corrections in Run G now have exactly 5. The overflow goes to fallback deletion.

| img_id | Run G errors | Run H errors | Change |
|---|---|---|---|
| 0f25f9ba117d9552 | 7 | 5 | capped |
| 1109ae1bfb4efe78 | 8 | 5 | capped |
| 55f1af9a02042d09 | 10 | 5 | capped |
| 584110c20a4695d9 | 7 | 5 | capped |

But the cap didn't prevent garble — 5 images with exactly 5 corrections still garbled. Llama garbles even at 5 corrections. The cap helped (10→5 is better) but 5 is still too many for some images.

### Garble Detection: WORKING, BUT CAUSES AGGRESSIVE DELETION

5/18 images had garble detected → batch rejected → fallback deletion. This is new — Run G had 0 detections. The expanded patterns are catching real garbles.

But the fallback is devastating:
- 584110c20a4695d9: 109→10 words (-91%)
- 55f1af9a02042d09: 108→10 words (-91%)
- 79a6209e9b93590d: 95→28 words (-71%)

When 5 corrections trigger fallback, all 5 sentences get deleted. The caption is gutted.

### Anti-Garble Prompt: NOT ENOUGH

Despite rules 9-10 telling Llama not to produce garbled spatial phrases, 5/18 accepted captions still have garbles:
- "jockeys rider and mount horses" — prompt didn't prevent this
- "jockeys wear helmets. helmets" — repeated phrase
- "woman above a black leather outfit" — wrong spatial replacement
- "woman below the room" — wrong spatial replacement
- "below the floor" — the exact pattern the prompt warned against

The prompt rules are not strong enough to override Llama's tendency to garble when given 5 simultaneous edits.

### Visual Description Still On: CONTRIBUTING TO FALSE NEGATIVES

5 visual_description evidence items are still in the NLI checks. Mixed evidence is at 21 (down from 45 in Run G, but still present). 10 false negatives, all from mixed evidence. Disabling visual description should reduce mixed further.

## The Real Problem

The correction cap + garble detection created a new failure mode: **aggressive fallback deletion**. When the batch corrector garbles (which happens at 5 corrections), the entire batch is rejected and all error sentences are deleted. This produces captions with 10-28 words from originals of 95-109 words.

This explains the supplemental accuracy drop to +0.0%: the gutted captions can't answer supplemental R-Bench questions correctly because most of the descriptive content is gone.

## What Would Actually Help

1. **Lower the cap to 3-4** — Llama garbles at 5. Try 3 or 4 corrections per batch.

2. **Disable visual description** — still contributing to mixed evidence and false negatives. The flag exists but wasn't active in this run.

3. **Smarter fallback** — instead of deleting all 5 sentences when garble is detected, try a second batch call with fewer corrections (top 2-3 only). Or use regex cleanup on the garbled output instead of rejecting it entirely.

4. **The fundamental tension**: NLI finds 5-10 errors per image, but the batch corrector can only handle 3-4 cleanly. The gap between detection capability and correction capability is the bottleneck.
