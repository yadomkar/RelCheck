# Failed Run — Qwen3.5-397B as VLM + Llama-3.3-70B as LLM

**Date:** 2026-04-04
**Captioner:** LLaVA-1.5-7B
**LLM:** meta-llama/Llama-3.3-70B-Instruct-Turbo
**VLM:** Qwen/Qwen3.5-397B-A17B
**Result:** FAILED — thinking model incompatible with yes/no VQA parsing

## Why it failed

Qwen3.5 is a "thinking" model that wraps answers in `<think>...</think>` tags.
The VQA parser in `_vqa.py` expects clean "Yes"/"No" responses with max_tokens=5.
The thinking tokens consume the token budget, and the hedged answers fall in the
uncertain zone (0.40–0.65 yes_ratio), producing UNKNOWN verdicts for nearly all triples.

## Evidence

- 78/81 spatial verdicts = UNKNOWN (only 3 CORRECT from KB synonym match)
- 18/18 action verdicts = UNKNOWN
- 14/14 attribute verdicts = UNKNOWN
- 0 INCORRECT verdicts → 0 guidance → 0 corrections
- 0% batch acceptance (nothing to correct)
- 20/20 CHANGED only from spatial addendum, not actual correction

## Lesson

Thinking models (Qwen3.5, DeepSeek R1) require stripping `<think>` tags and
bumping max_tokens across all VLM call sites before they can be used as VQA verifiers.
