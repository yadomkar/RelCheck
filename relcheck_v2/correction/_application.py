"""
RelCheck v2 — Correction Application
=======================================
Apply batch LLM correction, build per-error guidance strings,
and handle fallback deletion when batch correction fails.
"""

from __future__ import annotations

import re

import pysbd
from PIL import Image

from .._logging import log
from ..api import llm_call
from ..config import CORRECTION_LENGTH_RATIO_MIN, CORRECTION_LENGTH_RATIO_MAX
from ..detection import find_best_bbox_from_kb
from ..prompts import TRIPLE_CORRECT_PROMPT, BATCH_CORRECT_PROMPT
from ..types import CorrectionError, Confidence
from ._utils import core_noun, has_garble, extract_correct_rel_from_reason
from ._vqa import (
    check_entity_exists_vqa,
    query_correct_spatial_relation,
    query_correct_action_relation,
)

_SEGMENTER = pysbd.Segmenter(language="en", clean=False)


# ── Single triple correction ────────────────────────────────────────────


def apply_triple_correction(
    caption: str,
    wrong_phrase: str,
    correct_phrase: str,
    subj: str = "",
    obj_: str = "",
) -> str:
    """Fix exactly one relationship word/phrase in the caption.

    Tries direct string replacement first (with proximity disambiguation
    for multiple occurrences), then falls back to LLM-based correction.

    Args:
        caption: Current caption text.
        wrong_phrase: The incorrect phrase to replace.
        correct_phrase: The replacement phrase.
        subj: Subject entity (for proximity ranking).
        obj_: Object entity (for proximity ranking).

    Returns:
        Corrected caption string.
    """
    cap_lower = caption.lower()
    wp_lower = wrong_phrase.lower()

    if wp_lower in cap_lower:
        occurrences: list[int] = []
        start = 0
        while True:
            idx = cap_lower.find(wp_lower, start)
            if idx == -1:
                break
            occurrences.append(idx)
            start = idx + 1

        if len(occurrences) == 1:
            idx = occurrences[0]
        else:
            subj_idx = cap_lower.find(core_noun(subj)) if subj else -1
            obj_idx = cap_lower.find(core_noun(obj_)) if obj_ else -1

            def _proximity(i: int) -> int:
                d = 0
                if subj_idx >= 0:
                    d += abs(i - subj_idx)
                if obj_idx >= 0:
                    d += abs(i - obj_idx)
                return d

            idx = min(occurrences, key=_proximity)

        return caption[:idx] + correct_phrase + caption[idx + len(wrong_phrase):]

    # LLM fallback
    raw = llm_call(
        [{"role": "user", "content": TRIPLE_CORRECT_PROMPT.format(
            caption=caption,
            subj=subj or wrong_phrase,
            obj=obj_ or correct_phrase,
            wrong_phrase=wrong_phrase,
            correct_phrase=correct_phrase,
        )}],
        max_tokens=int(len(caption.split()) * 2.5),
    )

    if raw:
        raw = raw.strip().strip('"').strip("'")
        ratio = len(raw) / max(len(caption), 1)
        if 0.85 <= ratio <= 1.25:
            return raw

    return caption


# ── Batch correction ────────────────────────────────────────────────────


def apply_batch_correction(
    img_id: str,
    caption: str,
    errors: list[CorrectionError],
    kb: dict,
    pil_image: Image.Image,
) -> tuple[str, list[dict]]:
    """Build correction guidance and apply batch LLM correction.

    Args:
        img_id: Image identifier (for logging).
        caption: Original caption.
        errors: List of confirmed errors.
        kb: Visual KB dict.
        pil_image: Full PIL image.

    Returns:
        Tuple of (corrected_caption, applied_methods).
    """
    corrected = caption
    applied: list[dict] = []

    error_lines: list[str] = []
    for i, err in enumerate(errors, 1):
        subj = err.triple.subject
        rel = err.triple.relation
        obj_ = err.triple.object
        reason = err.reason
        err_type = err.triple.rel_type.value

        guidance = build_correction_guidance(
            subj, rel, obj_, reason, err_type, err.confidence,
            caption, kb, pil_image,
        )
        err.guidance = guidance
        error_lines.append(f'{i}. "{subj} {rel} {obj_}" — {guidance}')

    error_list_str = "\n".join(error_lines)
    prompt = BATCH_CORRECT_PROMPT.format(caption=caption, error_list=error_list_str)

    raw = llm_call(
        [{"role": "user", "content": prompt}],
        max_tokens=int(len(caption.split()) * 2.5 + 50),
    )

    if raw:
        candidate = raw.strip().strip('"').strip("'")
        orig_len = len(caption)
        cand_len = len(candidate)
        ratio = cand_len / max(orig_len, 1)

        is_too_short = len(candidate.split()) < 5
        is_long_caption = len(caption.split()) >= 30
        too_compressed = is_long_caption and ratio < CORRECTION_LENGTH_RATIO_MIN
        garble = has_garble(candidate)

        if (ratio <= CORRECTION_LENGTH_RATIO_MAX
                and candidate != caption
                and not garble
                and not is_too_short
                and not too_compressed):
            corrected = candidate
            applied = [{"method": "batch_llm", "n_errors": len(errors),
                        "errors": [e.triple.claim for e in errors]}]
            log.debug("[v2] batch correction: %d errors fixed (len %d→%d, ratio=%.2f)",
                      len(errors), orig_len, cand_len, ratio)
        else:
            log.debug("[v2] batch correction REJECTED (ratio=%.2f, compressed=%s, garble=%s)",
                      ratio, too_compressed, garble)
            high_errors = [e for e in errors if e.confidence in (Confidence.HIGH, Confidence.MEDIUM)]
            if high_errors:
                corrected, fallback_applied = _fallback_deletion(corrected, high_errors)
                if fallback_applied:
                    applied.append({"method": "fallback_deletion", "n_errors": len(high_errors)})

    return corrected, applied


def _fallback_deletion(
    caption: str,
    errors: list[CorrectionError],
) -> tuple[str, bool]:
    """Fallback: delete sentences containing confirmed errors.

    Args:
        caption: Current caption text.
        errors: High-confidence errors to delete.

    Returns:
        Tuple of (corrected_caption, was_applied).
    """
    delete_claims = [e.triple.claim for e in errors]
    delete_prompt = (
        f'Caption: "{caption}"\n\n'
        f"These claims are DEFINITELY WRONG and must be COMPLETELY DELETED:\n"
    )
    for claim in delete_claims:
        delete_prompt += f"  - {claim}\n"
    delete_prompt += (
        f"\nInstructions:\n"
        f"1. Find each false claim in the caption\n"
        f"2. COMPLETELY DELETE the entire sentence containing it\n"
        f"3. Keep all other sentences unchanged\n"
        f"4. Output ONLY the corrected caption with no explanation"
    )

    result = llm_call(
        [{"role": "user", "content": delete_prompt}],
        max_tokens=int(len(caption.split()) * 2),
    )

    if result:
        result = result.strip().strip('"').strip("'")
        if result and result != caption and len(result.split()) >= 3:
            return result, True

    return caption, False


# ── Correction guidance builder ─────────────────────────────────────────


def build_correction_guidance(
    subj: str,
    rel: str,
    obj_: str,
    reason: str,
    err_type: str,
    confidence: Confidence,
    caption: str,
    kb: dict,
    pil_image: Image.Image,
) -> str:
    """Build correction guidance string for one error.

    Determines whether to use word replacement, sentence deletion,
    or softening based on error type and available evidence.

    Args:
        subj: Subject entity.
        rel: Relation.
        obj_: Object entity.
        reason: Why the error was flagged.
        err_type: "SPATIAL", "ACTION", or "ATTRIBUTE".
        confidence: Confidence of the error detection.
        caption: Original caption text.
        kb: Visual KB dict.
        pil_image: Full PIL image.

    Returns:
        Guidance instruction string for the LLM corrector.
    """
    claim_str = f"{subj} {rel} {obj_}"

    if err_type == "SPATIAL" and "absence" in reason:
        m = re.search(r"object '([^']+)' absent", reason)
        absent = m.group(1) if m else subj
        return (
            f"'{absent}' does NOT exist in this image. "
            f"Find the sentence that expresses '{claim_str}' and COMPLETELY "
            f"DELETE that one sentence. Do not touch any other sentence. "
            f"Do NOT rephrase or keep any version of the deleted sentence."
        )

    if err_type == "SPATIAL":
        correct_rel = extract_correct_rel_from_reason(reason)
        if not correct_rel and pil_image is not None:
            correct_rel = query_correct_spatial_relation(subj, obj_, kb, pil_image)
        if correct_rel and correct_rel.strip() != rel.strip():
            return (
                f"The spatial relation '{rel}' in '{claim_str}' is WRONG "
                f"(deterministic bbox geometry). "
                f"Replace ONLY the word/phrase '{rel}' with '{correct_rel}'. "
                f"Keep '{subj}' and '{obj_}' and all other text unchanged."
            )
        return (
            f"The spatial claim '{claim_str}' is definitively WRONG "
            f"(bbox geometry contradicts it) and the correct relation is "
            f"unclear. COMPLETELY DELETE the sentence containing "
            f"'{claim_str}'. Do not touch any other sentence."
        )

    # ACTION/ATTRIBUTE
    if confidence in (Confidence.HIGH, Confidence.MEDIUM):
        obj_box_exists = find_best_bbox_from_kb(obj_, kb) is not None
        if not obj_box_exists and pil_image is not None:
            obj_exists_vqa = check_entity_exists_vqa(obj_, pil_image)
        else:
            obj_exists_vqa = True if obj_box_exists else None
        obj_absent = not obj_box_exists and obj_exists_vqa is False

        if obj_absent:
            return (
                f"'{obj_}' does NOT exist in this image. "
                f"Find the sentence that expresses '{claim_str}' and COMPLETELY "
                f"DELETE that one sentence. Do not touch any other sentence. "
                f"Do NOT rephrase or keep any version of the deleted sentence."
            )

        correct_rel = query_correct_action_relation(subj, rel, obj_, kb, pil_image)
        if correct_rel:
            return (
                f"The relation '{rel}' in '{claim_str}' is DEFINITELY "
                f"WRONG (VQA HIGH confidence). "
                f"Replace ONLY the word/phrase '{rel}' with '{correct_rel}'. "
                f"Keep '{subj}' and '{obj_}' and all other text unchanged. "
                f"Do NOT add or remove any other words."
            )

        # Check standalone-ness
        sentences = _SEGMENTER.segment(caption)
        cn_s = core_noun(subj)
        cn_o = core_noun(obj_)
        s_in_sentences = sum(1 for s in sentences if cn_s in s.lower())
        o_in_sentences = sum(1 for s in sentences if cn_o in s.lower())
        is_standalone = s_in_sentences <= 1 or o_in_sentences <= 1

        if is_standalone:
            return (
                f"The claim '{claim_str}' is DEFINITELY WRONG. "
                f"COMPLETELY DELETE the sentence containing "
                f"'{claim_str}'. Do not touch any other sentence."
            )
        return (
            f"The relation '{rel}' in '{claim_str}' is DEFINITELY "
            f"WRONG (VQA HIGH confidence) but the correct relation "
            f"is unclear. Replace ONLY '{rel}' with 'near' to remove "
            f"the false claim while keeping '{subj}' and '{obj_}'."
        )

    return (
        f"The '{rel}' relationship between '{subj}' and '{obj_}' appears "
        f"incorrect (VQA confidence MEDIUM — {reason}). "
        f"Soften or correct only the '{rel}' word — keep both '{subj}' "
        f"and '{obj_}' in the sentence."
    )
