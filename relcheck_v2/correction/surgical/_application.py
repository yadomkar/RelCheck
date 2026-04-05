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

from ..._logging import log
from ...api import llm_call
from ...config import CORRECTION_LENGTH_RATIO_MIN, CORRECTION_LENGTH_RATIO_MAX, MAX_CORRECTIONS_PER_BATCH
from ...detection import find_best_bbox_from_kb
from ...prompts import TRIPLE_CORRECT_PROMPT, BATCH_CORRECT_PROMPT, CLEANUP_PROMPT
from ...types import CorrectionError, Confidence
from .._metrics import (
    GUIDANCE_DELETE_SENTENCE, GUIDANCE_REPLACE_WORD, GUIDANCE_SOFTEN,
    SOURCE_ACTION_3STAGE, SOURCE_KB_VISUAL_DESC,
    SOURCE_SPATIAL_KB, SOURCE_VLM_QUERY,
    STAGE_BATCH_CANDIDATE, STAGE_FALLBACK_DELETION,
    MetricsCollector,
)
from .._utils import core_noun, has_garble, extract_correct_rel_from_reason
from .._vqa import (
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

    # Use word-boundary regex to avoid matching inside other words
    # e.g. replacing "on" should NOT match "on" inside "orientations"
    boundary_pattern = re.compile(
        r'(?<!\w)' + re.escape(wp_lower) + r'(?!\w)',
        re.IGNORECASE,
    )
    matches = list(boundary_pattern.finditer(caption))

    if matches:
        if len(matches) == 1:
            m = matches[0]
        else:
            # Multiple matches — pick the one closest to subject/object
            subj_idx = cap_lower.find(core_noun(subj).lower()) if subj else -1
            obj_idx = cap_lower.find(core_noun(obj_).lower()) if obj_ else -1

            def _proximity(m: re.Match) -> int:
                d = 0
                if subj_idx >= 0:
                    d += abs(m.start() - subj_idx)
                if obj_idx >= 0:
                    d += abs(m.start() - obj_idx)
                return d

            m = min(matches, key=_proximity)

        return caption[:m.start()] + correct_phrase + caption[m.end():]

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


# ── Sequential single-edit correction ────────────────────────────────────


def apply_sequential_correction(
    img_id: str,
    caption: str,
    errors: list[CorrectionError],
    kb: dict,
    pil_image: Image.Image,
    metrics: MetricsCollector | None = None,
) -> tuple[str, list[dict]]:
    """Apply corrections one at a time using single-edit calls.

    Each error gets its own LLM call with TRIPLE_CORRECT_PROMPT or
    direct string replacement. This avoids the garble problem from
    batch correction where multiple simultaneous edits confuse the LLM.

    For DELETE_SENTENCE guidance, uses local sentence segmentation
    instead of an LLM call.

    Args:
        img_id: Image identifier (for logging).
        caption: Original caption.
        errors: List of confirmed errors.
        kb: Visual KB dict.
        pil_image: Full PIL image.
        metrics: Optional metrics collector for path logging.

    Returns:
        Tuple of (corrected_caption, applied_methods).
    """
    corrected = caption
    applied: list[dict] = []

    for err in errors:
        subj = err.triple.subject
        rel = err.triple.relation
        obj_ = err.triple.object
        reason = err.reason
        err_type = err.triple.rel_type.value

        guidance = build_correction_guidance(
            subj, rel, obj_, reason, err_type, err.confidence,
            corrected, kb, pil_image,
            img_id=img_id, metrics=metrics,
        )
        err.guidance = guidance

        if "does NOT exist" in guidance:
            # Only delete for truly non-existent entities — not for wrong relations
            sentences = _SEGMENTER.segment(corrected)
            cn_s = core_noun(subj).lower()
            cn_o = core_noun(obj_).lower()
            kept = []
            deleted = False
            for sent in sentences:
                sl = sent.lower()
                if not deleted and cn_s in sl and cn_o in sl:
                    deleted = True
                    continue
                kept.append(sent)
            if deleted and kept and len(" ".join(kept).split()) >= 10:
                corrected = " ".join(kept).strip()
                applied.append({"method": "sentence_deletion", "claim": err.triple.claim})
        elif "COMPLETELY DELETE" in guidance:
            # Wrong relation but no correct one found — replace with "near" instead of deleting
            corrected = apply_triple_correction(
                corrected, rel, "near", subj=subj, obj_=obj_,
            )
            applied.append({"method": "single_edit_fallback", "claim": err.triple.claim,
                            "old": rel, "new": "near"})
        else:
            # Word replacement — use apply_triple_correction (single edit)
            correct_rel = extract_correct_rel_from_reason(reason)
            if not correct_rel:
                # Try to extract from guidance text
                m = re.search(r"rewrite it as '([^']+)'", guidance)
                if m:
                    full_rewrite = m.group(1)
                    # Extract just the relation part
                    if subj.lower() in full_rewrite.lower() and obj_.lower() in full_rewrite.lower():
                        start = full_rewrite.lower().find(core_noun(subj).lower())
                        end = full_rewrite.lower().find(core_noun(obj_).lower())
                        if start >= 0 and end > start:
                            correct_rel = full_rewrite[start + len(core_noun(subj)):end].strip()

            if correct_rel and correct_rel.strip() != rel.strip():
                corrected = apply_triple_correction(
                    corrected, rel, correct_rel, subj=subj, obj_=obj_,
                )
                applied.append({"method": "single_edit", "claim": err.triple.claim,
                                "old": rel, "new": correct_rel})
            else:
                # No correct relation found — try generic "near" replacement
                corrected = apply_triple_correction(
                    corrected, rel, "near", subj=subj, obj_=obj_,
                )
                applied.append({"method": "single_edit_fallback", "claim": err.triple.claim,
                                "old": rel, "new": "near"})

    # ── Final grammar cleanup pass ──
    # Catch any garbles introduced by string replacement or LLM edits
    if corrected != caption and has_garble(corrected):
        log.debug("[%s] Garble detected after sequential correction — running cleanup", img_id)
        cleanup_result = llm_call(
            [{"role": "user", "content": CLEANUP_PROMPT.format(caption=corrected)}],
            max_tokens=int(len(corrected.split()) * 1.5 + 20),
        )
        if cleanup_result:
            cleaned = cleanup_result.strip().strip('"').strip("'")
            # Accept cleanup only if it doesn't drastically change length
            ratio = len(cleaned) / max(len(corrected), 1)
            if 0.80 <= ratio <= 1.20 and len(cleaned.split()) >= 10:
                corrected = cleaned
                applied.append({"method": "grammar_cleanup"})

    # Record metrics
    if metrics is not None:
        # Record batch eval as "sequential" mode
        orig_len = len(caption)
        corr_len = len(corrected)
        ratio = corr_len / max(orig_len, 1)
        metrics.record_batch_eval(
            img_id,
            length_ratio=round(ratio, 4),
            garble_detected=False,
            too_short=len(corrected.split()) < 5,
            too_compressed=False,
            accepted=True,
        )
        metrics.record_caption_snapshot(
            img_id, STAGE_BATCH_CANDIDATE, corrected, accepted=True,
        )
        metrics.record_fallback_deletion(img_id, used=False, n_deleted=0)

    return corrected, applied


# ── Batch correction (kept for ablation) ────────────────────────────────


def apply_batch_correction(
    img_id: str,
    caption: str,
    errors: list[CorrectionError],
    kb: dict,
    pil_image: Image.Image,
    metrics: MetricsCollector | None = None,
) -> tuple[str, list[dict]]:
    """Build correction guidance and apply batch LLM correction.

    Args:
        img_id: Image identifier (for logging).
        caption: Original caption.
        errors: List of confirmed errors.
        kb: Visual KB dict.
        pil_image: Full PIL image.
        metrics: Optional metrics collector for path logging.

    Returns:
        Tuple of (corrected_caption, applied_methods).
    """
    corrected = caption
    applied: list[dict] = []

    # Cap corrections to avoid garbled output from too many simultaneous edits.
    # Prioritize HIGH confidence, then MEDIUM. Overflow goes to fallback deletion.
    overflow_errors: list[CorrectionError] = []
    if len(errors) > MAX_CORRECTIONS_PER_BATCH:
        log.debug("[v2] %d errors exceed cap of %d — splitting", len(errors), MAX_CORRECTIONS_PER_BATCH)
        # Sort: HIGH first, then MEDIUM, then LOW
        priority = {Confidence.HIGH: 0, Confidence.MEDIUM: 1, Confidence.LOW: 2}
        sorted_errors = sorted(errors, key=lambda e: priority.get(e.confidence, 2))
        batch_errors = sorted_errors[:MAX_CORRECTIONS_PER_BATCH]
        overflow_errors = sorted_errors[MAX_CORRECTIONS_PER_BATCH:]
        errors = batch_errors
        log.info("[%s] Correction cap: %d batch + %d overflow", img_id, len(errors), len(overflow_errors))

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
            img_id=img_id, metrics=metrics,
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

        accepted = (
            ratio <= CORRECTION_LENGTH_RATIO_MAX
            and candidate != caption
            and not garble
            and not is_too_short
            and not too_compressed
        )

        if metrics is not None:
            metrics.record_batch_eval(
                img_id,
                length_ratio=round(ratio, 4),
                garble_detected=bool(garble),
                too_short=is_too_short,
                too_compressed=too_compressed,
                accepted=accepted,
            )
            metrics.record_caption_snapshot(
                img_id, STAGE_BATCH_CANDIDATE, candidate, accepted=accepted,
            )

        if accepted:
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
                    if metrics is not None:
                        n_deleted = len(high_errors)
                        metrics.record_fallback_deletion(img_id, used=True, n_deleted=n_deleted)
                        metrics.record_caption_snapshot(
                            img_id, STAGE_FALLBACK_DELETION, corrected,
                        )
                else:
                    if metrics is not None:
                        metrics.record_fallback_deletion(img_id, used=False, n_deleted=0)
            else:
                if metrics is not None:
                    metrics.record_fallback_deletion(img_id, used=False, n_deleted=0)

    # Handle overflow errors (beyond the cap) via fallback deletion
    if overflow_errors:
        corrected, overflow_applied = _fallback_deletion(corrected, overflow_errors)
        if overflow_applied:
            applied.append({"method": "overflow_deletion", "n_errors": len(overflow_errors)})
            log.debug("[v2] overflow deletion: %d errors handled via sentence removal", len(overflow_errors))

    return corrected, applied


def _fallback_deletion(
    caption: str,
    errors: list[CorrectionError],
) -> tuple[str, bool]:
    """Fallback: delete only the specific sentence containing each error.

    Uses sentence segmentation to surgically remove just the sentences
    that contain hallucinated claims, preserving all other content.
    Caps deletion at 50% of total sentences to prevent gutting the caption.

    Args:
        caption: Current caption text.
        errors: High-confidence errors to delete.

    Returns:
        Tuple of (corrected_caption, was_applied).
    """
    sentences = _SEGMENTER.segment(caption)
    if not sentences:
        return caption, False

    delete_claims = {e.triple.claim.lower() for e in errors}
    # Also collect individual subject/object nouns for fuzzy matching
    delete_nouns = set()
    for e in errors:
        delete_nouns.add(core_noun(e.triple.subject))
        delete_nouns.add(core_noun(e.triple.object))

    to_delete: set[int] = set()
    for i, sent in enumerate(sentences):
        sent_lower = sent.lower()
        for claim in delete_claims:
            # Check if claim words appear in this sentence
            claim_words = claim.split()
            if len(claim_words) >= 2:
                subj_word = claim_words[0]
                obj_word = claim_words[-1]
                if subj_word in sent_lower and obj_word in sent_lower:
                    to_delete.add(i)
                    break

    # Cap: never delete more than half the sentences
    max_deletions = max(len(sentences) // 2, 1)
    if len(to_delete) > max_deletions:
        # Keep only the first max_deletions (by sentence order)
        to_delete = set(sorted(to_delete)[:max_deletions])

    if not to_delete:
        return caption, False

    kept = [s for i, s in enumerate(sentences) if i not in to_delete]
    result = " ".join(kept).strip()

    if result and len(result.split()) >= 10:
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
    img_id: str = "",
    metrics: MetricsCollector | None = None,
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
        img_id: Image identifier (for metrics recording).
        metrics: Optional metrics collector for path logging.

    Returns:
        Guidance instruction string for the LLM corrector.
    """
    claim_str = f"{subj} {rel} {obj_}"

    # Track guidance metadata for metrics
    guidance_type: str = GUIDANCE_SOFTEN
    correct_rel_found = False
    correct_rel_source: str | None = None

    if err_type == "SPATIAL" and "absence" in reason:
        m = re.search(r"object '([^']+)' absent", reason)
        absent = m.group(1) if m else subj
        guidance_type = GUIDANCE_DELETE_SENTENCE
        guidance_text = (
            f"'{absent}' does NOT exist in this image. "
            f"Find the sentence that expresses '{claim_str}' and COMPLETELY "
            f"DELETE that one sentence. Do not touch any other sentence. "
            f"Do NOT rephrase or keep any version of the deleted sentence."
        )
    elif err_type == "SPATIAL":
        correct_rel = extract_correct_rel_from_reason(reason)
        if correct_rel:
            correct_rel_source = SOURCE_SPATIAL_KB
        if not correct_rel and pil_image is not None:
            correct_rel = query_correct_spatial_relation(subj, obj_, kb, pil_image)
            if correct_rel:
                correct_rel_source = SOURCE_VLM_QUERY
        if correct_rel and correct_rel.strip() != rel.strip():
            guidance_type = GUIDANCE_REPLACE_WORD
            correct_rel_found = True
            guidance_text = (
                f"The spatial relation '{rel}' in '{claim_str}' is WRONG "
                f"(deterministic bbox geometry). "
                f"Find the phrase '{subj} {rel} {obj_}' and rewrite it as "
                f"'{subj} {correct_rel} {obj_}'. "
                f"Keep the rest of the sentence unchanged. "
                f"Make sure the result is grammatically correct English."
            )
        else:
            guidance_type = GUIDANCE_DELETE_SENTENCE
            guidance_text = (
                f"The spatial claim '{claim_str}' is definitively WRONG "
                f"(bbox geometry contradicts it) and the correct relation is "
                f"unclear. COMPLETELY DELETE the sentence containing "
                f"'{claim_str}'. Do not touch any other sentence."
            )
    elif confidence in (Confidence.HIGH, Confidence.MEDIUM):
        # ACTION/ATTRIBUTE with high/medium confidence
        obj_box_exists = find_best_bbox_from_kb(obj_, kb) is not None
        if not obj_box_exists and pil_image is not None:
            obj_exists_vqa = check_entity_exists_vqa(obj_, pil_image)
        else:
            obj_exists_vqa = True if obj_box_exists else None
        obj_absent = not obj_box_exists and obj_exists_vqa is False

        if obj_absent:
            guidance_type = GUIDANCE_DELETE_SENTENCE
            guidance_text = (
                f"'{obj_}' does NOT exist in this image. "
                f"Find the sentence that expresses '{claim_str}' and COMPLETELY "
                f"DELETE that one sentence. Do not touch any other sentence. "
                f"Do NOT rephrase or keep any version of the deleted sentence."
            )
        else:
            correct_rel = query_correct_action_relation(subj, rel, obj_, kb, pil_image)
            if correct_rel:
                guidance_type = GUIDANCE_REPLACE_WORD
                correct_rel_found = True
                correct_rel_source = SOURCE_ACTION_3STAGE
                guidance_text = (
                    f"The relation '{rel}' in '{claim_str}' is DEFINITELY "
                    f"WRONG (VQA HIGH confidence). "
                    f"Find the phrase '{subj} {rel} {obj_}' and rewrite it as "
                    f"'{subj} {correct_rel} {obj_}'. "
                    f"Keep the rest of the sentence unchanged. "
                    f"Make sure the result is grammatically correct English."
                )
            else:
                # Check standalone-ness
                sentences = _SEGMENTER.segment(caption)
                cn_s = core_noun(subj)
                cn_o = core_noun(obj_)
                s_in_sentences = sum(1 for s in sentences if cn_s in s.lower())
                o_in_sentences = sum(1 for s in sentences if cn_o in s.lower())
                is_standalone = s_in_sentences <= 1 or o_in_sentences <= 1

                if is_standalone:
                    guidance_type = GUIDANCE_DELETE_SENTENCE
                    guidance_text = (
                        f"The claim '{claim_str}' is DEFINITELY WRONG. "
                        f"COMPLETELY DELETE the sentence containing "
                        f"'{claim_str}'. Do not touch any other sentence."
                    )
                else:
                    guidance_type = GUIDANCE_REPLACE_WORD
                    correct_rel_found = True
                    correct_rel_source = None  # fallback to "near"
                    guidance_text = (
                        f"The relation '{rel}' in '{claim_str}' is DEFINITELY "
                        f"WRONG (VQA HIGH confidence) but the correct relation "
                        f"is unclear. Find the phrase '{subj} {rel} {obj_}' and "
                        f"rewrite it as '{subj} near {obj_}'. "
                        f"Make sure the result is grammatically correct English."
                    )
    else:
        guidance_type = GUIDANCE_SOFTEN
        guidance_text = (
            f"The '{rel}' relationship between '{subj}' and '{obj_}' appears "
            f"incorrect (VQA confidence MEDIUM — {reason}). "
            f"Soften or correct only the '{rel}' word — keep both '{subj}' "
            f"and '{obj_}' in the sentence."
        )

    if metrics is not None:
        metrics.record_guidance(img_id, {
            "claim": claim_str,
            "guidance_type": guidance_type,
            "correct_rel_found": correct_rel_found,
            "correct_rel_source": correct_rel_source,
        })

    return guidance_text
