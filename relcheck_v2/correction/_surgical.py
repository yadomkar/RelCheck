"""
RelCheck v2 — Long Caption Surgical Correction
================================================
Per-triple verification and minimal editing for captions >= 30 words
(e.g. LLaVA, Qwen). Key properties:

- Never shortens captions (surgical edit only)
- Type-aware verification: spatial → geometry, action → VQA
- Cross-captioner consensus pre-filter (free signal)
- Post-verification: checks corrected caption for new hallucinations
"""

from __future__ import annotations

import json
import re
from collections import Counter

import json_repair
import pysbd
from nltk.util import ngrams
from PIL import Image

# Module-level segmenter (stateless, safe to reuse)
_SEGMENTER = pysbd.Segmenter(language="en", clean=False)

from .._logging import log
from ..api import llm_call
from ..config import (
    ENTITY_SYNONYMS, SPATIAL_OPPOSITES,
    CORRECTION_LENGTH_RATIO_MIN, CORRECTION_LENGTH_RATIO_MAX,
)
from ..detection import find_best_bbox_from_kb
from ..entity import levenshtein_distance
from ..prompts import (
    TRIPLE_EXTRACT_PROMPT, TRIPLE_CORRECT_PROMPT,
    BATCH_CORRECT_PROMPT, MISSING_FACTS_PROMPT,
)
from ..spatial import SPATIAL_TRIPLE_RE
from ..types import (
    CorrectionMode, CorrectionResult, CorrectionError,
    VerificationResult, Triple, RelationType, Verdict, Confidence,
)
from ._geometry import (
    ACTION_GEOMETRY_TAXONOMY, classify_action_family,
    get_person_keypoints, check_action_geometry,
)
from ._utils import (
    core_noun, entity_matches, normalize_entity,
    has_garble, extract_correct_rel_from_reason,
)
from ._vqa import (
    check_entity_exists_vqa, verify_action_triple,
    query_correct_spatial_relation, query_correct_action_relation,
    _parse_spatial_facts,
)


# ── Spatial synonym groups ───────────────────────────────────────────────

_SPATIAL_SYNONYM_GROUPS: list[set[str]] = [
    {"on", "above", "on top of", "over"},
    {"under", "below", "beneath", "underneath"},
    {"left", "to the left", "to the left of"},
    {"right", "to the right", "to the right of"},
    {"in front of", "before"},
    {"behind", "in back of"},
]


def _spatial_synonyms(rel: str) -> set[str]:
    """Return the synonym set that contains rel, or {rel} if none."""
    rel_lower = rel.lower().strip()
    for group in _SPATIAL_SYNONYM_GROUPS:
        if rel_lower in group:
            return group
    return {rel_lower}


# ── Triple extraction ────────────────────────────────────────────────────

_SPATIAL_WORDS: frozenset[str] = frozenset({
    "left", "right", "above", "below", "behind", "front",
    "beside", "next to", "on top of", "under", "over",
    "in front of", "near", "inside", "outside", "between",
})
_ACTION_WORDS: frozenset[str] = frozenset({
    "riding", "holding", "carrying", "eating", "drinking",
    "wearing", "pushing", "pulling", "walking", "running",
    "sitting", "standing", "playing", "using", "throwing",
    "catching", "carrying", "holding", "driving", "leading",
})


def _extract_triples(caption: str) -> list[Triple]:
    """Use LLM to extract relational triples from a caption.

    Falls back to keyword-based type classification if the LLM
    does not provide a valid type.

    Args:
        caption: Caption text.

    Returns:
        List of Triple objects.
    """
    prompt = TRIPLE_EXTRACT_PROMPT.format(caption=caption)
    raw = llm_call([{"role": "user", "content": prompt}], max_tokens=600)
    if not raw:
        log.debug("[extract_triples] llm_call returned empty for caption: %.80r", caption)
        return []

    log.debug("[extract_triples] raw=%.200r", raw[:200])

    try:
        parsed = json_repair.loads(raw)
    except Exception as e:
        log.debug("[extract_triples] json_repair failed: %s | raw=%.300r", e, raw[:300])
        return []

    # Unwrap dict wrapper
    if isinstance(parsed, dict):
        for key in ("triples", "relations", "result", "data", "output"):
            if key in parsed and isinstance(parsed[key], list):
                parsed = parsed[key]
                break
        else:
            log.debug("[extract_triples] got dict but no list key: %s", list(parsed.keys()))
            return []

    if not isinstance(parsed, list):
        log.debug("[extract_triples] unexpected type: %s", type(parsed))
        return []

    result: list[Triple] = []
    for t in parsed:
        if not isinstance(t, dict):
            continue
        if not all(k in t for k in ("subject", "relation", "object")):
            continue

        typ = str(t.get("type", "")).upper().strip()
        if typ not in ("SPATIAL", "ACTION", "ATTRIBUTE"):
            rel_lower = t.get("relation", "").lower()
            if any(w in rel_lower for w in _SPATIAL_WORDS):
                typ = "SPATIAL"
            elif any(w in rel_lower for w in _ACTION_WORDS):
                typ = "ACTION"
            else:
                typ = "ACTION"

        result.append(Triple(
            subject=t["subject"].strip(),
            relation=t["relation"].strip(),
            object=t["object"].strip(),
            rel_type=RelationType(typ),
        ))
    return result


# ── Cross-captioner consensus ────────────────────────────────────────────


def _consensus_confirms_triple(
    subj: str,
    rel: str,
    obj: str,
    cross_captions: dict[str, str],
) -> bool:
    """Check if all cross-captioners mention (subj rel obj).

    Args:
        subj: Subject entity.
        rel: Relation.
        obj: Object entity.
        cross_captions: {captioner_name: caption_text} dict.

    Returns:
        True if all captioners confirm the triple.
    """
    if not cross_captions or len(cross_captions) < 2:
        return False

    core_s = core_noun(subj)
    core_o = core_noun(obj)
    rel_lower = rel.lower()

    confirmed = 0
    for cap_text in cross_captions.values():
        if not cap_text:
            continue
        cap_lower = cap_text.lower()
        if rel_lower not in cap_lower:
            continue
        idx = cap_lower.find(rel_lower)
        while idx >= 0:
            context = cap_lower[max(0, idx - 80) : idx + 80]
            subj_near = (
                core_s in context
                or any(s in context for s in ENTITY_SYNONYMS.get(core_s, []))
            )
            obj_near = (
                core_o in context
                or any(s in context for s in ENTITY_SYNONYMS.get(core_o, []))
            )
            if subj_near and obj_near:
                confirmed += 1
                break
            idx = cap_lower.find(rel_lower, idx + 1)

    return confirmed >= len(cross_captions)


# ── Caption text helpers ─────────────────────────────────────────────────


def _caption_name_for(kb_entity: str, cap_lower: str) -> str:
    """Return the synonym of kb_entity that appears in cap_lower."""
    cn = core_noun(kb_entity)
    if not cn:
        return kb_entity
    if cn in cap_lower:
        return cn
    for syn in ENTITY_SYNONYMS.get(cn, []):
        if syn in cap_lower:
            return syn
    return cn


def _relation_already_expressed(subj: str, rel: str, obj: str, cap_lower: str) -> bool:
    """Check if caption already states subj <rel> obj or equivalent."""
    rel_norm = rel.lower()
    rel_variants: dict[str, list[str]] = {
        "left": ["left"], "right": ["right"],
        "above": ["above", "on top of"], "below": ["below", "beneath", "under"],
        "behind": ["behind", "in back of"], "in front of": ["in front of", "front"],
    }
    keywords = rel_variants.get(rel_norm, [rel_norm])
    for kw in keywords:
        if kw not in cap_lower:
            continue
        idx = cap_lower.find(kw)
        context = cap_lower[max(0, idx - 80) : idx + 80]
        cn_s = core_noun(subj)
        cn_o = core_noun(obj)
        if cn_s in context or cn_o in context:
            return True
        for syn in ENTITY_SYNONYMS.get(cn_s, []):
            if syn in context:
                return True
        for syn in ENTITY_SYNONYMS.get(cn_o, []):
            if syn in context:
                return True
    return False


# ── Spatial addendum ─────────────────────────────────────────────────────


def _build_spatial_addendum(
    corrected_caption: str,
    kb: dict,
    max_facts: int = 5,
) -> tuple[str, int]:
    """Append missing KB spatial facts to caption.

    Args:
        corrected_caption: Current caption text.
        kb: Visual KB dict.
        max_facts: Maximum number of spatial facts to add.

    Returns:
        Tuple of (new_caption, n_facts_added).
    """
    spatial_facts = kb.get("spatial_facts", [])
    if not spatial_facts:
        return corrected_caption, 0

    cap_lower = corrected_caption.lower()
    missing: list[tuple[str, str, str, str, str, str]] = []

    for fact in spatial_facts:
        triples = _parse_spatial_facts([fact])
        if not triples:
            continue
        subj, rel, obj = triples[0]
        if not subj or not obj:
            continue
        if _relation_already_expressed(subj, rel, obj, cap_lower):
            continue
        cap_subj = _caption_name_for(subj, cap_lower)
        cap_obj = _caption_name_for(obj, cap_lower)
        missing.append((subj, rel, obj, fact, cap_subj, cap_obj))
        if len(missing) >= max_facts:
            break

    if not missing:
        return corrected_caption, 0

    fact_phrases = [f"the {cs} is {r} the {co}" for _, r, _, _, cs, co in missing]
    if len(fact_phrases) == 1:
        addendum = f"Spatially, {fact_phrases[0]}."
    elif len(fact_phrases) == 2:
        addendum = f"Spatially, {fact_phrases[0]}, and {fact_phrases[1]}."
    else:
        joined = ", ".join(fact_phrases[:-1]) + f", and {fact_phrases[-1]}"
        addendum = f"Spatially, {joined}."

    new_cap = corrected_caption.rstrip() + " " + addendum
    return new_cap, len(missing)


# ── Missing fact addendum ────────────────────────────────────────────────


def _add_missing_fact_addendum(
    corrected_caption: str,
    kb: dict,
) -> tuple[str, int]:
    """Insert facts from visual description that are absent from caption.

    Uses LLM to identify up to 2 missing facts and insert them
    at semantically appropriate positions.

    Args:
        corrected_caption: Current caption text.
        kb: Visual KB dict.

    Returns:
        Tuple of (new_caption, n_facts_added).
    """
    from ..config import ADDENDUM_MAX_WORDS_ADDED, ADDENDUM_SURVIVAL_RATIO

    visual_desc = kb.get("visual_description", "")
    if not visual_desc or len(visual_desc.strip()) < 20:
        return corrected_caption, 0
    if len(corrected_caption.split()) > 300:
        return corrected_caption, 0

    prompt = MISSING_FACTS_PROMPT.format(
        caption=corrected_caption[:1500],
        visual_description=visual_desc[:1500],
    )
    raw = llm_call([{"role": "user", "content": prompt}], max_tokens=800)
    if not raw:
        return corrected_caption, 0

    result = raw.strip().strip('"').strip("'").strip()
    if result.upper() == "NONE" or not result:
        return corrected_caption, 0

    orig_words = len(corrected_caption.split())
    new_words = len(result.split())
    added = new_words - orig_words

    if added <= 0 or added > ADDENDUM_MAX_WORDS_ADDED:
        return corrected_caption, 0

    # Check token survival (LLM didn't rewrite the whole thing)
    orig_tokens = corrected_caption.lower().split()
    result_lower = result.lower()
    surviving = sum(1 for t in orig_tokens if t in result_lower)
    if surviving / max(len(orig_tokens), 1) < ADDENDUM_SURVIVAL_RATIO:
        return corrected_caption, 0

    # Detect n-gram repetition (bigrams and trigrams)
    result_toks = result.lower().split()
    for n in (2, 3):
        counts = Counter(ngrams(result_toks, n))
        if any(c >= 3 for c in counts.values()):
            return corrected_caption, 0

    # Detect new bigram repetition (not present in original)
    orig_toks = corrected_caption.lower().split()
    orig_bigrams = set(ngrams(orig_toks, 2))
    result_bigram_counts = Counter(ngrams(result_toks, 2))
    for bg, count in result_bigram_counts.items():
        if bg not in orig_bigrams and count >= 2:
            return corrected_caption, 0

    return result, 1


# ── Triple correction helpers ────────────────────────────────────────────


def _apply_triple_correction(
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

        return caption[:idx] + correct_phrase + caption[idx + len(wrong_phrase) :]

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


# ── Spatial contradiction detection ──────────────────────────────────────


def _check_spatial_contradictions(
    caption: str,
    spatial_facts: list[str],
) -> list[str]:
    """Deterministically detect spatial contradictions between caption and KB.

    Args:
        caption: Caption text.
        spatial_facts: KB spatial fact strings.

    Returns:
        List of human-readable contradiction strings.
    """
    if not spatial_facts:
        return []

    kb_triples = _parse_spatial_facts(spatial_facts)
    if not kb_triples:
        return []

    # Extract spatial triples from caption
    caption_triples: list[tuple[str, str, str]] = []
    for m in SPATIAL_TRIPLE_RE.finditer(caption.lower()):
        subj = m.group(1).strip().rstrip(" ,;")
        rel = m.group(2).strip()
        obj = m.group(3).strip().rstrip(" ,;.")
        caption_triples.append((subj, rel, obj))

    if not caption_triples:
        return []

    contradictions: list[str] = []
    for cap_subj, cap_rel, cap_obj in caption_triples:
        cap_opposite = SPATIAL_OPPOSITES.get(cap_rel.lower())
        if not cap_opposite:
            continue

        for kb_subj, kb_rel, kb_obj in kb_triples:
            if not (entity_matches(cap_subj, kb_subj) and entity_matches(cap_obj, kb_obj)):
                continue
            if kb_rel == cap_opposite:
                kb_fact_str = next(
                    (f for f in spatial_facts
                     if normalize_entity(kb_subj) in f.lower()
                     and normalize_entity(kb_obj) in f.lower()),
                    f"'{kb_subj}' {kb_rel} '{kb_obj}'",
                )
                contradictions.append(
                    f"Caption says '{cap_subj} {cap_rel} {cap_obj}' "
                    f"but geometry shows: {kb_fact_str}"
                )
                break

    return contradictions


# ── Main surgical correction ─────────────────────────────────────────────

_PERSON_WORDS: frozenset[str] = frozenset({
    "person", "man", "woman", "boy", "girl",
    "child", "kid", "baby", "player", "rider",
    "worker", "people", "dog", "cat", "horse",
})


def correct_long_caption(
    img_id: str,
    caption: str,
    kb: dict,
    pil_image: Image.Image | None = None,
    cross_captions: dict[str, str] | None = None,
) -> CorrectionResult:
    """Per-triple verification and surgical correction for detailed captions.

    Pipeline:
        1. Extract relational triples via LLM
        2. For each triple: type-aware verification (geometry + VQA)
        3. Batch correction of all confirmed errors
        4. Post-verification: check for introduced hallucinations
        5. Append missing spatial facts and visual description facts

    Args:
        img_id: Image identifier (for logging).
        caption: Original caption text (>= 30 words).
        kb: Visual KB dict.
        pil_image: Full PIL image (required for VQA).
        cross_captions: Optional {captioner_name: caption_text} for consensus pre-filter.

    Returns:
        CorrectionResult with corrected caption and metadata.
    """
    if pil_image is None:
        return CorrectionResult(
            original=caption, corrected=caption,
            mode=CorrectionMode.CORRECT_V2, status="unchanged",
        )

    # ── Step 1: Extract triples ──
    triples = _extract_triples(caption)
    if not triples:
        log.debug("[%s] triple extraction returned 0 triples — addendum only", img_id)
        corrected, n_addendum = _build_spatial_addendum(caption, kb)
        edit_dist = levenshtein_distance(caption, corrected)
        return CorrectionResult(
            original=caption, corrected=corrected,
            mode=CorrectionMode.CORRECT_V2,
            edit_rate=edit_dist / max(len(caption), 1),
            n_triples=0, n_addendum=n_addendum,
            status="modified" if corrected != caption else "unchanged",
        )

    # ── Step 2: Verify each triple ──
    spatial_facts = kb.get("spatial_facts", [])
    geo_contras = _check_spatial_contradictions(caption, spatial_facts)
    geo_set = {c.lower() for c in geo_contras}

    errors: list[CorrectionError] = []
    all_checks: list[VerificationResult] = []

    for triple in triples:
        subj, rel, obj = triple.subject, triple.relation, triple.object
        claim_str = triple.claim

        if triple.rel_type == RelationType.SPATIAL:
            _verify_spatial_triple(
                triple, kb, pil_image, spatial_facts, geo_set,
                errors, all_checks, img_id,
            )
        else:
            _verify_action_attribute_triple(
                triple, kb, pil_image, cross_captions,
                errors, all_checks, img_id,
            )

    # ── Step 3: Apply corrections ──
    corrected = caption
    applied: list[dict] = []

    if errors:
        corrected, applied = _apply_batch_correction(
            img_id, caption, errors, kb, pil_image,
        )

    # ── Step 4: Post-verification ──
    if corrected != caption:
        new_triples = _extract_triples(corrected)
        introduced_errors: list[str] = []
        for nt in new_triples:
            if nt.rel_type == RelationType.SPATIAL:
                kb_triples = _parse_spatial_facts(kb.get("spatial_facts", []))
                for kb_s, kb_r, kb_o in kb_triples:
                    if entity_matches(nt.subject, kb_s) and entity_matches(nt.object, kb_o):
                        opp = SPATIAL_OPPOSITES.get(nt.relation.lower())
                        if opp and kb_r.lower() == opp:
                            introduced_errors.append(
                                f"{nt.claim} (KB says {kb_r})"
                            )
                            break

        if introduced_errors:
            log.info("[%s] POST-CHECK FAILED: correction introduced new errors: %s → reverting",
                     img_id, introduced_errors)
            corrected = caption
            applied = []

    # ── Step 5: Addendum ──
    corrected, n_addendum = _add_missing_fact_addendum(corrected, kb)
    if n_addendum:
        log.debug("[%s] addendum: +%d missing fact(s) appended", img_id, n_addendum)

    edit_dist = levenshtein_distance(caption, corrected)
    edit_rate_val = edit_dist / max(len(caption), len(corrected), 1)

    return CorrectionResult(
        original=caption,
        corrected=corrected,
        errors=errors,
        checks=all_checks,
        mode=CorrectionMode.CORRECT_V2,
        edit_rate=edit_rate_val,
        n_triples=len(triples),
        n_addendum=n_addendum,
        status="modified" if corrected != caption else "unchanged",
    )


# ── Spatial triple verification subroutine ───────────────────────────────


def _verify_spatial_triple(
    triple: Triple,
    kb: dict,
    pil_image: Image.Image,
    spatial_facts: list[str],
    geo_set: set[str],
    errors: list[CorrectionError],
    all_checks: list[VerificationResult],
    img_id: str,
) -> None:
    """Verify a SPATIAL triple and append results to errors/all_checks."""
    subj, rel, obj = triple.subject, triple.relation, triple.object
    claim_str = triple.claim

    kb_triples = _parse_spatial_facts(spatial_facts)
    verdict, confidence, reason = Verdict.UNKNOWN, Confidence.LOW, "no geometry available"

    for kb_s, kb_r, kb_o in kb_triples:
        if entity_matches(subj, kb_s) and entity_matches(obj, kb_o):
            if kb_r.lower() in _spatial_synonyms(rel):
                verdict, confidence, reason = Verdict.CORRECT, Confidence.HIGH, "geometry confirms"
                break
            opp = SPATIAL_OPPOSITES.get(rel)
            if opp and kb_r == opp:
                verdict, confidence, reason = Verdict.INCORRECT, Confidence.HIGH, f"geometry shows {kb_r}"
                break

    if any(claim_str.lower() in g or (subj.lower() in g and rel.lower() in g) for g in geo_set):
        verdict = Verdict.INCORRECT
        confidence = Confidence.HIGH
        reason = "deterministic geometry contradiction"

    # Cross-check geometry INCORRECT with VQA
    if verdict == Verdict.INCORRECT and confidence == Confidence.HIGH:
        vqa_confirm, _, _, _, _ = verify_action_triple(subj, rel, obj, kb, pil_image, n_questions=2)
        if vqa_confirm is True:
            verdict, confidence = Verdict.CORRECT, Confidence.MEDIUM
            reason = "geometry said INCORRECT but Maverick VQA confirmed → kept"
        elif vqa_confirm is None:
            verdict, confidence = Verdict.UNKNOWN, Confidence.LOW
            reason = "geometry INCORRECT but Maverick uncertain → skipped"

    # Fallback: check entity existence + VQA
    if verdict == Verdict.UNKNOWN:
        obj_box_check = find_best_bbox_from_kb(obj, kb)
        if obj_box_check is None:
            exists = check_entity_exists_vqa(obj, pil_image)
            if exists is False:
                verdict, confidence = Verdict.INCORRECT, Confidence.HIGH
                reason = f"object '{obj}' absent: not found by GDino and VQA confirms absence — remove this claim"
            else:
                vqa_result, _, _, _, _ = verify_action_triple(subj, rel, obj, kb, pil_image, n_questions=3)
                if vqa_result is False:
                    verdict, confidence = Verdict.INCORRECT, Confidence.MEDIUM
                    reason = f"spatial VQA (full-image, no GDino bbox) rejected '{rel}'"
                elif vqa_result is True:
                    verdict, confidence = Verdict.CORRECT, Confidence.MEDIUM
                    reason = "spatial VQA (full-image, no GDino bbox) confirmed"
        else:
            vqa_result, _, _, _, _ = verify_action_triple(subj, rel, obj, kb, pil_image, n_questions=3)
            if vqa_result is False:
                correct_spatial = query_correct_spatial_relation(subj, obj, kb, pil_image)
                verdict, confidence = Verdict.INCORRECT, Confidence.MEDIUM
                reason = (
                    f"spatial VQA rejected; correct relation: '{correct_spatial}'"
                    if correct_spatial else "spatial VQA fallback rejected"
                )
            elif vqa_result is True:
                verdict, confidence = Verdict.CORRECT, Confidence.MEDIUM
                reason = "spatial VQA fallback confirmed"

    all_checks.append(VerificationResult(
        triple=triple, verdict=verdict, confidence=confidence,
        reason=reason, evidence_source="spatial",
    ))

    if verdict == Verdict.INCORRECT and confidence in (Confidence.HIGH, Confidence.MEDIUM):
        errors.append(CorrectionError(
            triple=triple, reason=reason, confidence=confidence,
        ))


# ── Action/attribute triple verification subroutine ──────────────────────


def _verify_action_attribute_triple(
    triple: Triple,
    kb: dict,
    pil_image: Image.Image,
    cross_captions: dict[str, str] | None,
    errors: list[CorrectionError],
    all_checks: list[VerificationResult],
    img_id: str,
) -> None:
    """Verify an ACTION or ATTRIBUTE triple and append results."""
    subj, rel, obj = triple.subject, triple.relation, triple.object

    # Action geometry pre-screen
    geo_family = classify_action_family(rel)
    geo_prereq = None
    kp = None
    if geo_family:
        s_box = find_best_bbox_from_kb(subj, kb)
        o_box = find_best_bbox_from_kb(obj, kb)
        if s_box and o_box:
            family_spec = ACTION_GEOMETRY_TAXONOMY.get(geo_family, {})
            if family_spec.get("needs_keypoints"):
                subj_lower = core_noun(subj)
                obj_lower = core_noun(obj)
                if subj_lower in _PERSON_WORDS:
                    kp = get_person_keypoints(pil_image, s_box)
                elif obj_lower in _PERSON_WORDS:
                    kp = get_person_keypoints(pil_image, o_box)
                    s_box, o_box = o_box, s_box

            geo_prereq = check_action_geometry(geo_family, s_box, o_box, keypoints=kp)
            if geo_prereq is False:
                tier = "Tier2-keypoint" if kp else "Tier1-bbox"
                log.debug("[%s] geometry flag (%s/%s): '%s' — confirming with VQA",
                          img_id, geo_family, tier, triple.claim)

    # Consensus pre-filter
    if cross_captions and _consensus_confirms_triple(subj, rel, obj, cross_captions):
        all_checks.append(VerificationResult(
            triple=triple, verdict=Verdict.CORRECT, confidence=Confidence.HIGH,
            reason="cross-captioner consensus", evidence_source="consensus",
        ))
        return

    # VQA verification
    verified, v_yes, v_no, v_total, v_contrastive_no = verify_action_triple(
        subj, rel, obj, kb, pil_image, n_questions=3,
    )
    unanimous_no = verified is False and v_no == v_total and v_total >= 2
    std_no = (v_no - 1) if v_contrastive_no else v_no
    contrastive_high = v_contrastive_no and std_no >= 1

    log.debug("[%s] %s triple (%r,%r,%r) geo=%s vqa=%s votes=%dY/%dN/%dT",
              img_id, triple.rel_type.value, subj, rel, obj,
              geo_prereq, verified, v_yes, v_no, v_total)

    if verified is True:
        verdict, confidence = Verdict.CORRECT, Confidence.MEDIUM
        reason = "crop VQA confirmed"
    elif verified is False:
        if geo_prereq is False:
            tier = "Tier2-keypoint" if kp else "Tier1-bbox"
            verdict, confidence = Verdict.INCORRECT, Confidence.HIGH
            reason = f"geometry ({geo_family}/{tier}) + VQA both FAILED"
        elif unanimous_no:
            verdict, confidence = Verdict.INCORRECT, Confidence.HIGH
            reason = f"unanimous VQA rejection ({v_no}/{v_total} NO)"
        elif contrastive_high:
            verdict, confidence = Verdict.INCORRECT, Confidence.HIGH
            reason = f"Maverick contrastive NO + {std_no} standard NO ({v_no}/{v_total} total)"
        else:
            verdict, confidence = Verdict.INCORRECT, Confidence.MEDIUM
            reason = f"split VQA rejection ({v_no}/{v_total} NO, no contrastive confirm)"

        if confidence in (Confidence.HIGH, Confidence.MEDIUM):
            errors.append(CorrectionError(
                triple=triple, reason=reason, confidence=confidence,
            ))
    else:
        if geo_prereq is False:
            verdict, confidence = Verdict.UNKNOWN, Confidence.LOW
            reason = "geometry flagged but VQA inconclusive — skipping"
        else:
            verdict, confidence = Verdict.UNKNOWN, Confidence.LOW
            reason = "could not verify (missing bbox or uncertain)"

    all_checks.append(VerificationResult(
        triple=triple, verdict=verdict, confidence=confidence,
        reason=reason, evidence_source="action_vqa",
    ))


# ── Batch correction application ─────────────────────────────────────────


def _apply_batch_correction(
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

        guidance = _build_correction_guidance(
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
            log.debug("[v2] batch correction REJECTED (ratio=%.2f, compressed=%s, garble=%s, same=%s)",
                      ratio, too_compressed, garble, candidate == caption)
            # Fallback: deletion
            high_errors = [e for e in errors if e.confidence in (Confidence.HIGH, Confidence.MEDIUM)]
            if high_errors:
                corrected, fallback_applied = _fallback_deletion(corrected, high_errors)
                if fallback_applied:
                    applied.append({"method": "fallback_deletion", "n_errors": len(high_errors)})
                    log.debug("[v2] fallback: applied deletion for %d error(s)", len(high_errors))

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


def _build_correction_guidance(
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
