"""
RelCheck v2 — Triple Verification Subroutines
================================================
Type-aware verification of individual triples:

- SPATIAL: deterministic geometry → VQA fallback
- ACTION/ATTRIBUTE: action geometry pre-screen → consensus → crop VQA
"""

from __future__ import annotations

from PIL import Image

from ..._logging import log
from ...config import SPATIAL_OPPOSITES
from ...detection import find_best_bbox_from_kb
from ...spatial import SPATIAL_TRIPLE_RE
from ...types import (
    CorrectionError, VerificationResult,
    Triple, RelationType, Verdict, Confidence,
)
from .._geometry import (
    ACTION_GEOMETRY_TAXONOMY, classify_action_family,
    get_person_keypoints, check_action_geometry,
)
from .._utils import core_noun, entity_matches, normalize_entity
from .._vqa import (
    check_entity_exists_vqa, verify_action_triple,
    query_correct_spatial_relation, _parse_spatial_facts,
)
from ._consensus import spatial_synonyms


# ── Person-like entity words (for keypoint routing) ────────────────────

_PERSON_WORDS: frozenset[str] = frozenset({
    "person", "man", "woman", "boy", "girl",
    "child", "kid", "baby", "player", "rider",
    "worker", "people", "dog", "cat", "horse",
})


# ── Spatial contradiction detection ─────────────────────────────────────


def check_spatial_contradictions(
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


# ── Spatial triple verification ─────────────────────────────────────────


def verify_spatial_triple(
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
            if kb_r.lower() in spatial_synonyms(rel):
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


# ── Action/attribute triple verification ────────────────────────────────


def verify_action_attribute_triple(
    triple: Triple,
    kb: dict,
    pil_image: Image.Image,
    cross_captions: dict[str, str] | None,
    errors: list[CorrectionError],
    all_checks: list[VerificationResult],
    img_id: str,
) -> None:
    """Verify an ACTION or ATTRIBUTE triple and append results."""
    from ._consensus import consensus_confirms_triple

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
    if cross_captions and consensus_confirms_triple(subj, rel, obj, cross_captions):
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
