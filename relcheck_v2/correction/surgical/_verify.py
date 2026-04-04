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
from ...config import ABSTRACT_ENTITIES, SPATIAL_OPPOSITES
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
from .._metrics import MetricsCollector
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
    metrics: MetricsCollector | None = None,
) -> None:
    """Verify a SPATIAL triple and append results to errors/all_checks.

    Args:
        triple: The spatial triple to verify.
        kb: Visual KB dict.
        pil_image: Full PIL image.
        spatial_facts: KB spatial fact strings.
        geo_set: Set of lowercased geometry contradiction strings.
        errors: Accumulator for confirmed errors.
        all_checks: Accumulator for all verification results.
        img_id: Image identifier.
        metrics: Optional metrics collector for path logging.
    """
    subj, rel, obj = triple.subject, triple.relation, triple.object
    claim_str = triple.claim

    # Metrics tracking variables
    kb_synonym_match = False
    kb_opposite_match = False
    geo_contradiction_fired = False
    vqa_cross_check_override: bool | None = None
    entity_existence_triggered = False
    entity_existence_result: bool | None = None
    kb_provided_correct_rel = False

    kb_triples = _parse_spatial_facts(spatial_facts)
    kb_spatial_triples_parsed = len(kb_triples)
    verdict, confidence, reason = Verdict.UNKNOWN, Confidence.LOW, "no geometry available"

    for kb_s, kb_r, kb_o in kb_triples:
        if entity_matches(subj, kb_s) and entity_matches(obj, kb_o):
            if kb_r.lower() in spatial_synonyms(rel):
                verdict, confidence, reason = Verdict.CORRECT, Confidence.HIGH, "geometry confirms"
                kb_synonym_match = True
                kb_provided_correct_rel = True
                break
            opp = SPATIAL_OPPOSITES.get(rel)
            if opp and kb_r == opp:
                verdict, confidence, reason = Verdict.INCORRECT, Confidence.HIGH, f"geometry shows {kb_r}"
                kb_opposite_match = True
                break

    if any(claim_str.lower() in g or (subj.lower() in g and rel.lower() in g) for g in geo_set):
        verdict = Verdict.INCORRECT
        confidence = Confidence.HIGH
        reason = "deterministic geometry contradiction"
        geo_contradiction_fired = True

    # Cross-check geometry INCORRECT with VQA
    if verdict == Verdict.INCORRECT and confidence == Confidence.HIGH:
        vqa_confirm, _, _, _, _ = verify_action_triple(subj, rel, obj, kb, pil_image, n_questions=2)
        if vqa_confirm is True:
            verdict, confidence = Verdict.CORRECT, Confidence.MEDIUM
            reason = "geometry said INCORRECT but Maverick VQA confirmed → kept"
            vqa_cross_check_override = True
        elif vqa_confirm is None:
            verdict, confidence = Verdict.UNKNOWN, Confidence.LOW
            reason = "geometry INCORRECT but Maverick uncertain → skipped"
            vqa_cross_check_override = False

    # Fallback: check entity existence + VQA
    kb_bbox_found_subject = find_best_bbox_from_kb(subj, kb) is not None
    kb_bbox_found_object = find_best_bbox_from_kb(obj, kb) is not None

    if verdict == Verdict.UNKNOWN:
        # Skip object-existence check for abstract/positional terms
        # (e.g. "center", "left side", "game") — these are not physical objects
        _obj_lower = obj.lower().strip()
        _subj_lower = subj.lower().strip()
        if _obj_lower in ABSTRACT_ENTITIES or _subj_lower in ABSTRACT_ENTITIES:
            log.debug("[%s] Skipping abstract entity '%s'/'%s' — not detectable",
                      img_id, subj, obj)
            # Leave as UNKNOWN → no correction
            all_checks.append(VerificationResult(
                triple=triple, verdict=Verdict.UNKNOWN, confidence=Confidence.LOW,
                reason=f"abstract entity ('{obj}') — skipped",
                evidence_source="spatial",
            ))
            return

        obj_box_check = find_best_bbox_from_kb(obj, kb)
        if obj_box_check is None:
            exists = check_entity_exists_vqa(obj, pil_image)
            entity_existence_triggered = True
            entity_existence_result = exists
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
                if correct_spatial:
                    kb_provided_correct_rel = True
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

    if metrics is not None:
        metrics.record_spatial_verification(img_id, {
            "claim": claim_str,
            "kb_synonym_match": kb_synonym_match,
            "kb_opposite_match": kb_opposite_match,
            "geo_contradiction_fired": geo_contradiction_fired,
            "vqa_cross_check_override": vqa_cross_check_override,
            "entity_existence_triggered": entity_existence_triggered,
            "entity_existence_result": entity_existence_result,
            "verdict": verdict.value,
            "confidence": confidence.value,
            "evidence_source": "spatial",
            "kb_spatial_triples_parsed": kb_spatial_triples_parsed,
            "kb_bbox_found_subject": kb_bbox_found_subject,
            "kb_bbox_found_object": kb_bbox_found_object,
            "kb_provided_correct_rel": kb_provided_correct_rel,
        })


# ── Action/attribute triple verification ────────────────────────────────


def verify_action_attribute_triple(
    triple: Triple,
    kb: dict,
    pil_image: Image.Image,
    cross_captions: dict[str, str] | None,
    errors: list[CorrectionError],
    all_checks: list[VerificationResult],
    img_id: str,
    metrics: MetricsCollector | None = None,
) -> None:
    """Verify an ACTION or ATTRIBUTE triple and append results.

    Args:
        triple: The action/attribute triple to verify.
        kb: Visual KB dict.
        pil_image: Full PIL image.
        cross_captions: Optional cross-captioner captions for consensus.
        errors: Accumulator for confirmed errors.
        all_checks: Accumulator for all verification results.
        img_id: Image identifier.
        metrics: Optional metrics collector for path logging.
    """
    from ._consensus import consensus_confirms_triple

    subj, rel, obj = triple.subject, triple.relation, triple.object

    # Metrics tracking variables
    keypoints_loaded = False
    consensus_confirmed = False
    vqa_yes_votes = 0
    vqa_no_votes = 0
    vqa_total = 0
    vqa_contrastive_no = False
    vqa_decision_category = "inconclusive"

    # KB bbox tracking
    kb_bbox_found_subject = find_best_bbox_from_kb(subj, kb) is not None
    kb_bbox_found_object = find_best_bbox_from_kb(obj, kb) is not None
    used_crop_vqa = kb_bbox_found_subject and kb_bbox_found_object

    # Skip abstract / non-detectable entities
    if obj.lower().strip() in ABSTRACT_ENTITIES or subj.lower().strip() in ABSTRACT_ENTITIES:
        log.debug("[%s] Skipping abstract entity in action triple '%s'", img_id, triple.claim)
        all_checks.append(VerificationResult(
            triple=triple, verdict=Verdict.UNKNOWN, confidence=Confidence.LOW,
            reason=f"abstract entity — skipped", evidence_source="action_vqa",
        ))
        return

    # Action geometry pre-screen
    geo_family = classify_action_family(rel)
    geo_prereq = None
    geo_check_possible = False
    kp = None
    if geo_family:
        s_box = find_best_bbox_from_kb(subj, kb)
        o_box = find_best_bbox_from_kb(obj, kb)
        if s_box and o_box:
            geo_check_possible = True
            family_spec = ACTION_GEOMETRY_TAXONOMY.get(geo_family, {})
            if family_spec.get("needs_keypoints"):
                subj_lower = core_noun(subj)
                obj_lower = core_noun(obj)
                if subj_lower in _PERSON_WORDS:
                    kp = get_person_keypoints(pil_image, s_box)
                elif obj_lower in _PERSON_WORDS:
                    kp = get_person_keypoints(pil_image, o_box)
                    s_box, o_box = o_box, s_box
                if kp is not None:
                    keypoints_loaded = True

            geo_prereq = check_action_geometry(geo_family, s_box, o_box, keypoints=kp)
            if geo_prereq is False:
                tier = "Tier2-keypoint" if kp else "Tier1-bbox"
                log.debug("[%s] geometry flag (%s/%s): '%s' — confirming with VQA",
                          img_id, geo_family, tier, triple.claim)

    # Consensus pre-filter
    if cross_captions and consensus_confirms_triple(subj, rel, obj, cross_captions):
        consensus_confirmed = True
        all_checks.append(VerificationResult(
            triple=triple, verdict=Verdict.CORRECT, confidence=Confidence.HIGH,
            reason="cross-captioner consensus", evidence_source="consensus",
        ))
        if metrics is not None:
            metrics.record_action_verification(img_id, {
                "claim": triple.claim,
                "rel_type": triple.rel_type.value,
                "action_geo_family": geo_family,
                "geo_prereq_result": geo_prereq,
                "keypoints_loaded": keypoints_loaded,
                "consensus_confirmed": consensus_confirmed,
                "vqa_yes_votes": vqa_yes_votes,
                "vqa_no_votes": vqa_no_votes,
                "vqa_total": vqa_total,
                "vqa_contrastive_no": vqa_contrastive_no,
                "vqa_decision_category": "confirmed",
                "verdict": Verdict.CORRECT.value,
                "confidence": Confidence.HIGH.value,
                "evidence_source": "consensus",
                "kb_bbox_found_subject": kb_bbox_found_subject,
                "kb_bbox_found_object": kb_bbox_found_object,
                "used_crop_vqa": used_crop_vqa,
                "geo_check_possible": geo_check_possible,
            })
        return

    # VQA verification
    verified, v_yes, v_no, v_total, v_contrastive_no = verify_action_triple(
        subj, rel, obj, kb, pil_image, n_questions=3,
    )
    vqa_yes_votes = v_yes
    vqa_no_votes = v_no
    vqa_total = v_total
    vqa_contrastive_no = bool(v_contrastive_no)

    unanimous_no = verified is False and v_no == v_total and v_total >= 2
    std_no = (v_no - 1) if v_contrastive_no else v_no
    contrastive_high = v_contrastive_no and std_no >= 1

    log.debug("[%s] %s triple (%r,%r,%r) geo=%s vqa=%s votes=%dY/%dN/%dT",
              img_id, triple.rel_type.value, subj, rel, obj,
              geo_prereq, verified, v_yes, v_no, v_total)

    if verified is True:
        verdict, confidence = Verdict.CORRECT, Confidence.MEDIUM
        reason = "crop VQA confirmed"
        vqa_decision_category = "confirmed"
    elif verified is False:
        if geo_prereq is False:
            tier = "Tier2-keypoint" if kp else "Tier1-bbox"
            verdict, confidence = Verdict.INCORRECT, Confidence.HIGH
            reason = f"geometry ({geo_family}/{tier}) + VQA both FAILED"
            vqa_decision_category = "unanimous_no"
        elif unanimous_no:
            verdict, confidence = Verdict.INCORRECT, Confidence.HIGH
            reason = f"unanimous VQA rejection ({v_no}/{v_total} NO)"
            vqa_decision_category = "unanimous_no"
        elif contrastive_high:
            verdict, confidence = Verdict.INCORRECT, Confidence.HIGH
            reason = f"Maverick contrastive NO + {std_no} standard NO ({v_no}/{v_total} total)"
            vqa_decision_category = "contrastive_high"
        else:
            verdict, confidence = Verdict.INCORRECT, Confidence.MEDIUM
            reason = f"split VQA rejection ({v_no}/{v_total} NO, no contrastive confirm)"
            vqa_decision_category = "split_no"

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
        vqa_decision_category = "inconclusive"

    all_checks.append(VerificationResult(
        triple=triple, verdict=verdict, confidence=confidence,
        reason=reason, evidence_source="action_vqa",
    ))

    if metrics is not None:
        metrics.record_action_verification(img_id, {
            "claim": triple.claim,
            "rel_type": triple.rel_type.value,
            "action_geo_family": geo_family,
            "geo_prereq_result": geo_prereq,
            "keypoints_loaded": keypoints_loaded,
            "consensus_confirmed": consensus_confirmed,
            "vqa_yes_votes": vqa_yes_votes,
            "vqa_no_votes": vqa_no_votes,
            "vqa_total": vqa_total,
            "vqa_contrastive_no": vqa_contrastive_no,
            "vqa_decision_category": vqa_decision_category,
            "verdict": verdict.value,
            "confidence": confidence.value,
            "evidence_source": "action_vqa",
            "kb_bbox_found_subject": kb_bbox_found_subject,
            "kb_bbox_found_object": kb_bbox_found_object,
            "used_crop_vqa": used_crop_vqa,
            "geo_check_possible": geo_check_possible,
        })
