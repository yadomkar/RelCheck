"""
RelCheck v3 Correction Pipeline

Post-hoc relational hallucination correction for multimodal LLM captions.
Supports two modes:
  - ENRICHMENT (short captions < 30 words): KB-guided full rewrite with errors fixed + missing facts added
  - CORRECTION (long captions >= 30 words): Type-aware per-triple verification + surgical editing

Type-aware verification splits by relation type:
  - SPATIAL: deterministic bbox geometry (zero hallucination risk)
  - ACTION/ATTRIBUTE: crop-based multi-question VQA with debiased forced-choice

Key features:
  - Consensus pre-filter: cross-captioner verification skips VQA (free signal)
  - Geometry + VQA confirmation: two independent signals for HIGH confidence
  - Batch correction: 2+ hallucinations fixed in single LLM call (no cascading)
  - Post-verification: corrected captions checked for new hallucinations
"""

import json
import re
import time
import base64
from io import BytesIO
from collections import Counter

from .api import vlm_call, llm_call, encode_b64
from .entity import levenshtein_distance, edit_rate
from .config import (
    ENTITY_SYNONYMS, SPATIAL_OPPOSITES, LLM_MODEL, SHORT_CAPTION_THRESHOLD,
)
from .detection import find_best_bbox_from_kb, crop_to_bboxes
from .spatial import _SPATIAL_TRIPLE_RE
from .prompts import (
    ANALYSIS_PROMPT, VERIFY_PROMPT, TRIPLE_EXTRACT_PROMPT,
    TRIPLE_CORRECT_PROMPT, BATCH_CORRECT_PROMPT, MISSING_FACTS_PROMPT,
)


# ============================================================
# Spatial Contradiction Detection & Geometry
# ============================================================

_SPATIAL_SYNONYM_GROUPS = [
    {"on", "above", "on top of", "over"},
    {"under", "below", "beneath", "underneath"},
    {"left", "to the left", "to the left of"},
    {"right", "to the right", "to the right of"},
    {"in front of", "before"},
    {"behind", "in back of"},
]

_ENTITY_STOP = frozenset([
    "a","an","the","some","is","are","was","were","be","been","being",
    "sits","sit","standing","stand","positioned","position","placed","place",
    "seen","located","appears","appear","lying","lies","lay","resting","rest",
])

ACTION_GEOMETRY_TAXONOMY = {
    "mounting": {
        "verbs": {"riding", "sitting on", "standing on", "straddling",
                  "mounted on", "perched on", "atop", "on top of",
                  "perching on", "seated on", "crouching on"},
        "geometric_rule": "subject_above_object",
        "needs_keypoints": False,
    },
    "containment": {
        "verbs": {"inside", "in", "enclosed by", "covered by",
                  "contained in", "within", "trapped in", "wrapped in"},
        "geometric_rule": "subject_inside_object",
        "needs_keypoints": False,
    },
    "adjacency": {
        "verbs": {"next to", "beside", "near", "alongside", "adjacent to",
                  "close to", "leaning on", "leaning against"},
        "geometric_rule": "bboxes_close",
        "needs_keypoints": False,
    },
    "grasping": {
        "verbs": {"holding", "carrying", "picking up", "pulling", "pushing",
                  "grabbing", "gripping", "lifting", "dragging", "clutching",
                  "catching", "throwing", "tossing"},
        "geometric_rule": "wrist_near_object",
        "needs_keypoints": True,
    },
    "consuming": {
        "verbs": {"eating", "drinking", "tasting", "licking", "biting",
                  "sipping", "chewing", "feeding on"},
        "geometric_rule": "nose_near_object",
        "needs_keypoints": True,
    },
}

_VERB_TO_FAMILY = {}
for _fam, _spec in ACTION_GEOMETRY_TAXONOMY.items():
    for _v in _spec["verbs"]:
        _VERB_TO_FAMILY[_v] = _fam

KP_NOSE        = 0
KP_LEFT_WRIST  = 9
KP_RIGHT_WRIST = 10


def _spatial_synonyms(rel: str) -> set:
    """Return the synonym set that contains rel, or {rel} if none."""
    rel_lower = rel.lower().strip()
    for group in _SPATIAL_SYNONYM_GROUPS:
        if rel_lower in group:
            return group
    return {rel_lower}


def normalize_entity(text: str) -> str:
    """Lowercase and strip articles for fuzzy matching."""
    if not text:
        return ""
    text = text.lower().strip()
    for art in ["a ", "an ", "the ", "some "]:
        if text.startswith(art):
            text = text[len(art):]
    return text.strip()


def _classify_action_family(relation_verb: str) -> str | None:
    """Map a relation verb to its physical family (or None if no rule exists)."""
    rel = relation_verb.strip().lower()
    if rel in _VERB_TO_FAMILY:
        return _VERB_TO_FAMILY[rel]
    for verb, fam in _VERB_TO_FAMILY.items():
        if len(verb.split()) >= 2 and verb in rel:
            return fam
    return None


# ============================================================
# ViTPose Integration (Tier 2 Action Geometry)
# ============================================================

def _get_person_keypoints(pil_image, person_box_norm: list) -> dict | None:
    """
    Run ViTPose on a detected person to get 17 COCO keypoints.
    Returns dict with 'keypoints' (17×2 pixel coords) and 'scores' (17,)
    or None if detection fails.
    """
    try:
        import torch
        from transformers import ViTPoseProcessor, ViTPoseForPoseEstimation

        vitpose_model = ViTPoseForPoseEstimation.from_pretrained(
            "google/vitpose-base-simple-coco", device_map="auto"
        )
        vitpose_processor = ViTPoseProcessor.from_pretrained(
            "google/vitpose-base-simple-coco"
        )
    except Exception:
        return None

    W, H = pil_image.size
    x1, y1, x2, y2 = person_box_norm
    coco_box = [x1 * W, y1 * H, (x2 - x1) * W, (y2 - y1) * H]

    try:
        inputs = vitpose_processor(
            pil_image, boxes=[[coco_box]], return_tensors="pt"
        ).to(vitpose_model.device)

        with torch.no_grad():
            outputs = vitpose_model(**inputs)

        results = vitpose_processor.post_process_pose_estimation(
            outputs, boxes=[[coco_box]]
        )

        if results and results[0]:
            kp = results[0][0]
            keypoints = kp["keypoints"].cpu().numpy()
            scores = kp["scores"].cpu().numpy()
            keypoints[:, 0] /= W
            keypoints[:, 1] /= H
            return {"keypoints": keypoints, "scores": scores}
    except Exception as e:
        print(f"    ViTPose error: {e}")
    return None


def _check_action_geometry(family: str, subj_box: list, obj_box: list,
                           keypoints: dict | None = None) -> bool | None:
    """
    Test geometric prerequisite for an action family.
    Returns True (prerequisite met), False (violated), or None (cannot check).
    """
    sx1, sy1, sx2, sy2 = subj_box
    ox1, oy1, ox2, oy2 = obj_box

    s_cx, s_cy = (sx1 + sx2) / 2, (sy1 + sy2) / 2
    o_cx, o_cy = (ox1 + ox2) / 2, (oy1 + oy2) / 2
    s_h = sy2 - sy1
    o_h = oy2 - oy1
    o_w = ox2 - ox1

    if family == "mounting":
        top_region = oy1 + 0.65 * o_h
        subject_bottom_in_top = sy2 <= top_region + 0.05
        x_overlap = min(sx2, ox2) - max(sx1, ox1)
        has_x_overlap = x_overlap > 0.02 * max(o_w, 0.01)
        return subject_bottom_in_top and has_x_overlap

    elif family == "containment":
        inter_x = max(0, min(sx2, ox2) - max(sx1, ox1))
        inter_y = max(0, min(sy2, oy2) - max(sy1, oy1))
        inter_area = inter_x * inter_y
        subj_area = max((sx2 - sx1) * (sy2 - sy1), 1e-6)
        containment_ratio = inter_area / subj_area
        return containment_ratio > 0.50

    elif family == "adjacency":
        gap_x = max(0, max(sx1, ox1) - min(sx2, ox2))
        gap_y = max(0, max(sy1, oy1) - min(sy2, oy2))
        gap = (gap_x**2 + gap_y**2) ** 0.5
        avg_size = ((s_h + o_h) / 2 + ((sx2-sx1) + o_w) / 2) / 2
        return gap < 0.30 * max(avg_size, 0.01)

    if keypoints is None:
        return None

    kp_xy = keypoints["keypoints"]
    kp_sc = keypoints["scores"]
    KP_CONF_THRESH = 0.3

    if family == "grasping":
        margin_x = max(0.5 * o_w, 0.03)
        margin_y = max(0.5 * o_h, 0.03)
        obj_expanded = [ox1 - margin_x, oy1 - margin_y,
                        ox2 + margin_x, oy2 + margin_y]
        for wrist_idx in [KP_LEFT_WRIST, KP_RIGHT_WRIST]:
            if kp_sc[wrist_idx] < KP_CONF_THRESH:
                continue
            wx, wy = kp_xy[wrist_idx]
            if (obj_expanded[0] <= wx <= obj_expanded[2] and
                obj_expanded[1] <= wy <= obj_expanded[3]):
                return True
        return False

    elif family == "consuming":
        if kp_sc[KP_NOSE] < KP_CONF_THRESH:
            return None
        nx, ny = kp_xy[KP_NOSE]
        margin_x = max(0.75 * o_w, 0.04)
        margin_y = max(0.75 * o_h, 0.04)
        obj_expanded = [ox1 - margin_x, oy1 - margin_y,
                        ox2 + margin_x, oy2 + margin_y]
        if (obj_expanded[0] <= nx <= obj_expanded[2] and
            obj_expanded[1] <= ny <= obj_expanded[3]):
            return True
        return False

    return True


# ============================================================
# Spatial Fact Parsing & Triple Extraction
# ============================================================

def _parse_spatial_facts(spatial_facts: list) -> list:
    """Parse KB's spatial_facts list into (subj, rel, obj) tuples."""
    parsed = []
    for fact in spatial_facts:
        fact_clean = fact.replace("'", "").replace('"', "")
        for m in _SPATIAL_TRIPLE_RE.finditer(fact_clean.lower()):
            subj = m.group(1).strip().rstrip(" ,;")
            rel  = m.group(2).strip()
            obj  = m.group(3).strip().rstrip(" ,;.")
            parsed.append((subj, rel, obj))
    return parsed


def _core_noun(text: str) -> str:
    """Extract core noun from entity span, stripping filler words."""
    words = normalize_entity(text).split()
    while words and words[0] in _ENTITY_STOP:
        words = words[1:]
    while words and words[-1] in _ENTITY_STOP:
        words = words[:-1]
    return " ".join(words[:3]).strip()


def _entity_matches(cap_entity: str, kb_entity: str) -> bool:
    """Fuzzy match using core noun extraction."""
    core_cap = _core_noun(cap_entity)
    core_kb  = _core_noun(kb_entity)
    if not core_cap or not core_kb:
        return False
    return (core_kb in core_cap) or (core_cap in core_kb)


def _check_spatial_contradictions(caption: str, spatial_facts: list) -> list:
    """
    Deterministically detect spatial contradictions between caption and KB.
    Returns list of contradiction strings.
    """
    if not spatial_facts:
        return []

    kb_triples = _parse_spatial_facts(spatial_facts)
    if not kb_triples:
        return []

    caption_triples = _extract_spatial_triples_text(caption)
    if not caption_triples:
        return []

    contradictions = []
    for (cap_subj, cap_rel, cap_obj) in caption_triples:
        cap_opposite = SPATIAL_OPPOSITES.get(cap_rel.lower())
        if not cap_opposite:
            continue

        for (kb_subj, kb_rel, kb_obj) in kb_triples:
            if not (_entity_matches(cap_subj, kb_subj) and _entity_matches(cap_obj, kb_obj)):
                continue
            if kb_rel == cap_opposite:
                kb_fact_str = next(
                    (f for f in spatial_facts
                     if normalize_entity(kb_subj) in f.lower()
                     and normalize_entity(kb_obj) in f.lower()),
                    f"'{kb_subj}' {kb_rel} '{kb_obj}'"
                )
                contradictions.append(
                    f"Caption says '{cap_subj} {cap_rel} {cap_obj}' "
                    f"but geometry shows: {kb_fact_str}"
                )
                break

    return contradictions


def _extract_spatial_triples_text(text: str) -> list:
    """Extract (subject, relation, object) spatial triples from free text."""
    triples = []
    for m in _SPATIAL_TRIPLE_RE.finditer(text.lower()):
        subj = m.group(1).strip().rstrip(" ,;")
        rel  = m.group(2).strip()
        obj  = m.group(3).strip().rstrip(" ,;.")
        triples.append((subj, rel, obj))
    return triples


# ============================================================
# VQA & Relation Querying
# ============================================================

def _check_entity_exists_vqa(entity: str, pil_image, retries: int = 2) -> bool | None:
    """Full-image VQA: ask if an entity exists in the image."""
    if pil_image is None:
        return None
    buf = BytesIO()
    pil_image.convert("RGB").save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()

    questions = [
        f"Is there a {entity} visible anywhere in this image? Answer only YES or NO.",
        f"Can you see a {entity} in this scene? Look carefully. Answer YES or NO only.",
    ]
    yes_v = no_v = 0
    for q in questions:
        r = vlm_call([{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": q},
        ]}], max_tokens=5)
        if not r:
            continue
        rl = r.strip().lower()
        if "yes" in rl and "no" not in rl:
            yes_v += 1
        elif "no" in rl:
            no_v += 1
    total = yes_v + no_v
    if total == 0:
        return None
    if no_v > yes_v:
        return False
    if yes_v > no_v:
        return True
    return None


def _query_correct_spatial_relation(subj: str, obj: str, kb: dict,
                                   pil_image) -> str | None:
    """
    Determine correct spatial relation between subj and obj.
    KB-first: checks deterministic geometry before VLM.
    """
    if pil_image is None:
        return None

    spatial_facts = kb.get("spatial_facts", [])
    kb_triples = _parse_spatial_facts(spatial_facts)
    for kb_s, kb_r, kb_o in kb_triples:
        if _entity_matches(subj, kb_s) and _entity_matches(obj, kb_o):
            print(f"      [correct_spatial] KB hit: {kb_s} {kb_r} {kb_o}")
            return kb_r
        if _entity_matches(subj, kb_o) and _entity_matches(obj, kb_s):
            rev = SPATIAL_OPPOSITES.get(kb_r.lower())
            if rev:
                print(f"      [correct_spatial] KB hit (reversed): {rev}")
                return rev

    subj_box = find_best_bbox_from_kb(subj, kb)
    obj_box  = find_best_bbox_from_kb(obj,  kb)

    using_crop = bool(subj_box and obj_box)
    if using_crop:
        crop = crop_to_bboxes(pil_image, subj_box, obj_box, padding=0.25)
        buf = BytesIO()
        crop.convert("RGB").save(buf, format="JPEG", quality=85)
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        region_hint = "this cropped region"
    else:
        buf = BytesIO()
        pil_image.convert("RGB").save(buf, format="JPEG", quality=85)
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        region_hint = "the full image"

    prompt = (
        f"Look at this image. Where is the {obj} relative to the {subj}? "
        f"Reply with ONLY a short spatial phrase "
        f"(e.g. 'to the left of', 'to the right of', 'above', 'below', "
        f"'in front of', 'behind', 'next to'). "
        f"If you cannot determine the relationship clearly, reply 'unknown'."
    )
    raw = vlm_call([{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
        {"type": "text", "text": prompt},
    ]}], max_tokens=15)

    if raw:
        r = raw.strip().strip('"').lower()
        if r != "unknown" and len(r.split()) <= 5:
            verify_q = (
                f"Look at {region_hint}. Is the {obj} {r} the {subj}? "
                f"Answer YES or NO only."
            )
            verify_raw = vlm_call([{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                {"type": "text", "text": verify_q},
            ]}], max_tokens=5)
            if verify_raw and "no" in verify_raw.strip().lower():
                print(f"      [correct_spatial] VLM suggestion '{r}' REJECTED by verify")
                return "near"
            print(f"      [correct_spatial] VLM verified: {r}")
            return r
    return None


def _query_correct_action_relation(subj: str, wrong_rel: str, obj: str, kb: dict,
                                  pil_image) -> str | None:
    """
    Determine correct relation between subj and obj.
    Three-stage: KB lookup → constrained selection → verify.
    """
    if pil_image is None:
        return None

    spatial_facts = kb.get("spatial_facts", [])
    kb_triples = _parse_spatial_facts(spatial_facts)
    for kb_s, kb_r, kb_o in kb_triples:
        if _entity_matches(subj, kb_s) and _entity_matches(obj, kb_o):
            print(f"      [correct_rel] KB hit: {kb_s} {kb_r} {kb_o}")
            return kb_r
        if _entity_matches(subj, kb_o) and _entity_matches(obj, kb_s):
            rev = SPATIAL_OPPOSITES.get(kb_r.lower())
            if rev:
                print(f"      [correct_rel] KB hit (reversed): {rev}")
                return rev

    vis_desc = kb.get("visual_description", "")
    if vis_desc:
        subj_core = _core_noun(subj)
        obj_core  = _core_noun(obj)
        if subj_core in vis_desc.lower() and obj_core in vis_desc.lower():
            extract_prompt = (
                f"From this description, what is the relationship between "
                f"'{subj}' and '{obj}'? Description: \"{vis_desc}\"\n"
                f"Reply with ONLY a 1-4 word relation phrase. If not mentioned, reply 'unclear'."
            )
            raw = llm_call([{"role": "user", "content": extract_prompt}], max_tokens=15)
            if raw:
                r = raw.strip().strip('"\'').lower()
                if r not in ("unclear", "unknown", "not mentioned") and 1 <= len(r.split()) <= 4:
                    print(f"      [correct_rel] visual_description extraction: {r}")
                    return r

    subj_box = find_best_bbox_from_kb(subj, kb)
    obj_box  = find_best_bbox_from_kb(obj,  kb)

    using_crop = bool(subj_box and obj_box)
    if using_crop:
        crop = crop_to_bboxes(pil_image, subj_box, obj_box, padding=0.20)
        buf = BytesIO()
        crop.convert("RGB").save(buf, format="JPEG", quality=85)
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        region_hint = "this cropped region"
    else:
        buf = BytesIO()
        pil_image.convert("RGB").save(buf, format="JPEG", quality=85)
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        region_hint = "the full image"

    _ACTION_CANDIDATES = [
        "standing next to", "looking at", "walking toward", "sitting near",
        "reaching for", "talking to", "facing", "behind",
    ]
    _SPATIAL_CANDIDATES = [
        "to the left of", "to the right of", "above", "below",
        "next to", "in front of", "behind", "on top of",
    ]
    _spatial_kw = {"on", "above", "below", "left", "right", "next", "near",
                   "behind", "front", "top", "under", "over", "beside"}
    is_spatial = any(kw in wrong_rel.lower().split() for kw in _spatial_kw)
    candidates = _SPATIAL_CANDIDATES if is_spatial else _ACTION_CANDIDATES
    candidates = [c for c in candidates if c.lower() != wrong_rel.lower()][:6]

    letters = "ABCDEF"
    options_str = "\n".join(
        f"({letters[i]}) The {subj} is {c} the {obj}"
        for i, c in enumerate(candidates)
    )

    selection_prompt = (
        f"Look at {region_hint} carefully. The caption incorrectly says "
        f"the {subj} is '{wrong_rel}' the {obj}. "
        f"Which of these alternatives BEST describes their actual relationship?\n"
        f"{options_str}\n"
        f"(Z) None of these — the relationship is something else entirely\n\n"
        f"Answer with ONLY the letter."
    )

    raw = vlm_call([{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
        {"type": "text", "text": selection_prompt},
    ]}], max_tokens=5)

    selected_rel = None
    if raw:
        r = raw.strip().upper()
        for i, letter in enumerate(letters[:len(candidates)]):
            if letter in r and "Z" not in r:
                selected_rel = candidates[i]
                break

    if selected_rel is None:
        open_prompt = (
            f"Look at {region_hint} carefully. "
            f"The caption incorrectly claims the {subj} is '{wrong_rel}' the {obj}. "
            f"In 1-4 words, describe what the {subj} is ACTUALLY doing relative to the {obj}. "
            f"Reply ONLY with the short phrase. If truly unclear, reply 'unclear'."
        )
        raw2 = vlm_call([{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            {"type": "text", "text": open_prompt},
        ]}], max_tokens=15)
        if raw2:
            r2 = raw2.strip().strip('"\'').lower()
            if r2 not in ("unclear", "unknown") and 1 <= len(r2.split()) <= 4:
                selected_rel = r2

    if selected_rel is None:
        print(f"      [correct_rel] FAILED: no candidate selected")
        return None

    verify_prompt = (
        f"Look at {region_hint}. Is the {subj} {selected_rel} the {obj}? "
        f"Answer YES or NO only."
    )
    raw3 = vlm_call([{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
        {"type": "text", "text": verify_prompt},
    ]}], max_tokens=5)

    if raw3:
        r3 = raw3.strip().lower()
        if "yes" in r3 and "no" not in r3:
            print(f"      [correct_rel] VERIFIED: {selected_rel}")
            return selected_rel
        elif "no" in r3:
            print(f"      [correct_rel] REJECTED: {selected_rel} — falling back to 'near'")
            return "near"

    print(f"      [correct_rel] UNVERIFIED (VLM unclear): {selected_rel}")
    return selected_rel


def _verify_action_triple(subj: str, rel: str, obj: str, kb: dict,
                         pil_image, n_questions: int = 3) -> tuple:
    """
    Verify action/attribute triple using crop-based VQA with multi-question voting.
    Returns (verdict, yes_votes, no_votes, total, contrastive_no).
    """
    subj_box = find_best_bbox_from_kb(subj, kb)
    obj_box  = find_best_bbox_from_kb(obj,  kb)

    using_crop = bool(subj_box and obj_box)
    if using_crop:
        crop = crop_to_bboxes(pil_image, subj_box, obj_box, padding=0.15)
        buf  = BytesIO()
        crop.convert("RGB").save(buf, format="JPEG", quality=85)
        region_hint = "this cropped region"
    else:
        buf = BytesIO()
        pil_image.convert("RGB").save(buf, format="JPEG", quality=85)
        region_hint = "the full image"
    crop_b64 = base64.b64encode(buf.getvalue()).decode()

    _COUNTERFACTUAL_MAP = {
        "riding": "standing next to", "sitting on": "standing near",
        "holding": "standing next to", "carrying": "walking away from",
        "wearing": "next to", "eating": "looking at",
        "pulling": "pushing", "pushing": "pulling",
        "throwing": "holding", "catching": "dropping",
        "driving": "standing near", "leading": "following",
        "playing with": "ignoring", "using": "near",
        "walking": "standing", "running": "standing",
        "standing on": "next to", "lying on": "sitting near",
        "hanging from": "standing near", "leaning on": "standing near",
    }
    counterfactual_rel = _COUNTERFACTUAL_MAP.get(rel.lower(), f"not {rel}")
    _ab_flip = (hash(f"{subj}{rel}{obj}") % 2 == 1)
    if _ab_flip:
        _opt_a, _opt_b = counterfactual_rel, rel
    else:
        _opt_a, _opt_b = rel, counterfactual_rel
    contrastive_q = (
        f'Look at {region_hint} carefully. Which description is more accurate?\n'
        f'(A) The {subj} is {_opt_a} the {obj}\n'
        f'(B) The {subj} is {_opt_b} the {obj}\n'
        f'Answer with ONLY the letter A or B.'
    )

    question_templates = [
        f'In this image, is the {subj} {rel} the {obj}? '
        f'Look carefully at {region_hint}. Answer only YES or NO.',
        f'Does the {subj} appear to be {rel} the {obj} here? '
        f'Observe {region_hint} closely. Answer YES or NO only.',
    ]
    questions_yn = question_templates[:min(n_questions - 1, 2)]

    yes_votes = 0
    no_votes  = 0

    for q in questions_yn:
        result = vlm_call([{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{crop_b64}"}},
            {"type": "text", "text": q},
        ]}], max_tokens=5)
        if not result:
            continue
        r = result.strip().lower()
        if "yes" in r and "no" not in r:
            yes_votes += 1
        elif "no" in r:
            no_votes  += 1

    contrastive_no = False
    contrastive_result = vlm_call([{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{crop_b64}"}},
        {"type": "text", "text": contrastive_q},
    ]}], max_tokens=5)
    if contrastive_result:
        cr    = contrastive_result.strip().upper()
        chose_a = ("A" in cr and "B" not in cr)
        chose_b = ("B" in cr and "A" not in cr)
        if _ab_flip:
            if chose_b:
                yes_votes += 1
            elif chose_a:
                no_votes    += 1
                contrastive_no = True
        else:
            if chose_a:
                yes_votes += 1
            elif chose_b:
                no_votes    += 1
                contrastive_no = True

    total = yes_votes + no_votes
    if total == 0:
        result = None
    elif yes_votes > no_votes:
        result = True
    elif no_votes > yes_votes:
        result = False
    else:
        result = None

    return (result, yes_votes, no_votes, total, contrastive_no)


# ============================================================
# Caption Enrichment & Correction
# ============================================================

def _apply_geometry_corrections(caption: str, geo_contradictions: list) -> tuple:
    """Surgical fallback: replace wrong spatial phrases using deterministic geometry."""
    if not geo_contradictions:
        return caption, 0

    fixed = caption
    n_fixed = 0

    for contra in geo_contradictions:
        m_cap = re.search(r"Caption says '(.+?)'", contra, re.IGNORECASE)
        if not m_cap:
            continue
        cap_claim = m_cap.group(1).strip()

        m_rel = _SPATIAL_TRIPLE_RE.search(cap_claim.lower())
        if not m_rel:
            continue
        wrong_rel = m_rel.group(2).strip()
        correct_rel = SPATIAL_OPPOSITES.get(wrong_rel)
        if not correct_rel:
            continue

        claim_lower = cap_claim.lower()
        fixed_lower = fixed.lower()
        idx = fixed_lower.find(claim_lower)
        if idx < 0:
            rel_idx = fixed_lower.find(wrong_rel)
            if rel_idx < 0:
                continue
            context = fixed_lower[max(0, rel_idx - 60): rel_idx + 60]
            subj = m_rel.group(1).strip()
            obj_ = m_rel.group(3).strip()
            core_s = _core_noun(subj)
            core_o = _core_noun(obj_)
            if not (core_s in context or core_o in context):
                continue
            fixed = fixed[:rel_idx] + correct_rel + fixed[rel_idx + len(wrong_rel):]
            n_fixed += 1
            continue

        phrase = fixed[idx: idx + len(cap_claim)]
        corrected_phrase = re.sub(
            r'\b' + re.escape(wrong_rel) + r'\b',
            correct_rel, phrase, flags=re.IGNORECASE, count=1,
        )
        if corrected_phrase != phrase:
            fixed = fixed[:idx] + corrected_phrase + fixed[idx + len(cap_claim):]
            n_fixed += 1

    return fixed, n_fixed


def _caption_name_for(kb_entity: str, cap_lower: str) -> str:
    """Return the synonym of kb_entity that appears in cap_lower."""
    core = _core_noun(kb_entity)
    if not core:
        return kb_entity
    if core in cap_lower:
        return core
    for syn in ENTITY_SYNONYMS.get(core, []):
        if syn in cap_lower:
            return syn
    return core


def _relation_already_expressed(subj: str, rel: str, obj: str, cap_lower: str) -> bool:
    """Check if caption already states subj <rel> obj or equivalent."""
    rel_norm = rel.lower()
    rel_variants = {
        "left":  ["left"],  "right": ["right"],
        "above": ["above","on top of"],  "below": ["below","beneath","under"],
        "behind":["behind","in back of"], "in front of":["in front of","front"],
    }
    keywords = rel_variants.get(rel_norm, [rel_norm])
    for kw in keywords:
        if kw not in cap_lower:
            continue
        idx = cap_lower.find(kw)
        context = cap_lower[max(0, idx - 80): idx + 80]
        core_s = _core_noun(subj)
        core_o = _core_noun(obj)
        if core_s in context or core_o in context:
            return True
        for syn in ENTITY_SYNONYMS.get(core_s, []):
            if syn in context:
                return True
        for syn in ENTITY_SYNONYMS.get(core_o, []):
            if syn in context:
                return True
    return False


def _build_spatial_addendum(corrected_caption: str, kb: dict,
                            max_facts: int = 5) -> tuple:
    """Append missing KB spatial facts to caption."""
    spatial_facts = kb.get("spatial_facts", [])
    if not spatial_facts:
        return corrected_caption, 0

    cap_lower = corrected_caption.lower()
    missing = []

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
        cap_obj  = _caption_name_for(obj,  cap_lower)
        missing.append((subj, rel, obj, fact, cap_subj, cap_obj))
        if len(missing) >= max_facts:
            break

    if not missing:
        return corrected_caption, 0

    fact_phrases = [f"the {cs} is {r} the {co}" for s, r, o, _, cs, co in missing]
    if len(fact_phrases) == 1:
        addendum = f"Spatially, {fact_phrases[0]}."
    elif len(fact_phrases) == 2:
        addendum = f"Spatially, {fact_phrases[0]}, and {fact_phrases[1]}."
    else:
        joined = ", ".join(fact_phrases[:-1]) + f", and {fact_phrases[-1]}"
        addendum = f"Spatially, {joined}."

    new_cap = corrected_caption.rstrip() + " " + addendum
    return new_cap, len(missing)


def _extract_triples(caption: str) -> list:
    """Use Llama to extract (subject, relation, object, type) triples."""
    _SPATIAL_WORDS = {"left","right","above","below","behind","front",
                      "beside","next to","on top of","under","over",
                      "in front of","near","inside","outside","between"}
    _ACTION_WORDS  = {"riding","holding","carrying","eating","drinking",
                      "wearing","pushing","pulling","walking","running",
                      "sitting","standing","playing","using","throwing",
                      "catching","carrying","holding","driving","leading"}

    prompt = TRIPLE_EXTRACT_PROMPT.format(caption=caption)

    raw = llm_call(
        [{"role": "user", "content": prompt}],
        max_tokens=600,
    )
    if not raw:
        print(f"    [extract_triples] llm_call returned empty for caption: {caption[:80]!r}")
        return []

    raw_stripped = re.sub(r'^```json\s*', '', raw.strip(), flags=re.MULTILINE)
    raw_stripped = re.sub(r'```\s*$', '', raw_stripped.strip(), flags=re.MULTILINE)
    raw_stripped = raw_stripped.strip()

    print(f"    [extract_triples] raw={raw_stripped[:200]!r}")

    try:
        parsed = json.loads(raw_stripped)

        if isinstance(parsed, dict):
            for key in ("triples", "relations", "result", "data", "output"):
                if key in parsed and isinstance(parsed[key], list):
                    parsed = parsed[key]
                    print(f"    [extract_triples] unwrapped dict key '{key}'")
                    break
            else:
                print(f"    [extract_triples] got dict but no list key: {list(parsed.keys())}")
                return []

        if not isinstance(parsed, list):
            print(f"    [extract_triples] unexpected type: {type(parsed)}")
            return []

        result = []
        for t in parsed:
            if not isinstance(t, dict):
                continue
            if not all(k in t for k in ("subject", "relation", "object")):
                print(f"    [extract_triples] skipping malformed triple: {t}")
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

            t["type"] = typ
            result.append(t)

        return result

    except json.JSONDecodeError as e:
        print(f"    [extract_triples] JSON parse error: {e} | raw={raw_stripped[:300]!r}")
    except Exception as e:
        print(f"    [extract_triples] unexpected error: {e}")
    return []


def _consensus_confirms_triple(subj: str, rel: str, obj: str,
                               cross_captions: dict) -> bool:
    """Check if all cross-captioners mention (subj rel obj)."""
    if not cross_captions or len(cross_captions) < 2:
        return False

    core_s   = _core_noun(subj)
    core_o   = _core_noun(obj)
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
            context = cap_lower[max(0, idx - 80): idx + 80]
            subj_near = (core_s in context or
                         any(s in context for s in ENTITY_SYNONYMS.get(core_s, [])))
            obj_near  = (core_o in context or
                         any(s in context for s in ENTITY_SYNONYMS.get(core_o, [])))
            if subj_near and obj_near:
                confirmed += 1
                break
            idx = cap_lower.find(rel_lower, idx + 1)

    return confirmed >= len(cross_captions)


def _apply_triple_correction(caption: str, wrong_phrase: str, correct_phrase: str,
                            subj: str = "", obj_: str = "") -> str:
    """Fix exactly one relationship word/phrase in the caption."""
    cap_lower = caption.lower()
    wp_lower  = wrong_phrase.lower()

    if wp_lower in cap_lower:
        occurrences = []
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
            subj_idx = cap_lower.find(_core_noun(subj)) if subj else -1
            obj_idx  = cap_lower.find(_core_noun(obj_)) if obj_ else -1
            def _proximity(i):
                d = 0
                if subj_idx >= 0:
                    d += abs(i - subj_idx)
                if obj_idx >= 0:
                    d += abs(i - obj_idx)
                return d
            idx = min(occurrences, key=_proximity)

        return caption[:idx] + correct_phrase + caption[idx + len(wrong_phrase):]

    raw = llm_call([{"role": "user", "content": TRIPLE_CORRECT_PROMPT.format(
        caption=caption,
        subj=subj or wrong_phrase,
        obj=obj_ or correct_phrase,
        wrong_phrase=wrong_phrase,
        correct_phrase=correct_phrase,
    )}], max_tokens=int(len(caption.split()) * 2.5))

    if raw:
        raw = raw.strip().strip('"').strip("'")
        ratio = len(raw) / max(len(caption), 1)
        if 0.85 <= ratio <= 1.25:
            return raw

    return caption


def _add_missing_fact_addendum(corrected_caption: str, kb: dict) -> tuple:
    """Insert facts from visual description that are absent from caption."""
    visual_desc = kb.get("visual_description", "")
    if not visual_desc or len(visual_desc.strip()) < 20:
        return corrected_caption, 0

    if len(corrected_caption.split()) > 300:
        return corrected_caption, 0

    prompt = MISSING_FACTS_PROMPT.format(
        caption=corrected_caption[:1500],
        visual_description=visual_desc[:1500],
    )
    raw = llm_call([{"role": "user", "content": prompt}],
                   max_tokens=800)
    if not raw:
        return corrected_caption, 0

    result = raw.strip().strip('"').strip("'").strip()

    if result.upper() == "NONE" or not result:
        return corrected_caption, 0

    orig_words = len(corrected_caption.split())
    new_words  = len(result.split())
    added = new_words - orig_words

    if added <= 0:
        return corrected_caption, 0
    if added > 30:
        return corrected_caption, 0

    orig_lower = corrected_caption.lower()
    result_lower = result.lower()
    orig_tokens = orig_lower.split()
    surviving = sum(1 for t in orig_tokens if t in result_lower)
    if surviving / max(len(orig_tokens), 1) < 0.80:
        return corrected_caption, 0

    _STOPWORDS = {"a","an","the","and","or","but","in","on","at","to","of",
                  "is","are","was","were","be","been","being","with","by",
                  "for","from","that","this","it","its","he","she","they"}
    result_toks = result.lower().split()
    orig_toks   = corrected_caption.lower().split()

    for i in range(len(result_toks) - 5):
        window = result_toks[i:i+7]
        for span in (2, 3):
            gram = tuple(window[:span])
            rest = window[1:]
            for j in range(len(rest) - span + 1):
                if tuple(rest[j:j+span]) == gram:
                    return corrected_caption, 0

    orig_bigrams = set(zip(orig_toks, orig_toks[1:]))
    result_bigrams = [tuple(pair) for pair in zip(result_toks, result_toks[1:])]
    for bg in result_bigrams:
        if bg not in orig_bigrams and result_bigrams.count(bg) >= 2:
            return corrected_caption, 0

    return result, 1


def _enrich_short_caption(img_id: str, caption: str, kb: dict) -> dict:
    """Full KB-guided enrichment for captions under 30 words."""
    hard    = "\n".join(f"- {f}" for f in kb["hard_facts"])    or "- None detected"
    spatial = "\n".join(f"- {f}" for f in kb["spatial_facts"]) or "- No spatial facts"
    visual  = kb["visual_description"][:800]                     or "- No visual description"

    prompt = ANALYSIS_PROMPT.format(
        caption=caption, hard_facts=hard,
        spatial_facts=spatial, visual_description=visual
    )
    improved = caption
    errors, missing = [], []

    raw = llm_call([{"role": "user", "content": prompt}], max_tokens=1000)
    if raw:
        raw = re.sub(r'^```json\s*', '', raw)
        raw = re.sub(r'```\s*$', '', raw)
        if raw.count('{') > raw.count('}'):
            raw += '"}'
        try:
            result  = json.loads(raw)
            errors  = result.get("errors", [])
            missing = result.get("missing", [])
            cand    = result.get("improved_caption", "").strip().strip('"').strip("'")
            if cand and len(cand) >= 15:
                improved = cand
        except Exception:
            pass

    if improved != caption:
        n_sent = len([s for s in re.split(r'[.!?]+', improved) if s.strip()])
        if n_sent > 10:
            improved = caption

    if improved != caption:
        counts  = Counter(l for l, _, _ in kb.get("detections", []))
        obj_str = ", ".join(f"{c}x {l}" for l, c in counts.most_common(10))
        rel_str = kb["visual_description"][:500]
        verdict = llm_call(
            [{"role": "user", "content": VERIFY_PROMPT.format(
                rewritten=improved, objects=obj_str, relationships=rel_str
            )}],
            max_tokens=50,
        )
        if verdict and verdict.upper().startswith("FAIL"):
            improved = caption

    edit_distance = levenshtein_distance(caption, improved)
    edit_rate_val = edit_distance / max(len(caption), len(improved), 1)
    return {
        "caption": caption, "corrected": improved,
        "errors": errors,   "missing": missing,
        "edit_rate": edit_rate_val,
        "status": "modified" if improved != caption else "unchanged",
        "mode": "enrich",
    }


def _extract_correct_rel_from_reason(reason_str: str) -> str | None:
    """Parse stored correct relation from verifier reason strings."""
    m = re.search(r"geometry shows (\S+)", reason_str or "")
    if m:
        return m.group(1).replace("_", " ")
    m = re.search(r"correct relation: '([^']+)'", reason_str or "")
    if m:
        return m.group(1)
    return None


def _has_garble(text: str) -> bool:
    """Check for artifacts from bad character-level insertion."""
    t = text.lower()
    if re.search(r'\w+\s+it\s+up', t): return True
    if "mat it" in t: return True
    if re.search(r'\S{30,}', t): return True
    return False


def _has_self_contradiction(original_cap: str, corrected_cap: str) -> bool:
    """Check for contradictory replacements (e.g. 'above' and 'below' coexisting)."""
    _ALL_OPP = {**SPATIAL_OPPOSITES}
    _ALL_OPP.update({
        "pull": "push", "push": "pull",
        "pulling": "pushing", "pushing": "pulling",
        "pulled": "pushed", "pushed": "pulled",
        "lift": "lower", "lifting": "lowering", "lifted": "lowered",
        "lower": "lift", "lowering": "lifting", "lowered": "lifted",
        "open": "close", "opening": "closing", "opened": "closed",
        "close": "open", "closing": "opening", "closed": "opened",
        "enter": "exit", "entering": "exiting", "entered": "exited",
        "exit": "enter", "exiting": "entering", "exited": "entered",
        "stand": "sit", "standing": "sitting", "stood": "sat",
        "sit": "stand", "sitting": "standing",
        "ascend": "descend", "ascending": "descending",
        "descend": "ascend", "descending": "ascending",
    })

    orig_words = original_cap.lower().split()
    orig_cnt   = Counter(orig_words)
    corr_lower = corrected_cap.lower()
    corr_words = corr_lower.split()
    corr_cnt   = Counter(corr_words)

    for word in set(orig_words):
        if len(word) < 4:
            continue
        opp = _ALL_OPP.get(word)
        if opp is None:
            continue
        o_c = orig_cnt[word]
        c_c = corr_cnt.get(word, 0)

        if 0 < c_c < o_c and opp in corr_lower:
            return True

        if word in corr_lower and opp in corr_lower:
            return True

    return False


def _correct_long_caption_v2(img_id: str, caption: str, kb: dict, pil_image=None,
                            cross_captions: dict = None) -> dict:
    """
    Per-triple verification correction for detailed captions (>= 30 words).

    Returns dict with 'corrected' caption and metadata.
    """
    if pil_image is None:
        return {
            "caption": caption, "corrected": caption,
            "errors": [], "missing": [], "edit_rate": 0,
            "status": "unchanged", "mode": "correct_v2",
        }

    triples = _extract_triples(caption)
    if not triples:
        print(f"    [{img_id}] triple extraction returned 0 triples — addendum only")
        corrected, n_addendum = _build_spatial_addendum(caption, kb)
        edit_distance = levenshtein_distance(caption, corrected)
        return {
            "caption": caption, "corrected": corrected,
            "errors": [], "missing": [], "edit_rate": edit_distance / max(len(caption), 1),
            "status": "modified" if corrected != caption else "unchanged",
            "mode": "correct_v2", "n_triples": 0, "n_addendum": n_addendum,
        }

    spatial_facts  = kb.get("spatial_facts", [])
    geo_contras    = _check_spatial_contradictions(caption, spatial_facts)
    geo_set        = {c.lower() for c in geo_contras}

    errors       = []
    all_checks   = []

    for t in triples:
        subj = t["subject"].strip()
        rel  = t["relation"].strip()
        obj  = t["object"].strip()
        typ  = t["type"].upper()
        claim_str = f"{subj} {rel} {obj}"

        if typ == "SPATIAL":
            kb_triples = _parse_spatial_facts(spatial_facts)
            verdict, confidence, reason = "UNKNOWN", "LOW", "no geometry available"
            for kb_s, kb_r, kb_o in kb_triples:
                if _entity_matches(subj, kb_s) and _entity_matches(obj, kb_o):
                    if kb_r.lower() in _spatial_synonyms(rel):
                        verdict, confidence, reason = "CORRECT", "HIGH", "geometry confirms"
                        break
                    opp = SPATIAL_OPPOSITES.get(rel)
                    if opp and kb_r == opp:
                        verdict, confidence, reason = "INCORRECT", "HIGH", f"geometry shows {kb_r}"
                        break
            if any(claim_str.lower() in g or
                   (subj.lower() in g and rel.lower() in g) for g in geo_set):
                verdict, confidence, reason = "INCORRECT", "HIGH", "deterministic geometry contradiction"

            if verdict == "INCORRECT" and confidence == "HIGH":
                vqa_confirm, _, _, _, _ = _verify_action_triple(subj, rel, obj, kb, pil_image, n_questions=2)
                if vqa_confirm is True:
                    verdict    = "CORRECT"
                    confidence = "MEDIUM"
                    reason     = f"geometry said INCORRECT but Maverick VQA confirmed → kept"
                elif vqa_confirm is None:
                    verdict    = "UNKNOWN"
                    confidence = "LOW"
                    reason     = "geometry INCORRECT but Maverick uncertain → skipped"

            if verdict == "UNKNOWN":
                obj_box_check = find_best_bbox_from_kb(obj, kb)
                if obj_box_check is None:
                    exists = _check_entity_exists_vqa(obj, pil_image)
                    if exists is False:
                        verdict    = "INCORRECT"
                        confidence = "HIGH"
                        reason     = f"object '{obj}' absent: not found by GDino and VQA confirms absence — remove this claim"
                    else:
                        vqa_result, _, _, _, _ = _verify_action_triple(subj, rel, obj, kb, pil_image, n_questions=3)
                        if vqa_result is False:
                            verdict    = "INCORRECT"
                            confidence = "MEDIUM"
                            reason     = f"spatial VQA (full-image, no GDino bbox) rejected '{rel}' between '{subj}' and '{obj}'"
                        elif vqa_result is True:
                            verdict    = "CORRECT"
                            confidence = "MEDIUM"
                            reason     = "spatial VQA (full-image, no GDino bbox) confirmed"
                else:
                    vqa_result, _, _, _, _ = _verify_action_triple(subj, rel, obj, kb, pil_image, n_questions=3)
                    if vqa_result is False:
                        correct_spatial = _query_correct_spatial_relation(
                            subj, obj, kb, pil_image
                        )
                        verdict    = "INCORRECT"
                        confidence = "MEDIUM"
                        reason     = (f"spatial VQA rejected; correct relation: '{correct_spatial}'"
                                      if correct_spatial else "spatial VQA fallback rejected")
                    elif vqa_result is True:
                        verdict    = "CORRECT"
                        confidence = "MEDIUM"
                        reason     = "spatial VQA fallback confirmed"

            all_checks.append({"claim": claim_str, "type": typ,
                                "verdict": verdict, "confidence": confidence, "reason": reason})
            if verdict == "INCORRECT" and confidence in ("HIGH", "MEDIUM"):
                errors.append({"claim": claim_str, "subject": subj,
                               "relation": rel, "object": obj,
                               "reason": reason, "confidence": confidence,
                               "type": "SPATIAL"})

        else:  # ACTION or ATTRIBUTE
            geo_family = _classify_action_family(rel)
            geo_prereq = None
            if geo_family:
                s_box = find_best_bbox_from_kb(subj, kb)
                o_box = find_best_bbox_from_kb(obj, kb)
                if s_box and o_box:
                    kp = None
                    family_spec = ACTION_GEOMETRY_TAXONOMY.get(geo_family, {})
                    if family_spec.get("needs_keypoints"):
                        _PERSON_WORDS = {"person", "man", "woman", "boy", "girl",
                                         "child", "kid", "baby", "player", "rider",
                                         "worker", "people", "dog", "cat", "horse"}
                        subj_lower = _core_noun(subj)
                        obj_lower  = _core_noun(obj)
                        if subj_lower in _PERSON_WORDS:
                            kp = _get_person_keypoints(pil_image, s_box)
                        elif obj_lower in _PERSON_WORDS:
                            kp = _get_person_keypoints(pil_image, o_box)
                            s_box, o_box = o_box, s_box

                    geo_prereq = _check_action_geometry(
                        geo_family, s_box, o_box, keypoints=kp
                    )
                    if geo_prereq is False:
                        tier = "Tier2-keypoint" if kp else "Tier1-bbox"
                        print(f"    [{img_id}] geometry flag ({geo_family}/{tier}): '{claim_str}' — confirming with VQA")

            if cross_captions and _consensus_confirms_triple(subj, rel, obj, cross_captions):
                verdict, confidence, reason = "CORRECT", "HIGH", "cross-captioner consensus"
                all_checks.append({"claim": claim_str, "type": typ,
                                    "verdict": verdict, "confidence": confidence, "reason": reason})
                continue

            verified, v_yes, v_no, v_total, v_contrastive_no = _verify_action_triple(subj, rel, obj, kb, pil_image, n_questions=3)
            unanimous_no = (verified is False and v_no == v_total and v_total >= 2)
            std_no = (v_no - 1) if v_contrastive_no else v_no
            contrastive_high = v_contrastive_no and std_no >= 1
            print(f"    [{img_id}] {typ} triple ({subj!r},{rel!r},{obj!r}) "
                  f"geo={geo_prereq} vqa={verified} votes={v_yes}Y/{v_no}N/{v_total}T")

            if verified is True:
                verdict, confidence, reason = "CORRECT", "MEDIUM", "crop VQA confirmed"
            elif verified is False:
                if geo_prereq is False:
                    tier = "Tier2-keypoint" if kp else "Tier1-bbox"
                    verdict    = "INCORRECT"
                    confidence = "HIGH"
                    reason     = f"geometry ({geo_family}/{tier}) + VQA both FAILED"
                elif unanimous_no:
                    verdict    = "INCORRECT"
                    confidence = "HIGH"
                    reason     = f"unanimous VQA rejection ({v_no}/{v_total} NO)"
                elif contrastive_high:
                    verdict    = "INCORRECT"
                    confidence = "HIGH"
                    reason     = f"Maverick contrastive NO + {std_no} standard NO ({v_no}/{v_total} total)"
                else:
                    verdict    = "INCORRECT"
                    confidence = "MEDIUM"
                    reason     = f"split VQA rejection ({v_no}/{v_total} NO, no contrastive confirm)"
                if confidence in ("HIGH", "MEDIUM"):
                    errors.append({"claim": claim_str, "subject": subj,
                               "relation": rel, "object": obj,
                               "reason": reason, "confidence": confidence,
                               "type": typ})
            else:
                if geo_prereq is False:
                    verdict    = "UNKNOWN"
                    confidence = "LOW"
                    reason     = "geometry flagged but VQA inconclusive — skipping"
                else:
                    verdict, confidence, reason = "UNKNOWN", "LOW", "could not verify (missing bbox or uncertain)"
            all_checks.append({"claim": claim_str, "type": typ,
                                "verdict": verdict, "confidence": confidence, "reason": reason})

    corrected = caption
    applied   = []

    if errors:
        error_lines = []
        for i, err in enumerate(errors, 1):
            subj, rel, obj_ = err["subject"], err["relation"], err["object"]
            reason = err["reason"]
            err_type = err.get("type", "ACTION")

            if err_type == "SPATIAL" and "absence" in reason:
                import re as _re2
                _am = _re2.search(r"object '([^']+)' absent", reason)
                _absent = _am.group(1) if _am else subj
                _claim_str = err.get("claim", f"{subj} {rel} {obj_}")
                guidance = (
                    f"'{_absent}' does NOT exist in this image. "
                    f"Find the sentence that expresses '{_claim_str}' and COMPLETELY "
                    f"DELETE that one sentence. Do not touch any other sentence. "
                    f"Do NOT rephrase or keep any version of the deleted sentence."
                )
            elif err_type == "SPATIAL":
                _claim_str = err.get("claim", f"{subj} {rel} {obj_}")
                _correct_rel = _extract_correct_rel_from_reason(reason)
                if not _correct_rel and pil_image is not None:
                    _correct_rel = _query_correct_spatial_relation(subj, obj_, kb, pil_image)
                if _correct_rel and _correct_rel.strip() != rel.strip():
                    guidance = (
                        f"The spatial relation '{rel}' in '{_claim_str}' is WRONG "
                        f"(deterministic bbox geometry). "
                        f"Replace ONLY the word/phrase '{rel}' with '{_correct_rel}'. "
                        f"Keep '{subj}' and '{obj_}' and all other text unchanged."
                    )
                else:
                    guidance = (
                        f"The spatial claim '{_claim_str}' is definitively WRONG "
                        f"(bbox geometry contradicts it) and the correct relation is "
                        f"unclear. COMPLETELY DELETE the sentence containing "
                        f"'{_claim_str}'. Do not touch any other sentence."
                    )
            else:
                if err.get("confidence") in ("HIGH", "MEDIUM"):
                    obj_box_exists = find_best_bbox_from_kb(obj_, kb) is not None
                    if not obj_box_exists and pil_image is not None:
                        obj_exists_vqa = _check_entity_exists_vqa(obj_, pil_image)
                    else:
                        obj_exists_vqa = True if obj_box_exists else None
                    obj_absent = (not obj_box_exists and obj_exists_vqa is False)

                    if obj_absent:
                        _claim_str = err.get("claim", f"{subj} {rel} {obj_}")
                        guidance = (
                            f"'{obj_}' does NOT exist in this image. "
                            f"Find the sentence that expresses '{_claim_str}' and COMPLETELY "
                            f"DELETE that one sentence. Do not touch any other sentence. "
                            f"Do NOT rephrase or keep any version of the deleted sentence."
                        )
                    else:
                        _claim_str = err.get("claim", f"{subj} {rel} {obj_}")
                        _correct_rel = _query_correct_action_relation(
                            subj, rel, obj_, kb, pil_image
                        )
                        if _correct_rel:
                            guidance = (
                                f"The relation '{rel}' in '{_claim_str}' is DEFINITELY "
                                f"WRONG (VQA HIGH confidence). "
                                f"Replace ONLY the word/phrase '{rel}' with '{_correct_rel}'. "
                                f"Keep '{subj}' and '{obj_}' and all other text unchanged. "
                                f"Do NOT add or remove any other words."
                            )
                        else:
                            _core_s = _core_noun(subj)
                            _core_o = _core_noun(obj_)
                            _sentences = [s.strip() for s in re.split(r'[.!?]+', caption) if s.strip()]
                            _s_in_sentences = sum(1 for s in _sentences if _core_s in s.lower())
                            _o_in_sentences = sum(1 for s in _sentences if _core_o in s.lower())
                            _is_standalone = (_s_in_sentences <= 1 or _o_in_sentences <= 1)
                            if _is_standalone:
                                guidance = (
                                    f"The claim '{_claim_str}' is DEFINITELY WRONG. "
                                    f"COMPLETELY DELETE the sentence containing "
                                    f"'{_claim_str}'. Do not touch any other sentence."
                                )
                            else:
                                guidance = (
                                    f"The relation '{rel}' in '{_claim_str}' is DEFINITELY "
                                    f"WRONG (VQA HIGH confidence) but the correct relation "
                                    f"is unclear. Replace ONLY '{rel}' with 'near' to remove "
                                    f"the false claim while keeping '{subj}' and '{obj_}'."
                                )
                else:
                    guidance = (
                        f"The '{rel}' relationship between '{subj}' and '{obj_}' appears "
                        f"incorrect (VQA confidence MEDIUM — {reason}). "
                        f"Soften or correct only the '{rel}' word — keep both '{subj}' "
                        f"and '{obj_}' in the sentence."
                    )
            error_lines.append(f'{i}. "{subj} {rel} {obj_}" — {guidance}')

        error_list_str = "\n".join(error_lines)
        prompt = BATCH_CORRECT_PROMPT.format(caption=caption, error_list=error_list_str)

        raw = llm_call(
            [{"role": "user", "content": prompt}],
            max_tokens=int(len(caption.split()) * 2.5 + 50),
        )

        if raw:
            candidate = raw.strip().strip('"').strip("'")
            orig_len  = len(caption)
            cand_len  = len(candidate)
            ratio     = cand_len / max(orig_len, 1)

            has_garble       = _has_garble(candidate)
            has_contradiction = False

            is_too_short = len(candidate.split()) < 5
            is_long_caption = len(caption.split()) >= 30
            too_compressed = is_long_caption and ratio < 0.70
            if ratio <= 1.30 and candidate != caption and not has_garble and not has_contradiction and not is_too_short and not too_compressed:
                corrected = candidate
                applied = [{"method": "batch_llm", "n_errors": len(errors),
                             "errors": [e["claim"] for e in errors]}]
                print(f"    [v2] batch correction: {len(errors)} errors fixed "
                      f"(len {orig_len}→{cand_len}, ratio={ratio:.2f})")
            else:
                print(f"    [v2] batch correction REJECTED "
                      f"(ratio={ratio:.2f}, compressed={too_compressed}, garble={has_garble}, same={candidate==caption})")
                high_errors = [e for e in errors if e.get("confidence") in ("HIGH", "MEDIUM")]
                if high_errors:
                    delete_claims = [e["claim"] for e in high_errors]

                    delete_prompt = (
                        f"Caption: \"{corrected}\"\n\n"
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

                    deletion_result = llm_call(
                        [{"role": "user", "content": delete_prompt}],
                        max_tokens=int(len(corrected.split()) * 2)
                    )

                    if deletion_result:
                        deletion_result = deletion_result.strip().strip('"').strip("'")
                        if deletion_result and deletion_result != corrected and len(deletion_result.split()) >= 3:
                            corrected = deletion_result
                            applied.append({"method": "fallback_deletion", "n_errors": len(high_errors)})
                            print(f"    [v2] fallback: applied deletion for {len(high_errors)} error(s)")

    if corrected != caption:
        new_triples = _extract_triples(corrected)
        introduced_errors = []
        for nt in new_triples:
            ns, nr, no = nt["subject"].strip(), nt["relation"].strip(), nt["object"].strip()
            ntyp = nt["type"].upper()
            if ntyp == "SPATIAL":
                kb_triples = _parse_spatial_facts(kb.get("spatial_facts", []))
                for kb_s, kb_r, kb_o in kb_triples:
                    if _entity_matches(ns, kb_s) and _entity_matches(no, kb_o):
                        opp = SPATIAL_OPPOSITES.get(nr.lower())
                        if opp and kb_r.lower() == opp:
                            introduced_errors.append(f"{ns} {nr} {no} (KB says {kb_r})")
                            break

        if introduced_errors:
            print(f"    [{img_id}] POST-CHECK FAILED: correction introduced new errors: "
                  f"{introduced_errors} → reverting to original")
            corrected = caption
            applied = []

    corrected, n_addendum = _add_missing_fact_addendum(corrected, kb)
    if n_addendum:
        print(f"    [{img_id}] addendum: +{n_addendum} missing fact(s) appended")

    edit_distance = levenshtein_distance(caption, corrected)
    edit_rate_val = edit_distance / max(len(caption), len(corrected), 1)
    return {
        "caption":     caption,
        "corrected":   corrected,
        "errors":      errors,
        "all_checks":  all_checks,
        "applied":     applied,
        "missing":     [],
        "edit_rate":   edit_rate_val,
        "n_triples":   len(triples),
        "n_addendum":  n_addendum,
        "status":      "modified" if corrected != caption else "unchanged",
        "mode":        "correct_v2",
    }


def enrich_caption_v3(img_id: str, caption: str, kb: dict, pil_image=None,
                     cross_captions: dict = None) -> dict:
    """
    Plug-and-play RelCheck enrichment/correction.
    Strategy auto-selected from caption word count.

    Args:
        img_id: image identifier
        caption: original caption text
        kb: visual KB dict with hard_facts, spatial_facts, visual_description, detections
        pil_image: PIL Image (required for correction mode)
        cross_captions: dict of {captioner_name: caption_text} for consensus pre-filter

    Returns:
        dict with 'corrected' caption and metadata
    """
    word_count = len(caption.split())
    if word_count < SHORT_CAPTION_THRESHOLD:
        return _enrich_short_caption(img_id, caption, kb)
    else:
        return _correct_long_caption_v2(img_id, caption, kb, pil_image, cross_captions)
