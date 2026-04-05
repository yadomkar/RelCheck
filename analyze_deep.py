"""Deep analysis of path_logs — extract every possible insight."""
import json
from collections import Counter, defaultdict

with open("relcheck_v2/path_logs_addendum_false.json") as f:
    logs = json.load(f)

print(f"{'='*80}")
print(f"  DEEP PATH LOGS ANALYSIS — {len(logs)} images")
print(f"{'='*80}\n")

# ═══════════════════════════════════════════════════════════════════════
# 1. FULL CAPTION TEXT AT EVERY STAGE
# ═══════════════════════════════════════════════════════════════════════
print("=" * 80)
print("  1. FULL CAPTION SNAPSHOTS PER IMAGE")
print("=" * 80)

for img_id, rec in logs.items():
    snaps = rec.get("caption_snapshots", [])
    print(f"\n{'─'*80}")
    print(f"IMAGE: {img_id}")
    for s in snaps:
        stage = s.get("stage", "?")
        text = s.get("text", "")
        acc = f" [accepted={s['accepted']}]" if "accepted" in s else ""
        print(f"\n  [{stage}]{acc}")
        print(f"  {text}")

# ═══════════════════════════════════════════════════════════════════════
# 2. KB CONTENT PER IMAGE (what data was available)
# ═══════════════════════════════════════════════════════════════════════
print(f"\n\n{'='*80}")
print("  2. KB CONTENT PER IMAGE")
print("=" * 80)

for img_id, rec in logs.items():
    kb = rec.get("kb_content", {})
    hard = kb.get("hard_facts", [])
    spatial = kb.get("spatial_facts", [])
    vis = kb.get("visual_description", "")
    n_det = kb.get("n_detections", 0)
    print(f"\n{'─'*80}")
    print(f"IMAGE: {img_id}")
    print(f"  Detections: {n_det}")
    print(f"  Hard facts ({len(hard)}):")
    for f_ in hard:
        print(f"    - {f_}")
    print(f"  Spatial facts ({len(spatial)}):")
    for f_ in spatial[:10]:  # first 10
        print(f"    - {f_}")
    if len(spatial) > 10:
        print(f"    ... and {len(spatial)-10} more")
    print(f"  Visual description ({len(vis)} chars):")
    print(f"    {vis[:300]}{'...' if len(vis) > 300 else ''}")

# ═══════════════════════════════════════════════════════════════════════
# 3. EVERY VERIFICATION DECISION WITH FULL CONTEXT
# ═══════════════════════════════════════════════════════════════════════
print(f"\n\n{'='*80}")
print("  3. ALL VERIFICATION DECISIONS")
print("=" * 80)

all_verdicts = []
for img_id, rec in logs.items():
    for sv in rec.get("spatial_verifications", []):
        all_verdicts.append({**sv, "img_id": img_id, "type": "SPATIAL"})
    for av in rec.get("action_verifications", []):
        all_verdicts.append({**av, "img_id": img_id, "type": av.get("rel_type", "ACTION")})

# Group by verdict
for verdict_type in ["CORRECT", "INCORRECT", "UNKNOWN"]:
    group = [v for v in all_verdicts if v.get("verdict") == verdict_type]
    print(f"\n{'─'*80}")
    print(f"  {verdict_type} verdicts ({len(group)}):")
    print(f"{'─'*80}")
    for v in group:
        claim = v.get("claim", "?")
        conf = v.get("confidence", "?")
        img = v.get("img_id", "?")[:16]
        vtype = v.get("type", "?")
        
        # Evidence chain
        evidence = []
        if v.get("kb_synonym_match"): evidence.append("KB-synonym")
        if v.get("kb_opposite_match"): evidence.append("KB-opposite")
        if v.get("geo_contradiction_fired"): evidence.append("geo-contradiction")
        if v.get("vqa_cross_check_override") is True: evidence.append("VQA-override→CORRECT")
        if v.get("vqa_cross_check_override") is False: evidence.append("VQA-override→UNKNOWN")
        if v.get("entity_existence_triggered"):
            evidence.append(f"entity-exist={v.get('entity_existence_result')}")
        if v.get("kb_provided_correct_rel"): evidence.append("KB-has-correct-rel")
        
        # Action-specific
        geo_fam = v.get("action_geo_family")
        geo_res = v.get("geo_prereq_result")
        if geo_fam: evidence.append(f"geo-family={geo_fam}/{geo_res}")
        if v.get("keypoints_loaded"): evidence.append("KEYPOINTS")
        if v.get("consensus_confirmed"): evidence.append("CONSENSUS")
        vqa_cat = v.get("vqa_decision_category")
        if vqa_cat: evidence.append(f"vqa={vqa_cat}")
        votes = ""
        if "vqa_yes_votes" in v:
            votes = f"{v['vqa_yes_votes']}Y/{v['vqa_no_votes']}N/{v['vqa_total']}T"
            evidence.append(votes)
        if v.get("vqa_contrastive_no"): evidence.append("CONTRASTIVE-NO")
        
        # Bbox
        bbox_s = v.get("kb_bbox_found_subject")
        bbox_o = v.get("kb_bbox_found_object")
        if bbox_s is not None:
            evidence.append(f"bbox:{'S' if bbox_s else '_'}{'O' if bbox_o else '_'}")
        
        ev_str = " | ".join(evidence) if evidence else "no-evidence"
        print(f"  [{img}] [{vtype:>9}/{conf:>6}] {claim}")
        print(f"    Evidence: {ev_str}")

# ═══════════════════════════════════════════════════════════════════════
# 4. GUIDANCE → CORRECTION MAPPING
# ═══════════════════════════════════════════════════════════════════════
print(f"\n\n{'='*80}")
print("  4. ALL GUIDANCE INSTRUCTIONS")
print("=" * 80)

for img_id, rec in logs.items():
    guidance = rec.get("guidance", [])
    if not guidance:
        continue
    print(f"\n  [{img_id[:16]}] ({len(guidance)} corrections):")
    for g in guidance:
        gtype = g.get("guidance_type", "?")
        claim = g.get("claim", "?")
        found = g.get("correct_rel_found", False)
        src = g.get("correct_rel_source") or "none"
        print(f"    {gtype:<18} \"{claim}\"")
        print(f"      correct_rel_found={found}, source={src}")

# ═══════════════════════════════════════════════════════════════════════
# 5. BATCH EVAL DETAILS
# ═══════════════════════════════════════════════════════════════════════
print(f"\n\n{'='*80}")
print("  5. BATCH CORRECTION EVALUATION")
print("=" * 80)

for img_id, rec in logs.items():
    batch = rec.get("batch_eval", {})
    if batch.get("length_ratio", 0) == 0:
        continue
    n_errors = len(rec.get("guidance", []))
    print(f"  [{img_id[:16]}] errors={n_errors} ratio={batch.get('length_ratio',0):.3f} "
          f"garble={batch.get('garble_detected')} short={batch.get('too_short')} "
          f"compressed={batch.get('too_compressed')} accepted={batch.get('accepted')}")

# ═══════════════════════════════════════════════════════════════════════
# 6. AGGREGATE STATISTICS
# ═══════════════════════════════════════════════════════════════════════
print(f"\n\n{'='*80}")
print("  6. AGGREGATE STATISTICS")
print("=" * 80)

# Verdict breakdown by type
print("\n  Verdicts by relation type:")
type_verdicts = defaultdict(lambda: Counter())
for v in all_verdicts:
    type_verdicts[v["type"]][v.get("verdict", "?")] += 1
for vtype, counts in sorted(type_verdicts.items()):
    print(f"    {vtype:<12} C={counts.get('CORRECT',0)} I={counts.get('INCORRECT',0)} U={counts.get('UNKNOWN',0)}")

# Evidence source for INCORRECT
print("\n  How INCORRECT verdicts were reached:")
incorrect_evidence = Counter()
for v in all_verdicts:
    if v.get("verdict") != "INCORRECT":
        continue
    if v.get("kb_opposite_match"):
        incorrect_evidence["KB opposite match (deterministic)"] += 1
    elif v.get("geo_contradiction_fired"):
        incorrect_evidence["Geometry contradiction (deterministic)"] += 1
    elif v.get("entity_existence_result") is False:
        incorrect_evidence["Entity absent (GDino+VQA)"] += 1
    elif v.get("geo_prereq_result") is False:
        incorrect_evidence[f"Geometry violated + VQA ({v.get('vqa_decision_category','?')})"] += 1
    elif v.get("vqa_decision_category"):
        incorrect_evidence[f"VQA only ({v.get('vqa_decision_category','?')})"] += 1
    else:
        incorrect_evidence["VQA spatial fallback"] += 1
for src, cnt in incorrect_evidence.most_common():
    print(f"    {src:<50} {cnt:>3}")

# Correct relation sources
print("\n  Where correct relations came from:")
rel_sources = Counter()
for rec in logs.values():
    for g in rec.get("guidance", []):
        src = g.get("correct_rel_source") or "none (DELETE)"
        rel_sources[src] += 1
for src, cnt in rel_sources.most_common():
    print(f"    {src:<30} {cnt:>3}")

# Geometry effectiveness
print("\n  Geometry system effectiveness:")
geo_triples = [v for v in all_verdicts if v.get("action_geo_family")]
geo_with_result = [v for v in geo_triples if v.get("geo_prereq_result") is not None]
print(f"    Action triples with geo family: {len(geo_triples)}")
print(f"    Geo check actually ran (both bboxes): {len(geo_with_result)}")
for v in geo_with_result:
    claim = v.get("claim", "?")
    fam = v.get("action_geo_family")
    res = v.get("geo_prereq_result")
    verdict = v.get("verdict")
    kp = "KP" if v.get("keypoints_loaded") else "no-KP"
    agree = "AGREE" if (res is True and verdict == "CORRECT") or (res is False and verdict == "INCORRECT") else "DISAGREE"
    print(f"    [{fam}/{res}] [{verdict}] {kp} {agree}: {claim}")

# Keypoint details
print("\n  ViTPose keypoint details:")
kp_triples = [v for v in all_verdicts if v.get("keypoints_loaded")]
print(f"    Keypoints loaded: {len(kp_triples)} times")
for v in kp_triples:
    claim = v.get("claim", "?")
    fam = v.get("action_geo_family")
    res = v.get("geo_prereq_result")
    verdict = v.get("verdict")
    print(f"    [{fam}/{res}→{verdict}] {claim}")

# Bbox coverage detail
print("\n  Bbox coverage (per entity):")
bbox_found = bbox_total = 0
bbox_miss_entities = Counter()
for v in all_verdicts:
    if v.get("kb_bbox_found_subject") is not None:
        bbox_total += 1
        if v["kb_bbox_found_subject"]:
            bbox_found += 1
        else:
            # Extract subject from claim
            claim = v.get("claim", "")
            subj = claim.split()[0] if claim else "?"
            bbox_miss_entities[f"subj:{subj}"] += 1
    if v.get("kb_bbox_found_object") is not None:
        bbox_total += 1
        if v["kb_bbox_found_object"]:
            bbox_found += 1
        else:
            claim = v.get("claim", "")
            parts = claim.split()
            obj = parts[-1] if parts else "?"
            bbox_miss_entities[f"obj:{obj}"] += 1
print(f"    Found: {bbox_found}/{bbox_total} ({100*bbox_found/max(bbox_total,1):.1f}%)")
print(f"    Most missed entities:")
for ent, cnt in bbox_miss_entities.most_common(15):
    print(f"      {ent:<30} {cnt:>3}")

print(f"\n{'='*80}")
print("  DONE")
print(f"{'='*80}")
