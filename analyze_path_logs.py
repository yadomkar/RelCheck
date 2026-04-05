"""Analyze path_logs.json — extract every useful insight."""
import json, sys
from collections import Counter

with open("relcheck_v2/path_logs_addendum_false.json") as f:
    logs = json.load(f)

print(f"{'='*70}")
print(f"  PATH LOGS DEEP ANALYSIS — {len(logs)} images")
print(f"{'='*70}\n")

for img_id, rec in logs.items():
    snaps = rec.get("caption_snapshots", [])
    spatial_v = rec.get("spatial_verifications", [])
    action_v = rec.get("action_verifications", [])
    guidance = rec.get("guidance", [])
    batch = rec.get("batch_eval", {})
    kb = rec.get("kb_content", {})
    extraction = rec.get("extraction", {})
    post_v = rec.get("post_verification", {})
    spatial_add = rec.get("spatial_addendum", {})
    missing_add = rec.get("missing_fact_addendum", {})

    input_cap = snaps[0]["text"] if snaps else "?"
    final_cap = snaps[-1]["text"] if snaps else "?"
    changed = input_cap != final_cap

    print(f"\n{'─'*70}")
    print(f"IMAGE: {img_id}")
    print(f"{'─'*70}")
    print(f"  INPUT:   {input_cap}")
    print(f"  FINAL:   {final_cap}")
    print(f"  CHANGED: {changed}")
    print(f"  KB: {len(kb.get('hard_facts',[]))} hard, {len(kb.get('spatial_facts',[]))} spatial, "
          f"desc={len(kb.get('visual_description',''))}ch, {kb.get('n_detections',0)} dets")
    print(f"  TRIPLES: {extraction.get('total_triples',0)} "
          f"(S:{extraction.get('spatial_count',0)} A:{extraction.get('action_count',0)} "
          f"AT:{extraction.get('attribute_count',0)})")

    # Spatial verifications
    if spatial_v:
        print(f"\n  SPATIAL VERIFICATIONS ({len(spatial_v)}):")
        for sv in spatial_v:
            v = sv.get("verdict", "?")
            c = sv.get("confidence", "?")
            claim = sv.get("claim", "?")
            kb_syn = "KB-SYN" if sv.get("kb_synonym_match") else ""
            kb_opp = "KB-OPP" if sv.get("kb_opposite_match") else ""
            geo = "GEO-CONTRA" if sv.get("geo_contradiction_fired") else ""
            vqa_override = ""
            if sv.get("vqa_cross_check_override") is True:
                vqa_override = "VQA-OVERRIDE→CORRECT"
            elif sv.get("vqa_cross_check_override") is False:
                vqa_override = "VQA-OVERRIDE→UNKNOWN"
            ent_exist = ""
            if sv.get("entity_existence_triggered"):
                ent_exist = f"ENT-EXIST={sv.get('entity_existence_result')}"
            bbox_s = "bbox-S" if sv.get("kb_bbox_found_subject") else "no-bbox-S"
            bbox_o = "bbox-O" if sv.get("kb_bbox_found_object") else "no-bbox-O"
            kb_correct = "KB-CORRECT-REL" if sv.get("kb_provided_correct_rel") else ""
            flags = " ".join(f for f in [kb_syn, kb_opp, geo, vqa_override, ent_exist, bbox_s, bbox_o, kb_correct] if f)
            print(f"    [{v:>9}/{c:>6}] {claim}")
            if flags:
                print(f"                    {flags}")

    # Action verifications
    if action_v:
        print(f"\n  ACTION/ATTRIBUTE VERIFICATIONS ({len(action_v)}):")
        for av in action_v:
            v = av.get("verdict", "?")
            c = av.get("confidence", "?")
            claim = av.get("claim", "?")
            rel_type = av.get("rel_type", "?")
            geo_fam = av.get("action_geo_family") or "none"
            geo_res = av.get("geo_prereq_result")
            kp = "KP-YES" if av.get("keypoints_loaded") else "KP-NO"
            consensus = "CONSENSUS" if av.get("consensus_confirmed") else ""
            vqa_cat = av.get("vqa_decision_category", "?")
            votes = f"{av.get('vqa_yes_votes',0)}Y/{av.get('vqa_no_votes',0)}N/{av.get('vqa_total',0)}T"
            contrastive = "CONTRASTIVE-NO" if av.get("vqa_contrastive_no") else ""
            bbox_s = "bbox-S" if av.get("kb_bbox_found_subject") else "no-bbox-S"
            bbox_o = "bbox-O" if av.get("kb_bbox_found_object") else "no-bbox-O"
            crop = "CROP-VQA" if av.get("used_crop_vqa") else "FULL-IMG"
            geo_possible = "GEO-OK" if av.get("geo_check_possible") else "GEO-NA"
            flags = " ".join(f for f in [f"geo={geo_fam}/{geo_res}", kp, consensus, 
                                          f"vqa={vqa_cat}({votes})", contrastive,
                                          bbox_s, bbox_o, crop, geo_possible] if f)
            print(f"    [{v:>9}/{c:>6}] [{rel_type:>9}] {claim}")
            print(f"                    {flags}")

    # Guidance
    if guidance:
        print(f"\n  GUIDANCE ({len(guidance)}):")
        for g in guidance:
            gtype = g.get("guidance_type", "?")
            claim = g.get("claim", "?")
            found = g.get("correct_rel_found", False)
            src = g.get("correct_rel_source") or "none"
            print(f"    {gtype:<18} {claim}  (correct_rel={found}, src={src})")

    # Batch eval
    if batch.get("length_ratio", 0) > 0:
        print(f"\n  BATCH EVAL: ratio={batch.get('length_ratio'):.3f} "
              f"garble={batch.get('garble_detected')} "
              f"short={batch.get('too_short')} "
              f"compressed={batch.get('too_compressed')} "
              f"accepted={batch.get('accepted')}")

    # Caption snapshots
    if len(snaps) > 2:
        print(f"\n  CAPTION EVOLUTION ({len(snaps)} stages):")
        for s in snaps:
            stage = s.get("stage", "?")
            text = s.get("text", "")
            acc = f" [accepted={s['accepted']}]" if "accepted" in s else ""
            print(f"    {stage:<35} {text}{acc}")

    # Post verification
    if post_v.get("reverted"):
        print(f"\n  ⚠ POST-VERIFICATION REVERTED: {post_v.get('n_new_contradictions')} new contradictions")

    # Addendum
    if spatial_add.get("n_facts_added", 0) > 0:
        print(f"\n  SPATIAL ADDENDUM: +{spatial_add['n_facts_added']} facts "
              f"(available={spatial_add.get('kb_spatial_facts_available')}, "
              f"expressed={spatial_add.get('n_already_expressed')}, "
              f"novel={spatial_add.get('n_novel')})")

print(f"\n{'='*70}")
print("  AGGREGATE INSIGHTS")
print(f"{'='*70}")

# Aggregate stats
all_spatial = [sv for rec in logs.values() for sv in rec.get("spatial_verifications", [])]
all_action = [av for rec in logs.values() for av in rec.get("action_verifications", [])]
all_guidance = [g for rec in logs.values() for g in rec.get("guidance", [])]

# Evidence source breakdown for INCORRECT verdicts
incorrect_sources = Counter()
for sv in all_spatial:
    if sv.get("verdict") == "INCORRECT":
        if sv.get("kb_opposite_match"): incorrect_sources["KB opposite match"] += 1
        elif sv.get("geo_contradiction_fired"): incorrect_sources["Geo contradiction"] += 1
        elif sv.get("entity_existence_result") is False: incorrect_sources["Entity absent (GDino+VQA)"] += 1
        else: incorrect_sources["VQA fallback"] += 1

for av in all_action:
    if av.get("verdict") == "INCORRECT":
        cat = av.get("vqa_decision_category", "?")
        geo = av.get("geo_prereq_result")
        if geo is False:
            incorrect_sources[f"Geo+VQA ({cat})"] += 1
        else:
            incorrect_sources[f"VQA-only ({cat})"] += 1

print(f"\n  INCORRECT verdict sources:")
for src, cnt in incorrect_sources.most_common():
    print(f"    {src:<35} {cnt:>3}")

# Correct rel sources
correct_sources = Counter()
for g in all_guidance:
    src = g.get("correct_rel_source") or "none"
    correct_sources[src] += 1
print(f"\n  Correct relation sources (for corrections):")
for src, cnt in correct_sources.most_common():
    print(f"    {src:<35} {cnt:>3}")

# Geometry family effectiveness
geo_families = Counter()
geo_results = {"True": 0, "False": 0, "None": 0}
for av in all_action:
    fam = av.get("action_geo_family")
    if fam:
        geo_families[fam] += 1
    res = av.get("geo_prereq_result")
    geo_results[str(res)] += 1

print(f"\n  Geometry family hits:")
for fam, cnt in geo_families.most_common():
    print(f"    {fam:<25} {cnt:>3}")
print(f"\n  Geometry results: confirmed={geo_results['True']} violated={geo_results['False']} skipped={geo_results['None']}")

# Keypoint stats
kp_loaded = sum(1 for av in all_action if av.get("keypoints_loaded"))
kp_total = len(all_action)
print(f"  Keypoints loaded: {kp_loaded}/{kp_total}")

# Images where caption got worse (more text added than removed)
print(f"\n  Per-image caption changes:")
for img_id, rec in logs.items():
    snaps = rec.get("caption_snapshots", [])
    if len(snaps) >= 2:
        inp = snaps[0]["text"]
        fin = snaps[-1]["text"]
        delta = len(fin.split()) - len(inp.split())
        n_errors = len([sv for sv in rec.get("spatial_verifications", []) + rec.get("action_verifications", []) if sv.get("verdict") == "INCORRECT"])
        print(f"    [{img_id[:16]}] words: {len(inp.split()):>3}→{len(fin.split()):>3} (Δ{delta:>+3}) errors={n_errors}")
