"""
RelCheck v2 — Evaluation Metrics
==================================
R-POPE LLM-judge evaluation and summary statistics.

R-POPE (LLM-judge) measures caption quality directly by asking an LLM
to answer yes/no questions using ONLY the caption text (no image).
Compares answers against R-Bench ground truth to score accuracy.
"""

from __future__ import annotations

from .api import llm_call
from .config import RPOPE_PROMPT_TMPL


# ── R-POPE LLM-judge ─────────────────────────────────────────────────

def rpope_judge(caption: str, question: str) -> str | None:
    """Ask LLM whether the caption supports the given yes/no question.

    Uses Llama-3.3-70B as a text-only judge: the LLM reads the caption
    and answers the question without seeing the image. This directly
    measures caption quality because the LLM can only use the caption.

    Args:
        caption: The image caption to evaluate
        question: R-Bench yes/no question (e.g. "Is the dog on the couch?")

    Returns:
        'yes', 'no', or None (if response is ambiguous/missing)
    """
    prompt = (
        RPOPE_PROMPT_TMPL
        .replace("CAPTION_PLACEHOLDER", caption)
        .replace("QUESTION_PLACEHOLDER", question)
    )
    resp = llm_call([{"role": "user", "content": prompt}], max_tokens=5)
    if not resp:
        return None
    r = resp.strip().lower()
    if "yes" in r and "no" not in r:
        return "yes"
    if "no" in r:
        return "no"
    return None


# ── Summary statistics ────────────────────────────────────────────────

def compute_rpope_stats(
    results: list[dict],
) -> dict:
    """Compute R-POPE accuracy and breakdown from a list of result dicts.

    Each result dict should have:
        - 'gt_answer': str ('yes' or 'no')
        - 'orig_answer': str | None (LLM answer on original caption)
        - 'corr_answer': str | None (LLM answer on corrected caption)
        - 'rel_type': str ('SPATIAL', 'ACTION', 'ATTRIBUTE')

    Returns:
        Dict with overall accuracy, per-type breakdown, and change counts.
    """
    total = 0
    orig_correct = 0
    corr_correct = 0
    improved = 0
    regressed = 0
    per_type: dict[str, dict[str, int]] = {}

    for r in results:
        gt = r.get("gt_answer")
        orig = r.get("orig_answer")
        corr = r.get("corr_answer")
        rel_type = r.get("rel_type", "UNKNOWN")

        if gt is None or orig is None or corr is None:
            continue

        total += 1
        o_match = orig == gt
        c_match = corr == gt

        if o_match:
            orig_correct += 1
        if c_match:
            corr_correct += 1
        if not o_match and c_match:
            improved += 1
        if o_match and not c_match:
            regressed += 1

        # Per-type tracking
        if rel_type not in per_type:
            per_type[rel_type] = {
                "total": 0, "orig_correct": 0, "corr_correct": 0,
                "improved": 0, "regressed": 0,
            }
        pt = per_type[rel_type]
        pt["total"] += 1
        if o_match:
            pt["orig_correct"] += 1
        if c_match:
            pt["corr_correct"] += 1
        if not o_match and c_match:
            pt["improved"] += 1
        if o_match and not c_match:
            pt["regressed"] += 1

    return {
        "total": total,
        "orig_accuracy": orig_correct / max(total, 1),
        "corr_accuracy": corr_correct / max(total, 1),
        "orig_correct": orig_correct,
        "corr_correct": corr_correct,
        "improved": improved,
        "regressed": regressed,
        "net_change": improved - regressed,
        "per_type": per_type,
    }


def format_rpope_summary(stats: dict) -> str:
    """Format R-POPE stats dict into a human-readable summary string."""
    lines = [
        "=" * 60,
        "R-POPE LLM-Judge Results",
        "=" * 60,
        f"  Total questions:    {stats['total']}",
        f"  Original accuracy:  {stats['orig_correct']}/{stats['total']} "
        f"({stats['orig_accuracy']:.1%})",
        f"  Corrected accuracy: {stats['corr_correct']}/{stats['total']} "
        f"({stats['corr_accuracy']:.1%})",
        f"  Delta:              {stats['corr_correct'] - stats['orig_correct']:+d} "
        f"({stats['corr_accuracy'] - stats['orig_accuracy']:+.1%})",
        f"  Improved:           {stats['improved']}",
        f"  Regressed:          {stats['regressed']}",
        f"  Net:                {stats['net_change']:+d}",
        "",
        "  Per relation type:",
    ]

    for rtype, pt in sorted(stats.get("per_type", {}).items()):
        orig_acc = pt["orig_correct"] / max(pt["total"], 1)
        corr_acc = pt["corr_correct"] / max(pt["total"], 1)
        delta = pt["corr_correct"] - pt["orig_correct"]
        lines.append(
            f"    {rtype:12s}  n={pt['total']:3d}  "
            f"orig={orig_acc:.1%}  corr={corr_acc:.1%}  delta={delta:+d}"
        )

    lines.append("=" * 60)
    return "\n".join(lines)


# ── Synthetic test evaluation ────────────────────────────────────────

def run_synthetic_rpope(
    injected_data: dict,
    corrected_captions: dict,
    rbench_questions: dict | None = None,
    verbose: bool = True,
) -> dict:
    """Run full R-POPE LLM-judge evaluation on synthetic hallucination test.

    Evaluates three caption versions per image:
      - original (clean caption, before injection)
      - corrupted (after hallucination injection)
      - corrected (after RelCheck correction)

    Args:
        injected_data: {img_id: {original_caption, corrupted_caption,
                        injected_question, rel_type, ...}}
        corrected_captions: {img_id: corrected_caption_str}
        rbench_questions: optional {img_id: [{question, answer}]} for
                          supplemental evaluation on non-injected questions
        verbose: print per-image results

    Returns:
        Dict with 'main' results (injected questions) and 'supplemental'
        results (other R-Bench questions for same images).
    """
    corrected_img_ids = {
        img_id for img_id, inj in injected_data.items()
        if img_id in corrected_captions
        and corrected_captions[img_id] != inj["corrupted_caption"]
    }
    if verbose:
        print(f"Images modified by RelCheck: {len(corrected_img_ids)}/{len(injected_data)}")
        print("\nRunning R-POPE LLM-Judge on injected questions...\n")

    total = orig_no = corr_no = fixed_no = 0
    true_drops = recoveries = 0
    per_type: dict[str, dict] = {}
    rows: list[tuple] = []

    for img_id, inj in injected_data.items():
        if img_id not in corrected_captions:
            continue

        question = inj["injected_question"]
        rel_type = inj.get("rel_type", "?")
        orig_cap = inj["original_caption"]
        corr_cap = inj["corrupted_caption"]
        fixed_cap = corrected_captions[img_id]

        orig_ans = rpope_judge(orig_cap, question)
        corr_ans = rpope_judge(corr_cap, question)
        fixed_ans = rpope_judge(fixed_cap, question)

        total += 1
        if orig_ans == "no":
            orig_no += 1
        if corr_ans == "no":
            corr_no += 1
        if fixed_ans == "no":
            fixed_no += 1

        injected_ok = orig_ans == "no" and corr_ans == "yes"
        if injected_ok:
            true_drops += 1
            if fixed_ans == "no":
                recoveries += 1

        pt = per_type.setdefault(
            rel_type,
            {"total": 0, "orig_no": 0, "corr_no": 0, "fixed_no": 0,
             "drops": 0, "rec": 0},
        )
        pt["total"] += 1
        if orig_ans == "no":
            pt["orig_no"] += 1
        if corr_ans == "no":
            pt["corr_no"] += 1
        if fixed_ans == "no":
            pt["fixed_no"] += 1
        if injected_ok:
            pt["drops"] += 1
        if injected_ok and fixed_ans == "no":
            pt["rec"] += 1

        det = " <-- DETECTED" if injected_ok else ""
        rec = " + RECOVERED" if (injected_ok and fixed_ans == "no") else ""
        rows.append((img_id, rel_type, orig_ans, corr_ans, fixed_ans, det + rec))
        if verbose:
            print(f"  [{img_id}] [{rel_type}]{det}{rec}")
            print(f"    Q: {question}")
            print(f"    orig={orig_ans}  corr={corr_ans}  fixed={fixed_ans}")

    # ── Main summary ──
    main_results = {
        "total": total, "orig_no": orig_no, "corr_no": corr_no,
        "fixed_no": fixed_no, "true_drops": true_drops,
        "recoveries": recoveries, "per_type": per_type, "rows": rows,
    }

    if total == 0:
        print("WARNING: nothing evaluated — check injected_data and corrected_captions")
    elif verbose:
        _print_main_summary(main_results)

    # ── Supplemental evaluation ──
    supp_results: dict = {}
    if rbench_questions is not None and total > 0:
        supp_results = _run_supplemental_rpope(
            injected_data, corrected_captions, corrected_img_ids,
            rbench_questions, verbose,
        )

    return {"main": main_results, "supplemental": supp_results}


def _print_main_summary(r: dict) -> None:
    """Print main R-POPE summary for synthetic test."""
    total = r["total"]
    acc_orig = 100 * r["orig_no"] / total
    acc_corr = 100 * r["corr_no"] / total
    acc_fixed = 100 * r["fixed_no"] / total
    drop = acc_corr - acc_orig
    recovery = acc_fixed - acc_corr

    print(f"\n{'=' * 60}")
    print("R-POPE Results (GT=no for all injected questions)")
    print("=" * 60)
    print(f"  Images evaluated:   {total}")
    td = r["true_drops"]
    rec = r["recoveries"]
    print(f"  Injection detected: {td}/{total}  ({100 * td / total:.0f}%)")
    print(f"  Recoveries:         {rec}/{max(td, 1)}  "
          f"({100 * rec / max(td, 1):.0f}% of detectable)")
    print()
    print("  Accuracy (fraction answering no = correct):")
    print(f"    Original:   {acc_orig:.1f}%  (caption without hallucination)")
    print(f"    Corrupted:  {acc_corr:.1f}%  (after injection,   delta={drop:+.1f}%)")
    print(f"    Corrected:  {acc_fixed:.1f}%  (after RelCheck,   delta={recovery:+.1f}%)")
    print()
    print("  By relation type:")
    for rtype, c in sorted(r["per_type"].items()):
        t = c["total"]
        print(f"    {rtype:10} n={t}  orig={c['orig_no']}/{t}  "
              f"corr={c['corr_no']}/{t}  fixed={c['fixed_no']}/{t}  "
              f"drops={c['drops']}  rec={c['rec']}")


def _run_supplemental_rpope(
    injected_data: dict,
    corrected_captions: dict,
    corrected_img_ids: set,
    rbench_questions: dict,
    verbose: bool = True,
) -> dict:
    """Evaluate non-injected R-Bench questions on same images."""
    if verbose:
        print(f"\n{'=' * 60}")
        print("Supplemental: all other R-Bench questions (same images)")
        print("=" * 60)

    rb2_total = rb2_orig = rb2_corr = rb2_fixed = 0
    rbc_total = rbc_orig = rbc_corr = rbc_fixed = 0

    for img_id, inj in injected_data.items():
        if img_id not in corrected_captions:
            continue
        injected_q = inj["injected_question"]
        is_corrected = img_id in corrected_img_ids
        for qa in rbench_questions.get(img_id, []):
            if qa["question"] == injected_q:
                continue
            gt = qa["answer"].lower().strip()
            if gt not in ("yes", "no"):
                continue
            oa = rpope_judge(inj["original_caption"], qa["question"])
            ca = rpope_judge(inj["corrupted_caption"], qa["question"])
            fa = rpope_judge(corrected_captions[img_id], qa["question"])
            rb2_total += 1
            if oa == gt:
                rb2_orig += 1
            if ca == gt:
                rb2_corr += 1
            if fa == gt:
                rb2_fixed += 1
            if is_corrected:
                rbc_total += 1
                if oa == gt:
                    rbc_orig += 1
                if ca == gt:
                    rbc_corr += 1
                if fa == gt:
                    rbc_fixed += 1

    supp = {
        "all": {"total": rb2_total, "orig": rb2_orig,
                "corr": rb2_corr, "fixed": rb2_fixed},
        "corrected_only": {"total": rbc_total, "orig": rbc_orig,
                           "corr": rbc_corr, "fixed": rbc_fixed},
    }

    if verbose:
        if rb2_total > 0:
            s_orig = 100 * rb2_orig / rb2_total
            s_corr = 100 * rb2_corr / rb2_total
            s_fixed = 100 * rb2_fixed / rb2_total
            print(f"  All images ({rb2_total} other questions):")
            print(f"    Original:  {s_orig:.1f}%")
            print(f"    Corrupted: {s_corr:.1f}%  (delta={s_corr - s_orig:+.1f}%)")
            print(f"    Corrected: {s_fixed:.1f}%  (delta={s_fixed - s_corr:+.1f}%)")
            if rbc_total > 0:
                bc_orig = 100 * rbc_orig / rbc_total
                bc_corr = 100 * rbc_corr / rbc_total
                bc_fixed = 100 * rbc_fixed / rbc_total
                n_c = len(corrected_img_ids & set(injected_data))
                print(f"  Corrected images only (n={n_c}, {rbc_total} questions):")
                print(f"    Original:  {bc_orig:.1f}%")
                print(f"    Corrupted: {bc_corr:.1f}%  (delta={bc_corr - bc_orig:+.1f}%)")
                print(f"    Corrected: {bc_fixed:.1f}%  (delta={bc_fixed - bc_corr:+.1f}%)")
        else:
            print("  No supplemental questions found.")

    return supp
