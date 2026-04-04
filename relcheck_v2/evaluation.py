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
