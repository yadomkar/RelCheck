"""
RelCheck v2 — COCO Evaluation Metrics
=======================================
Reusable metric functions for the COCO controlled evaluation pipeline.
Includes R-CHAIR (relational CHAIR), hallucination removal rate,
collateral damage, NLG metrics, and ablation table construction.

All functions are pure (no side effects, no model loading) and operate
on plain dicts/strings for easy testing and composability.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from ._logging import log


# ════════════════════════════════════════════════════════════════════════════
# R-CHAIR — Relational CHAIR metric
# ════════════════════════════════════════════════════════════════════════════


def compute_r_chair(
    caption: str,
    coco_annotations: list[dict],
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    """Compute R-CHAIR_i and R-CHAIR_s components for one caption.

    Extracts spatial triples from the caption, looks up ground-truth
    bounding boxes from COCO annotations, and checks geometric
    consistency via ``spatial_verdict``.

    Args:
        caption: Caption text to evaluate.
        coco_annotations: List of dicts with ``category_name`` and
            ``bbox`` ([x, y, w, h] in pixels) keys.
        image_width: Image width in pixels.
        image_height: Image height in pixels.

    Returns:
        Dict with keys ``n_spatial_triples``, ``n_verifiable``,
        ``n_hallucinated``, ``r_chair_i``, ``has_hallucination``.
    """
    from .correction.surgical._extraction import extract_triples
    from .correction._utils import entity_matches
    from .spatial import spatial_verdict
    from .types import RelationType

    triples = extract_triples(caption)
    spatial = [t for t in triples if t.rel_type == RelationType.SPATIAL]

    bbox_lookup: dict[str, list[list[float]]] = {}
    for ann in coco_annotations:
        cat = ann["category_name"]
        bx, by, bw, bh = ann["bbox"]
        norm = [bx / image_width, by / image_height,
                (bx + bw) / image_width, (by + bh) / image_height]
        bbox_lookup.setdefault(cat, []).append(norm)

    n_verifiable = 0
    n_hallucinated = 0

    for t in spatial:
        subj_bbox = _find_bbox(t.subject, bbox_lookup, entity_matches)
        obj_bbox = _find_bbox(t.object, bbox_lookup, entity_matches)
        if subj_bbox is None or obj_bbox is None:
            continue

        verdict = spatial_verdict(subj_bbox, obj_bbox, t.relation)
        if verdict is None:
            continue  # ambiguous — exclude from count

        n_verifiable += 1
        if verdict is False:
            n_hallucinated += 1

    return {
        "n_spatial_triples": len(spatial),
        "n_verifiable": n_verifiable,
        "n_hallucinated": n_hallucinated,
        "r_chair_i": n_hallucinated / max(n_verifiable, 1),
        "has_hallucination": n_hallucinated > 0,
    }


def _find_bbox(
    entity: str,
    bbox_lookup: dict[str, list[list[float]]],
    match_fn: Any,
) -> list[float] | None:
    """Find the first matching bbox for *entity* in the lookup."""
    for cat, bboxes in bbox_lookup.items():
        if match_fn(entity, cat):
            return bboxes[0]
    return None


def aggregate_r_chair(results: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate per-caption R-CHAIR dicts into corpus-level metrics.

    Args:
        results: List of dicts from :func:`compute_r_chair`.

    Returns:
        Dict with ``r_chair_i`` (mean) and ``r_chair_s`` (fraction of
        captions with at least one hallucinated relation).
    """
    if not results:
        return {"r_chair_i": 0.0, "r_chair_s": 0.0, "n": 0}
    n = len(results)
    return {
        "r_chair_i": sum(r["r_chair_i"] for r in results) / n,
        "r_chair_s": sum(1 for r in results if r["has_hallucination"]) / n,
        "n": n,
    }


# ════════════════════════════════════════════════════════════════════════════
# Hallucination Removal & Collateral Damage
# ════════════════════════════════════════════════════════════════════════════


def hallucination_removed(
    corrupted: str,
    corrected: str,
    injected_statement: str,
) -> bool:
    """Check whether the injected hallucination was removed.

    A simple containment check: the injected statement (stripped of
    trailing period, case-insensitive) should no longer appear in the
    corrected caption.

    Args:
        corrupted: Caption with injected hallucination.
        corrected: Caption after correction pipeline.
        injected_statement: The exact statement that was injected.

    Returns:
        True if the hallucination was successfully removed.
    """
    needle = injected_statement.lower().strip().rstrip(".")
    return needle not in corrected.lower()


def collateral_damage(original: str, corrected: str) -> float:
    """Fraction of original sentences altered or removed in corrected.

    Splits on period boundaries and checks containment. A sentence
    counts as "damaged" if it no longer appears (case-insensitive)
    in the corrected caption.

    Args:
        original: Original (pre-corruption) caption.
        corrected: Caption after correction pipeline.

    Returns:
        Float in [0, 1]. 0 means no original content was lost.
    """
    orig_sents = [s.strip() for s in original.split(".") if s.strip()]
    if not orig_sents:
        return 0.0
    damaged = sum(
        1 for s in orig_sents
        if s.strip().lower() not in corrected.lower()
    )
    return damaged / len(orig_sents)


# ════════════════════════════════════════════════════════════════════════════
# NLG Metrics (BLEU-4, METEOR)
# ════════════════════════════════════════════════════════════════════════════


def compute_bleu4(reference: str, hypothesis: str) -> float:
    """Sentence-level BLEU-4 with smoothing.

    Args:
        reference: Ground-truth caption.
        hypothesis: Generated/corrected caption.

    Returns:
        BLEU-4 score in [0, 1].
    """
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

    ref_tok = nltk.word_tokenize(reference.lower())
    hyp_tok = nltk.word_tokenize(hypothesis.lower())
    if not hyp_tok:
        return 0.0
    return sentence_bleu(
        [ref_tok], hyp_tok,
        smoothing_function=SmoothingFunction().method1,
    )


def compute_meteor(reference: str, hypothesis: str) -> float:
    """METEOR score between reference and hypothesis.

    Args:
        reference: Ground-truth caption.
        hypothesis: Generated/corrected caption.

    Returns:
        METEOR score in [0, 1].
    """
    import nltk
    from nltk.translate.meteor_score import meteor_score as _meteor

    ref_tok = nltk.word_tokenize(reference.lower())
    hyp_tok = nltk.word_tokenize(hypothesis.lower())
    if not hyp_tok:
        return 0.0
    return _meteor([ref_tok], hyp_tok)


# ════════════════════════════════════════════════════════════════════════════
# Ablation Table Builder
# ════════════════════════════════════════════════════════════════════════════

# Column order for the ablation table
ABLATION_METRICS: list[str] = [
    "Removal Rate", "R-CHAIR_i", "Collateral", "BLEU-4",
    "METEOR", "CLIPScore", "Judge Win%",
]


def build_ablation_table(
    metrics_per_image: list[dict[str, Any]],
    r_chair_by_run: dict[str, list[dict[str, Any]]],
    judge_by_run: dict[str, dict[str, str]],
    injection_types: list[str],
    run_names: tuple[str, ...] = ("baseline", "reltr"),
) -> list[dict[str, Any]]:
    """Build the ablation comparison table.

    Produces one row per injection type plus an "Overall" row. Each row
    contains per-run metric values and delta columns.

    Args:
        metrics_per_image: List of per-image metric dicts. Each must have
            ``injection_type``, ``img_id``, and per-run keys like
            ``removal_baseline``, ``bleu4_reltr``, etc.
        r_chair_by_run: Mapping of run name → list of per-image R-CHAIR
            dicts (same order as *metrics_per_image*).
        judge_by_run: Mapping of run name → {img_id: verdict_str}.
        injection_types: Ordered list of injection type labels.
        run_names: Tuple of run name strings (default baseline/reltr).

    Returns:
        List of row dicts suitable for ``pandas.DataFrame``.
    """
    rows: list[dict[str, Any]] = []

    for inj_type in injection_types + ["Overall"]:
        if inj_type == "Overall":
            subset = metrics_per_image
            indices = list(range(len(metrics_per_image)))
        else:
            indices = [
                i for i, m in enumerate(metrics_per_image)
                if m["injection_type"] == inj_type
            ]
            subset = [metrics_per_image[i] for i in indices]

        if not subset:
            continue

        row: dict[str, Any] = {"Injection Type": inj_type, "N": len(subset)}

        for run in run_names:
            row[f"Removal Rate ({run})"] = _mean_key(subset, f"removal_{run}")
            row[f"Collateral ({run})"] = _mean_key(subset, f"collateral_{run}")
            row[f"BLEU-4 ({run})"] = _mean_key(subset, f"bleu4_{run}")
            row[f"METEOR ({run})"] = _mean_key(subset, f"meteor_{run}")
            row[f"CLIPScore ({run})"] = _mean_key(subset, f"clipscore_{run}")

            # R-CHAIR
            rc_list = r_chair_by_run.get(run, [])
            rc_sub = [rc_list[i] for i in indices if i < len(rc_list)]
            row[f"R-CHAIR_i ({run})"] = (
                sum(r["r_chair_i"] for r in rc_sub) / len(rc_sub)
                if rc_sub else 0.0
            )

            # Judge win rate
            jr = judge_by_run.get(run, {})
            img_ids = [m["img_id"] for m in subset]
            wins = sum(1 for iid in img_ids if jr.get(iid) == "corrected_wins")
            row[f"Judge Win% ({run})"] = wins / max(len(img_ids), 1)

        # Deltas (reltr − baseline)
        if len(run_names) >= 2:
            a, b = run_names[0], run_names[1]
            row["Δ Removal"] = row.get(f"Removal Rate ({b})", 0) - row.get(f"Removal Rate ({a})", 0)
            row["Δ R-CHAIR_i"] = row.get(f"R-CHAIR_i ({b})", 0) - row.get(f"R-CHAIR_i ({a})", 0)

        rows.append(row)

    return rows


def _mean_key(items: list[dict], key: str) -> float:
    """Mean of *key* across *items*, treating missing/bool values correctly."""
    vals = []
    for m in items:
        v = m.get(key)
        if v is None:
            continue
        if isinstance(v, bool):
            vals.append(float(v))
        else:
            vals.append(float(v))
    return sum(vals) / max(len(vals), 1)
