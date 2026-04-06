"""
RelCheck v2 — Correction Path Metrics Collector
==================================================
Structured per-image decision-path instrumentation for the correction
pipeline.  Records every branch (dispatch, enrichment guards, per-triple
verification cascades, guidance selection, batch acceptance, addendum
filters, post-verification reverts) without altering pipeline behaviour.

Usage::

    from relcheck_v2.correction._metrics import MetricsCollector

    mc = MetricsCollector()
    result = enrich_caption_v3(img_id, caption, kb, pil_image=img, metrics=mc)
    mc.save("/content/drive/MyDrive/path_logs.json")
    mc.print_summary()
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from typing import Any

from .._logging import log

# ════════════════════════════════════════════════════════════════════════════
# STRING CONSTANTS — stage names
# ════════════════════════════════════════════════════════════════════════════

STAGE_INPUT: str = "input"
STAGE_BATCH_CANDIDATE: str = "batch_correction_candidate"
STAGE_FALLBACK_DELETION: str = "fallback_deletion"
STAGE_POST_VERIFY_REVERT: str = "post_verification_revert"
STAGE_SPATIAL_ADDENDUM: str = "after_spatial_addendum"
STAGE_MISSING_FACT_ADDENDUM: str = "after_missing_fact_addendum"
STAGE_FINAL: str = "final"

# ════════════════════════════════════════════════════════════════════════════
# STRING CONSTANTS — guidance types
# ════════════════════════════════════════════════════════════════════════════

GUIDANCE_REPLACE_WORD: str = "REPLACE_WORD"
GUIDANCE_DELETE_SENTENCE: str = "DELETE_SENTENCE"
GUIDANCE_SOFTEN: str = "SOFTEN"

# ════════════════════════════════════════════════════════════════════════════
# STRING CONSTANTS — rejection reasons
# ════════════════════════════════════════════════════════════════════════════

REJECT_TOO_MANY_WORDS: str = "too_many_words"
REJECT_LOW_SURVIVAL: str = "low_survival"
REJECT_NGRAM_REPETITION: str = "ngram_repetition"
REJECT_NEW_BIGRAM_REPETITION: str = "new_bigram_repetition"

# ════════════════════════════════════════════════════════════════════════════
# STRING CONSTANTS — evidence sources
# ════════════════════════════════════════════════════════════════════════════

SOURCE_SPATIAL_KB: str = "spatial_kb"
SOURCE_KB_VISUAL_DESC: str = "kb_visual_desc"
SOURCE_VLM_QUERY: str = "vlm_query"
SOURCE_ACTION_3STAGE: str = "action_3stage"



# ════════════════════════════════════════════════════════════════════════════
# METRICS COLLECTOR
# ════════════════════════════════════════════════════════════════════════════


def _empty_path_log(img_id: str) -> dict[str, Any]:
    """Return a fresh PathLog dict with all top-level keys at empty defaults.

    Args:
        img_id: Image identifier to embed in the record.

    Returns:
        Dict matching the PathLog schema with zeroed/empty values.
    """
    return {
        "img_id": img_id,
        "caption_snapshots": [],
        "kb_content": {
            "hard_facts": [],
            "spatial_facts": [],
            "visual_description": "",
            "n_detections": 0,
        },
        "dispatch": {
            "word_count": 0,
            "threshold": 0,
            "mode": "",
        },
        "enrichment": {},
        "extraction": {
            "total_triples": 0,
            "spatial_count": 0,
            "action_count": 0,
            "attribute_count": 0,
            "addendum_only": False,
        },
        "spatial_verifications": [],
        "action_verifications": [],
        "guidance": [],
        "batch_eval": {
            "length_ratio": 0.0,
            "garble_detected": False,
            "too_short": False,
            "too_compressed": False,
            "accepted": False,
        },
        "fallback_deletion": {
            "used": False,
            "n_sentences_deleted": 0,
        },
        "post_verification": {
            "n_new_contradictions": 0,
            "reverted": False,
        },
        "spatial_addendum": {
            "n_facts_added": 0,
            "kb_spatial_facts_available": 0,
            "n_already_expressed": 0,
            "n_novel": 0,
        },
        "missing_fact_addendum": {
            "llm_returned_facts": False,
            "n_words_added": 0,
            "accepted": False,
            "rejection_reason": None,
            "visual_description_input": "",
            "visual_description_too_short": False,
        },
        "scene_graph": {
            "n_triples": 0,
            "n_evidence_hits": 0,
            "enabled": False,
        },
    }


class MetricsCollector:
    """Accumulates per-image path logs and computes aggregate summaries.

    Each image gets a ``PathLog`` dict (see design doc for schema) that
    records every decision point taken during correction.  The collector
    provides JSON serialization and an aggregate summary suitable for
    Colab output cells.

    Attributes:
        _logs: Mapping of ``img_id`` → ``PathLog`` dict.
    """

    # ── Construction ────────────────────────────────────────────────────

    def __init__(self) -> None:
        self._logs: dict[str, dict[str, Any]] = {}

    # ── Image lifecycle ─────────────────────────────────────────────────

    def init_image(self, img_id: str) -> None:
        """Initialize a fresh PathLog for *img_id*.

        If *img_id* already exists the previous record is overwritten and
        a warning is logged (handles Colab re-runs gracefully).

        Args:
            img_id: Unique image identifier.
        """
        if img_id in self._logs:
            log.warning("MetricsCollector: overwriting existing log for '%s'", img_id)
        self._logs[img_id] = _empty_path_log(img_id)

    def _ensure_image(self, img_id: str) -> None:
        """Auto-initialize *img_id* if not yet present (defensive guard).

        Args:
            img_id: Image identifier to check.
        """
        if img_id not in self._logs:
            log.debug("MetricsCollector: auto-initializing log for '%s'", img_id)
            self._logs[img_id] = _empty_path_log(img_id)

    # ── Per-image recording methods ─────────────────────────────────────

    def record_dispatch(
        self,
        img_id: str,
        word_count: int,
        threshold: int,
        mode: str,
    ) -> None:
        """Record the dispatch decision for *img_id*.

        Args:
            img_id: Image identifier.
            word_count: Number of words in the caption.
            threshold: ``SHORT_CAPTION_THRESHOLD`` value used.
            mode: ``"enrichment"`` or ``"surgical"``.
        """
        self._ensure_image(img_id)
        self._logs[img_id]["dispatch"] = {
            "word_count": word_count,
            "threshold": threshold,
            "mode": mode,
        }

    def record_caption_snapshot(
        self,
        img_id: str,
        stage: str,
        text: str,
        accepted: bool | None = None,
    ) -> None:
        """Append a caption snapshot at a pipeline stage.

        Args:
            img_id: Image identifier.
            stage: One of the ``STAGE_*`` constants.
            text: Caption text at this stage.
            accepted: Whether the candidate was accepted (batch eval only).
        """
        self._ensure_image(img_id)
        entry: dict[str, Any] = {"stage": stage, "text": text}
        if accepted is not None:
            entry["accepted"] = accepted
        self._logs[img_id]["caption_snapshots"].append(entry)

    def record_kb_content(
        self,
        img_id: str,
        hard_facts: list[str],
        spatial_facts: list[str],
        visual_description: str,
        n_detections: int,
    ) -> None:
        """Record the full KB content for *img_id*.

        Args:
            img_id: Image identifier.
            hard_facts: List of hard-fact strings from the KB.
            spatial_facts: List of spatial-fact strings from the KB.
            visual_description: VLM-generated visual description.
            n_detections: Number of GroundingDINO detections.
        """
        self._ensure_image(img_id)
        self._logs[img_id]["kb_content"] = {
            "hard_facts": list(hard_facts),
            "spatial_facts": list(spatial_facts),
            "visual_description": visual_description,
            "n_detections": n_detections,
        }

    def record_enrichment(self, img_id: str, **fields: Any) -> None:
        """Record enrichment-mode decisions for *img_id*.

        Accepts arbitrary keyword arguments matching the ``enrichment``
        sub-dict schema (``llm_analysis_success``, ``json_parse_success``,
        ``n_errors_found``, ``n_missing_found``, etc.).

        Args:
            img_id: Image identifier.
            **fields: Key-value pairs for the enrichment record.
        """
        self._ensure_image(img_id)
        self._logs[img_id]["enrichment"] = dict(fields)

    def record_extraction(
        self,
        img_id: str,
        total: int,
        spatial: int,
        action: int,
        attribute: int,
        addendum_only: bool,
    ) -> None:
        """Record triple extraction results for *img_id*.

        Args:
            img_id: Image identifier.
            total: Total number of extracted triples.
            spatial: Count of SPATIAL triples.
            action: Count of ACTION triples.
            attribute: Count of ATTRIBUTE triples.
            addendum_only: Whether the pipeline fell through to addendum-only.
        """
        self._ensure_image(img_id)
        self._logs[img_id]["extraction"] = {
            "total_triples": total,
            "spatial_count": spatial,
            "action_count": action,
            "attribute_count": attribute,
            "addendum_only": addendum_only,
        }

    def record_spatial_verification(self, img_id: str, entry: dict[str, Any]) -> None:
        """Append one spatial verification record for *img_id*.

        Args:
            img_id: Image identifier.
            entry: Dict matching the ``spatial_verifications`` item schema.
        """
        self._ensure_image(img_id)
        self._logs[img_id]["spatial_verifications"].append(entry)

    def record_action_verification(self, img_id: str, entry: dict[str, Any]) -> None:
        """Append one action/attribute verification record for *img_id*.

        Args:
            img_id: Image identifier.
            entry: Dict matching the ``action_verifications`` item schema.
        """
        self._ensure_image(img_id)
        self._logs[img_id]["action_verifications"].append(entry)

    def record_guidance(self, img_id: str, entry: dict[str, Any]) -> None:
        """Append one guidance record for *img_id*.

        Args:
            img_id: Image identifier.
            entry: Dict with ``guidance_type``, ``correct_rel_found``,
                   ``correct_rel_source``.
        """
        self._ensure_image(img_id)
        self._logs[img_id]["guidance"].append(entry)

    def record_batch_eval(self, img_id: str, **fields: Any) -> None:
        """Record batch correction evaluation for *img_id*.

        Args:
            img_id: Image identifier.
            **fields: Key-value pairs (``length_ratio``, ``garble_detected``,
                      ``too_short``, ``too_compressed``, ``accepted``).
        """
        self._ensure_image(img_id)
        self._logs[img_id]["batch_eval"] = dict(fields)

    def record_fallback_deletion(
        self,
        img_id: str,
        used: bool,
        n_deleted: int,
    ) -> None:
        """Record fallback deletion outcome for *img_id*.

        Args:
            img_id: Image identifier.
            used: Whether fallback deletion was triggered.
            n_deleted: Number of sentences deleted.
        """
        self._ensure_image(img_id)
        self._logs[img_id]["fallback_deletion"] = {
            "used": used,
            "n_sentences_deleted": n_deleted,
        }

    def record_post_verification(
        self,
        img_id: str,
        n_contradictions: int,
        reverted: bool,
    ) -> None:
        """Record post-verification outcome for *img_id*.

        Args:
            img_id: Image identifier.
            n_contradictions: Number of new contradictions detected.
            reverted: Whether the correction was reverted.
        """
        self._ensure_image(img_id)
        self._logs[img_id]["post_verification"] = {
            "n_new_contradictions": n_contradictions,
            "reverted": reverted,
        }

    def record_spatial_addendum(
        self,
        img_id: str,
        n_facts_added: int,
        kb_spatial_facts_available: int = 0,
        n_already_expressed: int = 0,
        n_novel: int = 0,
    ) -> None:
        """Record spatial addendum outcome for *img_id*.

        Args:
            img_id: Image identifier.
            n_facts_added: Number of spatial facts appended.
            kb_spatial_facts_available: Total KB spatial facts available.
            n_already_expressed: Facts skipped (already in caption).
            n_novel: Facts that were actually novel additions.
        """
        self._ensure_image(img_id)
        self._logs[img_id]["spatial_addendum"] = {
            "n_facts_added": n_facts_added,
            "kb_spatial_facts_available": kb_spatial_facts_available,
            "n_already_expressed": n_already_expressed,
            "n_novel": n_novel,
        }

    def record_missing_fact_addendum(self, img_id: str, **fields: Any) -> None:
        """Record missing-fact addendum outcome for *img_id*.

        Args:
            img_id: Image identifier.
            **fields: Key-value pairs (``llm_returned_facts``,
                      ``n_words_added``, ``accepted``, ``rejection_reason``,
                      ``visual_description_input``,
                      ``visual_description_too_short``).
        """
        self._ensure_image(img_id)
        self._logs[img_id]["missing_fact_addendum"] = dict(fields)

    def record_scene_graph(
        self,
        img_id: str,
        n_triples: int,
        n_evidence_hits: int,
        enabled: bool,
    ) -> None:
        """Record scene graph stats for *img_id*.

        Args:
            img_id: Image identifier.
            n_triples: Number of RelTR triples produced.
            n_evidence_hits: Number of NLI evidence items from scene graph.
            enabled: Whether ENABLE_RELTR was True.
        """
        self._ensure_image(img_id)
        self._logs[img_id]["scene_graph"] = {
            "n_triples": n_triples,
            "n_evidence_hits": n_evidence_hits,
            "enabled": enabled,
        }

    # ── Serialization ───────────────────────────────────────────────────

    def to_json(self) -> dict[str, dict[str, Any]]:
        """Return all PathLog records as a JSON-compatible dict.

        Returns:
            Dict keyed by ``img_id``, each value a PathLog dict containing
            only JSON-primitive types.
        """
        return self._logs

    def save(self, path: str) -> None:
        """Write all PathLog records to a JSON file.

        Args:
            path: Filesystem path for the output JSON file.

        Raises:
            IOError: If the file cannot be written.
        """
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.to_json(), f, indent=2)
        except OSError as exc:
            raise IOError(f"Failed to write metrics to '{path}': {exc}") from exc

    # ── Aggregation ─────────────────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        """Compute aggregate summary across all recorded images.

        Returns a dict matching the ``AggregateSummary`` schema.  All rate
        values are clamped to ``[0.0, 1.0]``.  Returns a valid zero-state
        dict when no images have been recorded.

        Returns:
            Aggregate summary dict.
        """
        logs = list(self._logs.values())
        n = len(logs)

        if n == 0:
            return self._zero_summary()

        # ── Dispatch counts ─────────────────────────────────────────────
        dispatch_counts: dict[str, int] = {"enrichment": 0, "surgical": 0}
        for rec in logs:
            mode = rec.get("dispatch", {}).get("mode", "")
            if mode in dispatch_counts:
                dispatch_counts[mode] += 1

        # ── Triples per image ──────────────────────────────────────────
        triple_counts = [
            rec.get("extraction", {}).get("total_triples", 0) for rec in logs
        ]
        mean_triples = statistics.mean(triple_counts) if triple_counts else 0.0
        median_triples = statistics.median(triple_counts) if triple_counts else 0.0

        # ── Verdict distribution ────────────────────────────────────────
        verdict_dist: dict[str, dict[str, int]] = {
            "SPATIAL": {"CORRECT": 0, "INCORRECT": 0, "UNKNOWN": 0},
            "ACTION": {"CORRECT": 0, "INCORRECT": 0, "UNKNOWN": 0},
            "ATTRIBUTE": {"CORRECT": 0, "INCORRECT": 0, "UNKNOWN": 0},
        }
        evidence_counter: Counter[str] = Counter()

        for rec in logs:
            for sv in rec.get("spatial_verifications", []):
                verdict = sv.get("verdict", "UNKNOWN")
                if verdict in verdict_dist["SPATIAL"]:
                    verdict_dist["SPATIAL"][verdict] += 1
                evidence_counter[sv.get("evidence_source", "unknown")] += 1

            for av in rec.get("action_verifications", []):
                rel_type = av.get("rel_type", "ACTION")
                verdict = av.get("verdict", "UNKNOWN")
                if rel_type in verdict_dist and verdict in verdict_dist[rel_type]:
                    verdict_dist[rel_type][verdict] += 1
                evidence_counter[av.get("evidence_source", "unknown")] += 1

        top_evidence = [
            {"source": src, "count": cnt}
            for src, cnt in evidence_counter.most_common()
        ]

        # ── Guidance type distribution ──────────────────────────────────
        guidance_dist: dict[str, int] = {
            GUIDANCE_REPLACE_WORD: 0,
            GUIDANCE_DELETE_SENTENCE: 0,
            GUIDANCE_SOFTEN: 0,
        }
        for rec in logs:
            for g in rec.get("guidance", []):
                gtype = g.get("guidance_type", "")
                if gtype in guidance_dist:
                    guidance_dist[gtype] += 1

        # ── Rate computations ───────────────────────────────────────────
        batch_accepted = sum(
            1 for rec in logs if rec.get("batch_eval", {}).get("accepted", False)
        )
        batch_total = sum(
            1 for rec in logs if rec.get("batch_eval", {}).get("length_ratio", 0.0) > 0
        )

        fallback_used = sum(
            1 for rec in logs if rec.get("fallback_deletion", {}).get("used", False)
        )

        post_reverted = sum(
            1 for rec in logs if rec.get("post_verification", {}).get("reverted", False)
        )
        post_total = sum(
            1 for rec in logs
            if rec.get("post_verification", {}).get("n_new_contradictions", 0) > 0
            or rec.get("post_verification", {}).get("reverted", False)
        )

        addendum_accepted = sum(
            1 for rec in logs
            if rec.get("missing_fact_addendum", {}).get("accepted", False)
        )
        addendum_total = sum(
            1 for rec in logs
            if rec.get("missing_fact_addendum", {}).get("llm_returned_facts", False)
        )

        # ── Path effectiveness ──────────────────────────────────────────
        path_effectiveness = self._compute_path_effectiveness(logs)

        # ── KB usage ────────────────────────────────────────────────────
        kb_usage = self._compute_kb_usage(logs)

        # ── Geometry usage ──────────────────────────────────────────────
        geometry_usage = self._compute_geometry_usage(logs)

        # ── NLI usage ───────────────────────────────────────────────────
        nli_usage = self._compute_nli_usage(logs)

        # ── RelTR usage ─────────────────────────────────────────────────
        reltr_usage = self._compute_reltr_usage(logs)

        return {
            "total_images": n,
            "dispatch_counts": dispatch_counts,
            "triples_per_image": {
                "mean": round(mean_triples, 2),
                "median": round(median_triples, 2),
            },
            "verdict_distribution": verdict_dist,
            "top_evidence_sources": top_evidence,
            "guidance_type_distribution": guidance_dist,
            "batch_correction_acceptance_rate": _safe_rate(batch_accepted, batch_total),
            "fallback_deletion_rate": _safe_rate(fallback_used, n),
            "post_verification_revert_rate": _safe_rate(post_reverted, max(post_total, n)),
            "addendum_acceptance_rate": _safe_rate(addendum_accepted, max(addendum_total, 1)),
            "path_effectiveness": path_effectiveness,
            "kb_usage": kb_usage,
            "geometry_usage": geometry_usage,
            "nli_usage": nli_usage,
            "reltr_usage": reltr_usage,
        }

    # ── Summary helpers (private) ───────────────────────────────────────

    @staticmethod
    def _zero_summary() -> dict[str, Any]:
        """Return a valid summary dict for the zero-image case.

        Returns:
            Summary dict with all keys set to zero/empty defaults.
        """
        return {
            "total_images": 0,
            "dispatch_counts": {"enrichment": 0, "surgical": 0},
            "triples_per_image": {"mean": 0.0, "median": 0.0},
            "verdict_distribution": {
                "SPATIAL": {"CORRECT": 0, "INCORRECT": 0, "UNKNOWN": 0},
                "ACTION": {"CORRECT": 0, "INCORRECT": 0, "UNKNOWN": 0},
                "ATTRIBUTE": {"CORRECT": 0, "INCORRECT": 0, "UNKNOWN": 0},
            },
            "top_evidence_sources": [],
            "guidance_type_distribution": {
                GUIDANCE_REPLACE_WORD: 0,
                GUIDANCE_DELETE_SENTENCE: 0,
                GUIDANCE_SOFTEN: 0,
            },
            "batch_correction_acceptance_rate": 0.0,
            "fallback_deletion_rate": 0.0,
            "post_verification_revert_rate": 0.0,
            "addendum_acceptance_rate": 0.0,
            "path_effectiveness": {},
            "kb_usage": {
                "mean_hard_facts": 0.0,
                "mean_spatial_facts": 0.0,
                "mean_visual_desc_len": 0.0,
                "mean_detections": 0.0,
                "spatial_fact_hit_rate": 0.0,
                "bbox_coverage": 0.0,
                "correct_rel_kb_first_rate": 0.0,
                "addendum_novelty_rate": 0.0,
            },
            "geometry_usage": {
                "total_action_triples": 0,
                "geo_check_possible": 0,
                "geo_check_rate": 0.0,
                "keypoints_loaded": 0,
                "keypoints_rate": 0.0,
                "geo_confirmed": 0,
                "geo_violated": 0,
                "family_counts": {},
                "geo_vqa_agreement_rate": 0.0,
            },
            "nli_usage": {
                "total_checks": 0,
                "support_count": 0,
                "contradict_count": 0,
                "neutral_count": 0,
                "vqa_calls_saved": 0,
                "evidence_hit_rate": 0.0,
                "batch_calls_made": 0,
                "entity_existence_flags": 0,
                "contradict_high_geometry": 0,
                "contradict_high_entity": 0,
                "contradict_low_visual": 0,
            },
            "reltr_usage": {
                "total_triples": 0,
                "mean_triples_per_image": 0.0,
                "total_evidence_hits": 0,
                "evidence_hit_rate": 0.0,
            },
        }

    @staticmethod
    def _compute_path_effectiveness(
        logs: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """Group images by dominant evidence source and compute modification rate.

        Args:
            logs: List of PathLog dicts.

        Returns:
            Dict keyed by evidence source, each with ``n_images`` and
            ``fraction_modified``.
        """
        source_images: dict[str, list[bool]] = {}

        for rec in logs:
            # Collect all INCORRECT verdicts and their evidence sources
            incorrect_sources: Counter[str] = Counter()
            for sv in rec.get("spatial_verifications", []):
                if sv.get("verdict") == "INCORRECT":
                    incorrect_sources[sv.get("evidence_source", "unknown")] += 1
            for av in rec.get("action_verifications", []):
                if av.get("verdict") == "INCORRECT":
                    incorrect_sources[av.get("evidence_source", "unknown")] += 1

            if not incorrect_sources:
                continue

            dominant_source = incorrect_sources.most_common(1)[0][0]
            snapshots = rec.get("caption_snapshots", [])
            modified = False
            if len(snapshots) >= 2:
                modified = snapshots[0].get("text", "") != snapshots[-1].get("text", "")

            source_images.setdefault(dominant_source, []).append(modified)

        result: dict[str, dict[str, Any]] = {}
        for source, modified_flags in source_images.items():
            n_imgs = len(modified_flags)
            frac = sum(modified_flags) / max(n_imgs, 1)
            result[source] = {
                "n_images": n_imgs,
                "fraction_modified": _clamp01(frac),
            }
        return result

    @staticmethod
    def _compute_kb_usage(logs: list[dict[str, Any]]) -> dict[str, float]:
        """Compute aggregate KB usage statistics.

        Args:
            logs: List of PathLog dicts.

        Returns:
            Dict with mean KB layer sizes and hit rates.
        """
        n = len(logs)
        if n == 0:
            return {
                "mean_hard_facts": 0.0,
                "mean_spatial_facts": 0.0,
                "mean_visual_desc_len": 0.0,
                "mean_detections": 0.0,
                "spatial_fact_hit_rate": 0.0,
                "bbox_coverage": 0.0,
                "correct_rel_kb_first_rate": 0.0,
                "addendum_novelty_rate": 0.0,
            }

        # KB layer sizes
        hard_counts = [len(r.get("kb_content", {}).get("hard_facts", [])) for r in logs]
        spatial_counts = [len(r.get("kb_content", {}).get("spatial_facts", [])) for r in logs]
        vdesc_lens = [len(r.get("kb_content", {}).get("visual_description", "")) for r in logs]
        det_counts = [r.get("kb_content", {}).get("n_detections", 0) for r in logs]

        # Spatial fact hit rate: fraction of spatial verifications where KB
        # gave a direct answer (synonym or opposite match)
        spatial_total = 0
        spatial_kb_hit = 0
        for rec in logs:
            for sv in rec.get("spatial_verifications", []):
                spatial_total += 1
                if sv.get("kb_synonym_match") or sv.get("kb_opposite_match"):
                    spatial_kb_hit += 1

        # Bbox coverage: fraction of entity lookups that succeeded
        bbox_found = 0
        bbox_total = 0
        for rec in logs:
            for sv in rec.get("spatial_verifications", []):
                bbox_total += 2  # subject + object
                if sv.get("kb_bbox_found_subject"):
                    bbox_found += 1
                if sv.get("kb_bbox_found_object"):
                    bbox_found += 1
            for av in rec.get("action_verifications", []):
                bbox_total += 2
                if av.get("kb_bbox_found_subject"):
                    bbox_found += 1
                if av.get("kb_bbox_found_object"):
                    bbox_found += 1

        # Correct-rel KB-first rate
        guidance_total = 0
        kb_first = 0
        for rec in logs:
            for g in rec.get("guidance", []):
                guidance_total += 1
                src = g.get("correct_rel_source")
                if src in (SOURCE_SPATIAL_KB, SOURCE_KB_VISUAL_DESC):
                    kb_first += 1

        # Addendum novelty rate
        novel_total = 0
        available_total = 0
        for rec in logs:
            sa = rec.get("spatial_addendum", {})
            novel_total += sa.get("n_novel", 0)
            available_total += sa.get("kb_spatial_facts_available", 0)

        return {
            "mean_hard_facts": round(statistics.mean(hard_counts), 2) if hard_counts else 0.0,
            "mean_spatial_facts": round(statistics.mean(spatial_counts), 2) if spatial_counts else 0.0,
            "mean_visual_desc_len": round(statistics.mean(vdesc_lens), 2) if vdesc_lens else 0.0,
            "mean_detections": round(statistics.mean(det_counts), 2) if det_counts else 0.0,
            "spatial_fact_hit_rate": _safe_rate(spatial_kb_hit, spatial_total),
            "bbox_coverage": _safe_rate(bbox_found, bbox_total),
            "correct_rel_kb_first_rate": _safe_rate(kb_first, guidance_total),
            "addendum_novelty_rate": _safe_rate(novel_total, available_total),
        }

    @staticmethod
    def _compute_geometry_usage(
        logs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Compute aggregate geometry and ViTPose usage for action triples.

        Args:
            logs: List of PathLog dicts.

        Returns:
            Dict with geometry check rates, keypoint usage, family breakdown,
            and geometry-VQA agreement rate.
        """
        total_action = 0
        geo_possible = 0
        kp_loaded = 0
        geo_confirmed = 0  # geo_prereq True
        geo_violated = 0   # geo_prereq False
        family_counts: Counter[str] = Counter()
        # Track agreement: when geo has an opinion, does VQA agree?
        geo_vqa_agree = 0
        geo_vqa_total = 0

        for rec in logs:
            for av in rec.get("action_verifications", []):
                total_action += 1
                family = av.get("action_geo_family")
                geo_result = av.get("geo_prereq_result")
                kp = av.get("keypoints_loaded", False)
                verdict = av.get("verdict", "UNKNOWN")

                if av.get("geo_check_possible", False):
                    geo_possible += 1
                if kp:
                    kp_loaded += 1
                if family:
                    family_counts[family] += 1
                if geo_result is True:
                    geo_confirmed += 1
                elif geo_result is False:
                    geo_violated += 1

                # Agreement: geo says True + VQA says CORRECT, or
                #            geo says False + VQA says INCORRECT
                if geo_result is not None:
                    geo_vqa_total += 1
                    if (geo_result is True and verdict == "CORRECT") or \
                       (geo_result is False and verdict == "INCORRECT"):
                        geo_vqa_agree += 1

        return {
            "total_action_triples": total_action,
            "geo_check_possible": geo_possible,
            "geo_check_rate": _safe_rate(geo_possible, total_action),
            "keypoints_loaded": kp_loaded,
            "keypoints_rate": _safe_rate(kp_loaded, total_action),
            "geo_confirmed": geo_confirmed,
            "geo_violated": geo_violated,
            "family_counts": dict(family_counts),
            "geo_vqa_agreement_rate": _safe_rate(geo_vqa_agree, geo_vqa_total),
        }

    @staticmethod
    def _compute_nli_usage(logs: list[dict[str, Any]]) -> dict[str, Any]:
        """Compute aggregate NLI pre-filter usage statistics.

        Args:
            logs: List of PathLog dicts.

        Returns:
            Dict with NLI check counts, verdict breakdown, VQA savings,
            and tiered CONTRADICT counts.
        """
        total_checks = 0
        support_count = 0
        contradict_count = 0
        neutral_count = 0
        vqa_saved = 0
        evidence_hit = 0
        entity_existence_flags = 0
        contradict_high_geometry = 0
        contradict_high_entity = 0
        contradict_low_visual = 0

        for rec in logs:
            for entries in (rec.get("spatial_verifications", []),
                            rec.get("action_verifications", [])):
                for entry in entries:
                    nli_v = entry.get("nli_verdict")
                    if nli_v is None:
                        continue
                    total_checks += 1
                    if nli_v == "SUPPORT":
                        support_count += 1
                    elif nli_v == "CONTRADICT":
                        contradict_count += 1
                    else:
                        neutral_count += 1

                    if entry.get("nli_skipped_vqa"):
                        vqa_saved += 1
                    if entry.get("nli_evidence_count", 0) > 0:
                        evidence_hit += 1
                    if entry.get("nli_evidence_source") == "entity_existence":
                        entity_existence_flags += 1

                    tier = entry.get("nli_contradict_tier")
                    if tier == "high_geometry":
                        contradict_high_geometry += 1
                    elif tier == "high_entity":
                        contradict_high_entity += 1
                    elif tier == "low_visual":
                        contradict_low_visual += 1

        # batch_calls_made: count images that had at least one NLI check
        batch_calls = sum(
            1 for rec in logs
            if any(
                e.get("nli_verdict") is not None
                for entries in (rec.get("spatial_verifications", []),
                                rec.get("action_verifications", []))
                for e in entries
            )
        )

        return {
            "total_checks": total_checks,
            "support_count": support_count,
            "contradict_count": contradict_count,
            "neutral_count": neutral_count,
            "vqa_calls_saved": vqa_saved,
            "evidence_hit_rate": _safe_rate(evidence_hit, total_checks),
            "batch_calls_made": batch_calls,
            "entity_existence_flags": entity_existence_flags,
            "contradict_high_geometry": contradict_high_geometry,
            "contradict_high_entity": contradict_high_entity,
            "contradict_low_visual": contradict_low_visual,
        }

    # ── Human-readable output ───────────────────────────────────────────

    @staticmethod
    def _compute_reltr_usage(logs: list[dict[str, Any]]) -> dict[str, Any]:
        """Compute aggregate RelTR scene graph usage statistics.

        Args:
            logs: List of PathLog dicts.

        Returns:
            Dict with total_triples, mean_triples_per_image,
            total_evidence_hits, evidence_hit_rate.
        """
        n = len(logs)
        if n == 0:
            return {
                "total_triples": 0,
                "mean_triples_per_image": 0.0,
                "total_evidence_hits": 0,
                "evidence_hit_rate": 0.0,
            }

        total_triples = 0
        total_evidence_hits = 0
        images_with_hits = 0

        for rec in logs:
            sg = rec.get("scene_graph", {})
            total_triples += sg.get("n_triples", 0)
            hits = sg.get("n_evidence_hits", 0)
            total_evidence_hits += hits
            if hits > 0:
                images_with_hits += 1

        return {
            "total_triples": total_triples,
            "mean_triples_per_image": round(total_triples / n, 2),
            "total_evidence_hits": total_evidence_hits,
            "evidence_hit_rate": _safe_rate(images_with_hits, n),
        }

    def print_summary(self) -> None:
        """Print a human-readable summary to stdout.

        Suitable for Google Colab output cells.  Calls :meth:`summary`
        internally and formats the result as aligned tables.
        """
        s = self.summary()
        n = s["total_images"]

        print("=" * 60)
        print("  CORRECTION PATH SUMMARY")
        print("=" * 60)
        print(f"  Total images processed:  {n}")
        print()

        if n == 0:
            print("  (no images recorded)")
            print("=" * 60)
            return

        # Dispatch
        dc = s["dispatch_counts"]
        print("  Dispatch Mode")
        print(f"    Enrichment (short):    {dc['enrichment']}")
        print(f"    Surgical   (long):     {dc['surgical']}")
        print()

        # Triples
        tp = s["triples_per_image"]
        print("  Triples per Image")
        print(f"    Mean:                  {tp['mean']:.1f}")
        print(f"    Median:                {tp['median']:.1f}")
        print()

        # Verdict distribution
        print("  Verdict Distribution")
        print(f"    {'Type':<12} {'CORRECT':>8} {'INCORRECT':>10} {'UNKNOWN':>8}")
        print(f"    {'-' * 40}")
        for rel_type in ("SPATIAL", "ACTION", "ATTRIBUTE"):
            vd = s["verdict_distribution"][rel_type]
            print(f"    {rel_type:<12} {vd['CORRECT']:>8} {vd['INCORRECT']:>10} {vd['UNKNOWN']:>8}")
        print()

        # Evidence sources
        if s["top_evidence_sources"]:
            print("  Top Evidence Sources")
            for item in s["top_evidence_sources"][:5]:
                print(f"    {item['source']:<25} {item['count']:>5}")
            print()

        # Guidance types
        gd = s["guidance_type_distribution"]
        print("  Guidance Types")
        for gtype, count in gd.items():
            print(f"    {gtype:<25} {count:>5}")
        print()

        # Rates
        print("  Pipeline Rates")
        print(f"    Batch acceptance:      {s['batch_correction_acceptance_rate']:.1%}")
        print(f"    Fallback deletion:     {s['fallback_deletion_rate']:.1%}")
        print(f"    Post-verify revert:    {s['post_verification_revert_rate']:.1%}")
        print(f"    Addendum acceptance:   {s['addendum_acceptance_rate']:.1%}")
        print()

        # KB usage
        kb = s["kb_usage"]
        print("  KB Usage")
        print(f"    Mean hard facts:       {kb['mean_hard_facts']:.1f}")
        print(f"    Mean spatial facts:    {kb['mean_spatial_facts']:.1f}")
        print(f"    Mean visual desc len:  {kb['mean_visual_desc_len']:.0f}")
        print(f"    Mean detections:       {kb['mean_detections']:.1f}")
        print(f"    Spatial fact hit rate:  {kb['spatial_fact_hit_rate']:.1%}")
        print(f"    Bbox coverage:         {kb['bbox_coverage']:.1%}")
        print(f"    KB-first correct rel:  {kb['correct_rel_kb_first_rate']:.1%}")
        print(f"    Addendum novelty:      {kb['addendum_novelty_rate']:.1%}")
        print()

        # Path effectiveness
        pe = s["path_effectiveness"]
        if pe:
            print("  Path Effectiveness (by dominant evidence source)")
            print(f"    {'Source':<25} {'Images':>7} {'Modified':>9}")
            print(f"    {'-' * 43}")
            for source, data in pe.items():
                print(f"    {source:<25} {data['n_images']:>7} {data['fraction_modified']:>8.1%}")
            print()

        # Geometry usage
        gu = s["geometry_usage"]
        if gu["total_action_triples"] > 0:
            print("  Geometry & Pose Usage (action/attribute triples)")
            print(f"    Total action triples:  {gu['total_action_triples']}")
            print(f"    Geo check possible:    {gu['geo_check_possible']} ({gu['geo_check_rate']:.1%})")
            print(f"    Keypoints loaded:      {gu['keypoints_loaded']} ({gu['keypoints_rate']:.1%})")
            print(f"    Geo confirmed (True):  {gu['geo_confirmed']}")
            print(f"    Geo violated (False):  {gu['geo_violated']}")
            print(f"    Geo-VQA agreement:     {gu['geo_vqa_agreement_rate']:.1%}")
            if gu["family_counts"]:
                print(f"    Family breakdown:")
                for fam, cnt in sorted(gu["family_counts"].items(), key=lambda x: -x[1]):
                    print(f"      {fam:<22} {cnt:>5}")
            print()

        # NLI usage
        nu = s.get("nli_usage", {})
        if nu.get("total_checks", 0) > 0:
            print("  NLI Pre-Filter Usage")
            print(f"    Total NLI checks:      {nu['total_checks']}")
            print(f"    SUPPORT:               {nu['support_count']}")
            print(f"    CONTRADICT:            {nu['contradict_count']}")
            print(f"    NEUTRAL:               {nu['neutral_count']}")
            print(f"    VQA calls saved:       {nu['vqa_calls_saved']}")
            print(f"    Evidence hit rate:      {nu['evidence_hit_rate']:.1%}")
            print(f"    Batch LLM calls:       {nu['batch_calls_made']}")
            print(f"    Entity existence flags: {nu['entity_existence_flags']}")
            print(f"    CONTRADICT tiers:")
            print(f"      High (geometry):     {nu['contradict_high_geometry']}")
            print(f"      High (entity):       {nu['contradict_high_entity']}")
            print(f"      Low  (visual):       {nu['contradict_low_visual']}")
            print()

        print("=" * 60)


# ════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL HELPERS
# ════════════════════════════════════════════════════════════════════════════


def _clamp01(value: float) -> float:
    """Clamp *value* to the ``[0.0, 1.0]`` interval.

    Args:
        value: Numeric value to clamp.

    Returns:
        Clamped float.
    """
    return max(0.0, min(1.0, value))


def _safe_rate(numerator: int | float, denominator: int | float) -> float:
    """Compute a rate clamped to ``[0.0, 1.0]``, returning ``0.0`` on zero denominator.

    Args:
        numerator: Count of successes.
        denominator: Total count.

    Returns:
        Rate as a float in ``[0.0, 1.0]``.
    """
    if denominator <= 0:
        return 0.0
    return _clamp01(numerator / denominator)
