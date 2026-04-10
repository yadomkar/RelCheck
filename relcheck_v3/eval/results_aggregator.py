"""Results aggregation for the multi-benchmark evaluation harness.

Scans per-run JSON result files, builds a master results table and ablation
delta view, and exports to CSV + Markdown for thesis inclusion.

Defines lightweight Pydantic models for ``SplitResult`` and ``RunResult``
used to parse the per-run JSON files produced by the evaluation runner.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models for per-run JSON files
# ---------------------------------------------------------------------------


class SplitResult(BaseModel):
    """Metrics for one benchmark split."""

    split: str
    benchmark: str
    system_id: str
    n_samples: int
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    yes_ratio: float | None = None
    accuracy_plus: float | None = None
    chair: float | None = None
    cover: float | None = None
    hal: float | None = None
    cog: float | None = None
    reltr_tag: bool | None = None


class RunResult(BaseModel):
    """Full result for one benchmark × system run."""

    benchmark: str
    system_id: str
    mllm_model_id: str
    corrector_model: str
    splits: list[SplitResult]


# ---------------------------------------------------------------------------
# Column mapping: (benchmark, split) → master-table column name + metric key
# ---------------------------------------------------------------------------


# Each entry maps (benchmark, split) to (column_name, metric_field).
# POPE uses F1, MME uses the combined score (acc + acc+) × 100,
# AMBER discriminative uses accuracy, AMBER generative uses CHAIR.
_COLUMN_MAP: list[tuple[str, str, str, str]] = [
    # (benchmark, split, column_name, metric_field)
    ("pope", "random", "POPE-random F1", "f1"),
    ("pope", "popular", "POPE-popular F1", "f1"),
    ("pope", "adversarial", "POPE-adversarial F1", "f1"),
    ("mme", "existence", "MME-existence", "accuracy"),
    ("mme", "count", "MME-count", "accuracy"),
    ("mme", "position", "MME-position", "accuracy"),
    ("mme", "color", "MME-color", "accuracy"),
    ("amber", "de", "AMBER-de", "accuracy"),
    ("amber", "da", "AMBER-da", "accuracy"),
    ("amber", "dr", "AMBER-dr", "accuracy"),
    ("amber", "g", "AMBER-g-CHAIR", "chair"),
]

# Ordered system IDs for consistent row ordering.
_SYSTEM_ORDER: list[str] = [
    "raw",
    "woodpecker",
    "claim",
    "claim+geom",
    "full",
]


class ResultsAggregator:
    """Aggregate per-run JSON results into a master table and ablation delta.

    Args:
        results_dir: Path to directory containing per-run JSON result files.
            Each file must be a valid ``RunResult`` JSON object.
    """

    def __init__(self, results_dir: str) -> None:
        self._results_dir = Path(results_dir)
        self._run_results: list[RunResult] = []
        self._master_table: pd.DataFrame | None = None
        self._ablation_delta: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_run_results(self) -> list[RunResult]:
        """Load all ``RunResult`` JSON files from the results directory."""
        results: list[RunResult] = []
        if not self._results_dir.exists():
            logger.warning("Results directory does not exist: %s", self._results_dir)
            return results

        for json_path in sorted(self._results_dir.glob("*.json")):
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
                results.append(RunResult.model_validate(data))
                logger.debug("Loaded result file: %s", json_path.name)
            except (json.JSONDecodeError, Exception) as exc:  # noqa: BLE001
                logger.warning("Skipping invalid result file %s: %s", json_path.name, exc)
        return results

    @staticmethod
    def _extract_metric(split: SplitResult, metric_field: str) -> float | None:
        """Extract a metric value from a ``SplitResult`` by field name."""
        return getattr(split, metric_field, None)

    def _build_table_from_results(
        self,
        run_results: list[RunResult],
        column_map: list[tuple[str, str, str, str]],
        suffix: str = "",
    ) -> pd.DataFrame:
        """Build a DataFrame from run results using the given column map.

        Args:
            run_results: Parsed run result objects.
            column_map: List of (benchmark, split, column_name, metric_field).
            suffix: Optional suffix to append to column names (for stratified).

        Returns:
            DataFrame with system_id as index and metric columns.
        """
        # Index splits by (system_id, benchmark, split) for fast lookup.
        split_index: dict[tuple[str, str, str], SplitResult] = {}
        for run in run_results:
            for split in run.splits:
                key = (run.system_id, split.benchmark, split.split)
                split_index[key] = split

        rows: list[dict[str, Any]] = []
        for system_id in _SYSTEM_ORDER:
            row: dict[str, Any] = {"system_id": system_id}
            for benchmark, split, col_name, metric_field in column_map:
                full_col = f"{col_name}{suffix}" if suffix else col_name
                sr = split_index.get((system_id, benchmark, split))
                if sr is not None:
                    row[full_col] = self._extract_metric(sr, metric_field)
                else:
                    row[full_col] = None
            rows.append(row)

        df = pd.DataFrame(rows)
        df = df.set_index("system_id")
        return df

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_master_table(self) -> pd.DataFrame:
        """Scan results_dir for per-run JSON files, aggregate into master table.

        Rows correspond to the five correction systems (raw, woodpecker,
        claim, claim+geom, full). Columns correspond to per-benchmark
        metrics as defined in the design document.

        Returns:
            ``pandas.DataFrame`` with ``system_id`` as index.
        """
        self._run_results = self._load_run_results()
        if not self._run_results:
            logger.warning("No run results found in %s", self._results_dir)

        self._master_table = self._build_table_from_results(
            self._run_results, _COLUMN_MAP
        )
        return self._master_table

    def build_ablation_delta(
        self, baseline: str = "woodpecker"
    ) -> pd.DataFrame:
        """Compute each system's gain over the baseline per column.

        Uses ``pandas`` vectorized subtraction so every numeric cell becomes
        ``master[system][metric] - master[baseline][metric]``.

        Args:
            baseline: System ID to use as the reference row.

        Returns:
            ``pandas.DataFrame`` with the same shape as the master table.

        Raises:
            ValueError: If the master table has not been built yet or the
                baseline system is not present.
        """
        if self._master_table is None:
            self.build_master_table()
        assert self._master_table is not None  # for type checker

        if baseline not in self._master_table.index:
            raise ValueError(
                f"Baseline system '{baseline}' not found in master table. "
                f"Available systems: {list(self._master_table.index)}"
            )

        baseline_row = self._master_table.loc[baseline]
        self._ablation_delta = self._master_table.subtract(baseline_row, axis="columns")
        return self._ablation_delta

    def build_stratified_tables(self) -> dict[str, pd.DataFrame]:
        """Build stratified tables split by ``reltr_tag`` metadata.

        Produces separate metric columns for "SCENE present"
        (``reltr_tag=True``) and "SCENE empty" (``reltr_tag=False``)
        subsets. Only applicable to benchmarks that provide ``reltr_tag``
        (i.e. POPE with COCO annotations).

        Returns:
            Dictionary with keys ``"scene_present"`` and ``"scene_empty"``,
            each mapping to a ``pandas.DataFrame``.
        """
        if not self._run_results:
            self._run_results = self._load_run_results()

        # Partition splits by reltr_tag.
        present_results: list[RunResult] = []
        empty_results: list[RunResult] = []

        for run in self._run_results:
            present_splits = [s for s in run.splits if s.reltr_tag is True]
            empty_splits = [s for s in run.splits if s.reltr_tag is False]

            if present_splits:
                present_results.append(
                    RunResult(
                        benchmark=run.benchmark,
                        system_id=run.system_id,
                        mllm_model_id=run.mllm_model_id,
                        corrector_model=run.corrector_model,
                        splits=present_splits,
                    )
                )
            if empty_splits:
                empty_results.append(
                    RunResult(
                        benchmark=run.benchmark,
                        system_id=run.system_id,
                        mllm_model_id=run.mllm_model_id,
                        corrector_model=run.corrector_model,
                        splits=empty_splits,
                    )
                )

        scene_present = self._build_table_from_results(
            present_results, _COLUMN_MAP, suffix=" (SCENE present)"
        )
        scene_empty = self._build_table_from_results(
            empty_results, _COLUMN_MAP, suffix=" (SCENE empty)"
        )

        return {"scene_present": scene_present, "scene_empty": scene_empty}

    def export(self, output_dir: str) -> None:
        """Save master table and ablation delta as CSV and Markdown.

        Creates the output directory if it does not exist. Produces four
        files:

        - ``master_table.csv``
        - ``master_table.md``
        - ``ablation_delta.csv``
        - ``ablation_delta.md``

        If stratified data is available, also exports:

        - ``stratified_scene_present.csv``
        - ``stratified_scene_present.md``
        - ``stratified_scene_empty.csv``
        - ``stratified_scene_empty.md``

        Args:
            output_dir: Directory path for output files.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Ensure tables are built.
        if self._master_table is None:
            self.build_master_table()
        assert self._master_table is not None

        if self._ablation_delta is None:
            self.build_ablation_delta()
        assert self._ablation_delta is not None

        # Master table.
        master_csv = out / "master_table.csv"
        master_md = out / "master_table.md"
        self._master_table.to_csv(master_csv)
        master_md.write_text(
            self._master_table.to_markdown() or "", encoding="utf-8"
        )
        logger.info("Exported master table to %s and %s", master_csv, master_md)

        # Ablation delta.
        delta_csv = out / "ablation_delta.csv"
        delta_md = out / "ablation_delta.md"
        self._ablation_delta.to_csv(delta_csv)
        delta_md.write_text(
            self._ablation_delta.to_markdown() or "", encoding="utf-8"
        )
        logger.info("Exported ablation delta to %s and %s", delta_csv, delta_md)

        # Stratified tables (only if reltr_tag data exists).
        stratified = self.build_stratified_tables()
        for tag_key, df in stratified.items():
            if df.notna().any().any():
                csv_path = out / f"stratified_{tag_key}.csv"
                md_path = out / f"stratified_{tag_key}.md"
                df.to_csv(csv_path)
                md_path.write_text(df.to_markdown() or "", encoding="utf-8")
                logger.info("Exported stratified table to %s", csv_path)
