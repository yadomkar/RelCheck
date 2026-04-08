"""ResultStore: JSONL append, pandas CSV export, summary stats."""

import json
import logging
import os
from statistics import mean, median

import pandas as pd

from relcheck_v3.hallucination_generation.models import RecordStatus, ResultRecord, SummaryStats

logger = logging.getLogger(__name__)


class ResultStore:
    """Manages JSONL checkpoint + pandas CSV export + summary statistics.

    File paths:
        - JSONL: {output_dir}/output.jsonl
        - CSV:   {output_dir}/output.csv
        - Summary: {output_dir}/summary_stats.json
    """

    def __init__(self, output_dir: str) -> None:
        """Initialize ResultStore and create output directory if needed.

        Args:
            output_dir: Directory for output files.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.jsonl_path = os.path.join(output_dir, "output.jsonl")
        self.csv_path = os.path.join(output_dir, "output.csv")
        self.summary_path = os.path.join(output_dir, "summary_stats.json")

    def load_checkpoint(self) -> set[str]:
        """Read existing JSONL file and return set of '{image_id}::{gt_cap}' keys.

        Skips corrupted lines with a warning log.

        Returns:
            Set of checkpoint keys for already-processed records.
        """
        keys: set[str] = set()

        if not os.path.exists(self.jsonl_path):
            return keys

        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    image_id = data["image_id"]
                    gt_cap = data["gt_cap"]
                    keys.add(f"{image_id}::{gt_cap}")
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(
                        "Skipping corrupted line %d in %s: %s",
                        line_num,
                        self.jsonl_path,
                        e,
                    )

        return keys

    def append(self, record: ResultRecord) -> None:
        """Append a single result record as JSON line to JSONL file.

        Flushes immediately for crash safety.

        Args:
            record: The ResultRecord to persist.
        """
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(record.model_dump_json() + "\n")
            f.flush()
            os.fsync(f.fileno())

    def export_csv(self) -> None:
        """Read JSONL into pandas DataFrame and export as CSV."""
        records = self._read_jsonl()
        df = pd.DataFrame(records)
        df.to_csv(self.csv_path, index=False)

    def write_summary(self, duration_seconds: float) -> None:
        """Read JSONL into pandas, compute SummaryStats, write summary_stats.json.

        Args:
            duration_seconds: Total pipeline duration in seconds.
        """
        records = self._read_jsonl()
        total = len(records)

        if total == 0:
            stats = SummaryStats(
                total_processed=0,
                accepted_count=0,
                rejected_count_too_small=0,
                rejected_count_too_large=0,
                parse_failure_count=0,
                api_error_count=0,
                type_distribution={},
                type_percentages={},
                edit_distance_mean=0.0,
                edit_distance_median=0.0,
                edit_distance_min=0,
                edit_distance_max=0,
                duration_seconds=duration_seconds,
            )
        else:
            df = pd.DataFrame(records)

            accepted_count = int((df["status"] == RecordStatus.ACCEPTED.value).sum())

            # Rejected records with edit_distance <= 5
            rejected_mask = df["status"] == RecordStatus.REJECTED.value
            rejected_count_too_small = int(
                (rejected_mask & (df["edit_distance"] <= 5)).sum()
            )
            rejected_count_too_large = int(
                (rejected_mask & (df["edit_distance"] >= 50)).sum()
            )

            parse_failure_count = int(
                (df["status"] == RecordStatus.PARSE_FAILURE.value).sum()
            )
            api_error_count = int(
                (df["status"] == RecordStatus.API_ERROR.value).sum()
            )

            # Type distribution
            type_counts = df["hallucination_type"].value_counts().to_dict()
            type_distribution = {str(k): int(v) for k, v in type_counts.items()}
            type_percentages = {
                str(k): float(v) / total for k, v in type_counts.items()
            }

            # Edit distance stats from accepted records only
            accepted_distances = df.loc[
                df["status"] == RecordStatus.ACCEPTED.value, "edit_distance"
            ].tolist()

            if accepted_distances:
                ed_mean = float(mean(accepted_distances))
                ed_median = float(median(accepted_distances))
                ed_min = int(min(accepted_distances))
                ed_max = int(max(accepted_distances))
            else:
                ed_mean = 0.0
                ed_median = 0.0
                ed_min = 0
                ed_max = 0

            stats = SummaryStats(
                total_processed=total,
                accepted_count=accepted_count,
                rejected_count_too_small=rejected_count_too_small,
                rejected_count_too_large=rejected_count_too_large,
                parse_failure_count=parse_failure_count,
                api_error_count=api_error_count,
                type_distribution=type_distribution,
                type_percentages=type_percentages,
                edit_distance_mean=ed_mean,
                edit_distance_median=ed_median,
                edit_distance_min=ed_min,
                edit_distance_max=ed_max,
                duration_seconds=duration_seconds,
            )

        with open(self.summary_path, "w", encoding="utf-8") as f:
            f.write(stats.model_dump_json(indent=2) + "\n")

    def _read_jsonl(self) -> list[dict]:
        """Read all valid records from the JSONL file.

        Returns:
            List of record dicts. Empty list if file doesn't exist.
        """
        if not os.path.exists(self.jsonl_path):
            return []

        records: list[dict] = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(
                        "Skipping corrupted line %d in %s: %s",
                        line_num,
                        self.jsonl_path,
                        e,
                    )

        return records
