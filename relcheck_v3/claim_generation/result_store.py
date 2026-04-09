"""ResultStore: JSONL checkpoint, CSV/JSONL export."""

import json
import logging
import os

import pandas as pd

from relcheck_v3.claim_generation.models import SampleResult

logger = logging.getLogger(__name__)


class ResultStore:
    """Manages JSONL checkpoint + CSV/JSONL export for claim generation results.

    File paths:
        - JSONL checkpoint: {output_dir}/output.jsonl
        - CSV export:       {output_dir}/output.csv
        - JSONL export:     {output_dir}/results.jsonl
    """

    def __init__(self, output_dir: str, checkpoint_interval: int = 50) -> None:
        """Initialize ResultStore and create output directory if needed.

        Args:
            output_dir: Directory for output files.
            checkpoint_interval: Save checkpoint every N samples.
        """
        self.output_dir = output_dir
        self.checkpoint_interval = checkpoint_interval
        os.makedirs(output_dir, exist_ok=True)

        self.jsonl_path = os.path.join(output_dir, "output.jsonl")
        self.csv_path = os.path.join(output_dir, "output.csv")
        self.export_jsonl_path = os.path.join(output_dir, "results.jsonl")

    def load_checkpoint(self) -> dict[str, SampleResult]:
        """Load completed results from JSONL checkpoint file.

        Returns dict mapping image_id -> SampleResult. Handles corrupted
        lines gracefully by logging a warning and skipping them.

        Returns:
            Dict of {image_id: SampleResult} for already-processed samples.
        """
        results: dict[str, SampleResult] = {}

        if not os.path.exists(self.jsonl_path):
            return results

        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    result = SampleResult.model_validate_json(line)
                    results[result.image_id] = result
                except Exception as e:
                    logger.warning(
                        "Skipping corrupted line %d in %s: %s",
                        line_num,
                        self.jsonl_path,
                        e,
                    )

        return results

    def append(self, result: SampleResult) -> None:
        """Append one result to JSONL checkpoint, flush immediately.

        Args:
            result: The SampleResult to persist.
        """
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(result.model_dump_json() + "\n")
            f.flush()
            os.fsync(f.fileno())

    def should_save_checkpoint(self, count: int) -> bool:
        """Check if a checkpoint should be saved.

        Returns True when count is a positive multiple of checkpoint_interval.

        Args:
            count: Number of samples processed so far.

        Returns:
            True if checkpoint should be saved.
        """
        return count > 0 and count % self.checkpoint_interval == 0

    def export_csv(self) -> None:
        """Export batch results to CSV with key columns.

        Columns: image_id, ref_cap, vkb_text, key_concepts, success, error_message.
        """
        records = self._read_jsonl()
        if not records:
            pd.DataFrame().to_csv(self.csv_path, index=False)
            return

        rows = []
        for rec in records:
            rows.append(
                {
                    "image_id": rec["image_id"],
                    "ref_cap": rec["ref_cap"],
                    "vkb_text": rec.get("vkb_text", ""),
                    "key_concepts": ", ".join(rec.get("key_concepts", [])),
                    "success": rec.get("success", False),
                    "error_message": rec.get("error_message"),
                }
            )

        df = pd.DataFrame(rows)
        df.to_csv(self.csv_path, index=False)

    def export_jsonl(self) -> None:
        """Export full per-sample results to JSONL for detailed analysis."""
        records = self._read_jsonl()

        with open(self.export_jsonl_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

    def _read_jsonl(self) -> list[dict]:
        """Read all valid records from the JSONL checkpoint file.

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
