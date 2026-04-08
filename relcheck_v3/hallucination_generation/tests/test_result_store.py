"""Tests for ResultStore."""

import json
import os

import pytest

from relcheck_v3.hallucination_generation.models import RecordStatus, ResultRecord, SummaryStats
from relcheck_v3.hallucination_generation.result_store import ResultStore


def _make_record(
    image_id: str = "12345",
    gt_cap: str = "a cat on a mat",
    ref_cap: str = "a dog on a mat",
    hallucination_type: str = "Object Existence",
    reason: str = "replaced cat with dog",
    edit_distance: int = 6,
    status: RecordStatus = RecordStatus.ACCEPTED,
) -> ResultRecord:
    return ResultRecord(
        image_id=image_id,
        gt_cap=gt_cap,
        ref_cap=ref_cap,
        hallucination_type=hallucination_type,
        reason=reason,
        edit_distance=edit_distance,
        status=status,
    )


class TestResultStoreInit:
    def test_creates_output_dir(self, tmp_path):
        out = str(tmp_path / "new_dir" / "nested")
        store = ResultStore(out)
        assert os.path.isdir(out)
        assert store.jsonl_path == os.path.join(out, "output.jsonl")
        assert store.csv_path == os.path.join(out, "output.csv")
        assert store.summary_path == os.path.join(out, "summary_stats.json")

    def test_existing_dir_ok(self, tmp_path):
        store = ResultStore(str(tmp_path))
        assert os.path.isdir(str(tmp_path))


class TestAppend:
    def test_append_creates_file_and_writes_json_line(self, tmp_path):
        store = ResultStore(str(tmp_path))
        rec = _make_record()
        store.append(rec)

        with open(store.jsonl_path, "r") as f:
            lines = f.readlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["image_id"] == "12345"
        assert data["status"] == "accepted"

    def test_append_multiple_records(self, tmp_path):
        store = ResultStore(str(tmp_path))
        store.append(_make_record(image_id="1"))
        store.append(_make_record(image_id="2"))
        store.append(_make_record(image_id="3"))

        with open(store.jsonl_path, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        assert len(lines) == 3
        ids = [json.loads(l)["image_id"] for l in lines]
        assert ids == ["1", "2", "3"]


class TestLoadCheckpoint:
    def test_empty_when_no_file(self, tmp_path):
        store = ResultStore(str(tmp_path))
        keys = store.load_checkpoint()
        assert keys == set()

    def test_loads_keys_correctly(self, tmp_path):
        store = ResultStore(str(tmp_path))
        store.append(_make_record(image_id="A", gt_cap="cap1"))
        store.append(_make_record(image_id="B", gt_cap="cap2"))

        keys = store.load_checkpoint()
        assert keys == {"A::cap1", "B::cap2"}

    def test_skips_corrupted_lines(self, tmp_path):
        store = ResultStore(str(tmp_path))
        store.append(_make_record(image_id="good", gt_cap="ok"))

        # Write a corrupted line
        with open(store.jsonl_path, "a") as f:
            f.write("this is not json\n")

        store.append(_make_record(image_id="also_good", gt_cap="fine"))

        keys = store.load_checkpoint()
        assert keys == {"good::ok", "also_good::fine"}

    def test_skips_lines_missing_keys(self, tmp_path):
        store = ResultStore(str(tmp_path))
        # Write a valid JSON line but missing required keys
        with open(store.jsonl_path, "w") as f:
            f.write(json.dumps({"foo": "bar"}) + "\n")
            f.write(
                json.dumps({"image_id": "x", "gt_cap": "y", "other": "z"}) + "\n"
            )

        keys = store.load_checkpoint()
        # First line missing image_id/gt_cap -> skipped; second has both
        assert keys == {"x::y"}


class TestExportCsv:
    def test_export_creates_csv(self, tmp_path):
        store = ResultStore(str(tmp_path))
        store.append(_make_record(image_id="1"))
        store.append(_make_record(image_id="2"))
        store.export_csv()

        assert os.path.exists(store.csv_path)
        import pandas as pd

        df = pd.read_csv(store.csv_path)
        assert len(df) == 2
        assert list(df["image_id"].astype(str)) == ["1", "2"]

    def test_export_empty_jsonl(self, tmp_path):
        store = ResultStore(str(tmp_path))
        store.export_csv()
        assert os.path.exists(store.csv_path)


class TestWriteSummary:
    def test_summary_with_mixed_records(self, tmp_path):
        store = ResultStore(str(tmp_path))

        # 2 accepted
        store.append(
            _make_record(
                image_id="1",
                status=RecordStatus.ACCEPTED,
                edit_distance=10,
                hallucination_type="Object Existence",
            )
        )
        store.append(
            _make_record(
                image_id="2",
                status=RecordStatus.ACCEPTED,
                edit_distance=20,
                hallucination_type="Attribute",
            )
        )
        # 1 rejected too small
        store.append(
            _make_record(
                image_id="3",
                status=RecordStatus.REJECTED,
                edit_distance=3,
                hallucination_type="Interaction",
            )
        )
        # 1 rejected too large
        store.append(
            _make_record(
                image_id="4",
                status=RecordStatus.REJECTED,
                edit_distance=55,
                hallucination_type="Count",
            )
        )
        # 1 parse failure
        store.append(
            _make_record(
                image_id="5",
                status=RecordStatus.PARSE_FAILURE,
                edit_distance=0,
                hallucination_type="Object Existence",
            )
        )

        store.write_summary(duration_seconds=42.5)

        assert os.path.exists(store.summary_path)
        with open(store.summary_path, "r") as f:
            stats_data = json.load(f)

        stats = SummaryStats(**stats_data)
        assert stats.total_processed == 5
        assert stats.accepted_count == 2
        assert stats.rejected_count_too_small == 1
        assert stats.rejected_count_too_large == 1
        assert stats.parse_failure_count == 1
        assert stats.api_error_count == 0
        assert stats.duration_seconds == 42.5

        # Edit distance stats from accepted only (10, 20)
        assert stats.edit_distance_mean == 15.0
        assert stats.edit_distance_median == 15.0
        assert stats.edit_distance_min == 10
        assert stats.edit_distance_max == 20

        # Type distribution
        assert stats.type_distribution["Object Existence"] == 2
        assert stats.type_distribution["Attribute"] == 1

    def test_summary_no_accepted_records(self, tmp_path):
        store = ResultStore(str(tmp_path))
        store.append(
            _make_record(
                image_id="1",
                status=RecordStatus.REJECTED,
                edit_distance=3,
            )
        )

        store.write_summary(duration_seconds=1.0)

        with open(store.summary_path, "r") as f:
            stats_data = json.load(f)

        stats = SummaryStats(**stats_data)
        assert stats.total_processed == 1
        assert stats.accepted_count == 0
        assert stats.edit_distance_mean == 0.0
        assert stats.edit_distance_median == 0.0
        assert stats.edit_distance_min == 0
        assert stats.edit_distance_max == 0

    def test_summary_empty_jsonl(self, tmp_path):
        store = ResultStore(str(tmp_path))
        store.write_summary(duration_seconds=0.0)

        with open(store.summary_path, "r") as f:
            stats_data = json.load(f)

        stats = SummaryStats(**stats_data)
        assert stats.total_processed == 0
        assert stats.accepted_count == 0

    def test_summary_type_percentages(self, tmp_path):
        store = ResultStore(str(tmp_path))
        for i in range(4):
            types = [
                "Object Existence",
                "Attribute",
                "Interaction",
                "Count",
            ]
            store.append(
                _make_record(
                    image_id=str(i),
                    status=RecordStatus.ACCEPTED,
                    edit_distance=10,
                    hallucination_type=types[i],
                )
            )

        store.write_summary(duration_seconds=5.0)

        with open(store.summary_path, "r") as f:
            stats_data = json.load(f)

        stats = SummaryStats(**stats_data)
        for t in ["Object Existence", "Attribute", "Interaction", "Count"]:
            assert stats.type_distribution[t] == 1
            assert stats.type_percentages[t] == 0.25
