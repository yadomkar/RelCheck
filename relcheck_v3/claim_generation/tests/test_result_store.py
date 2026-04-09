"""Tests for ResultStore."""

import json
import os

import pytest

from relcheck_v3.claim_generation.models import (
    AttributeQA,
    AttributeQuestion,
    CountClaim,
    ObjectAnswer,
    OverallClaim,
    SampleResult,
    SpecificClaim,
    StageTimings,
    VisualKnowledgeBase,
)
from relcheck_v3.claim_generation.result_store import ResultStore


def _make_sample_result(
    image_id: str = "img_001",
    ref_cap: str = "a cat on a mat",
    success: bool = True,
    error_message: str | None = None,
) -> SampleResult:
    """Create a minimal SampleResult for testing."""
    return SampleResult(
        image_id=image_id,
        ref_cap=ref_cap,
        key_concepts=["cat", "mat"],
        object_questions=[
            "Is there any cat in the image? How many are there?",
            "Is there any mat in the image? How many are there?",
        ],
        attribute_questions=[
            AttributeQuestion(question="What color is the cat?", entities=["cat"]),
        ],
        object_answers={
            "cat": ObjectAnswer(object_name="cat", count=1, bboxes=[[0.1, 0.2, 0.3, 0.4]]),
            "mat": ObjectAnswer(object_name="mat", count=1, bboxes=[[0.5, 0.6, 0.7, 0.8]]),
        },
        attribute_answers=[
            AttributeQA(question="What color is the cat?", entities=["cat"], answer="orange"),
        ],
        visual_knowledge_base=VisualKnowledgeBase(
            count_claims=[
                CountClaim(object_name="cat", count=1, claim_text="There is 1 cat.", bboxes=[[0.1, 0.2, 0.3, 0.4]]),
            ],
            specific_claims=[
                SpecificClaim(object_name="cat", claim_text="The cat is orange."),
            ],
            overall_claims=[
                OverallClaim(claim_text="The cat is on the mat."),
            ],
        ),
        vkb_text="Count:\n1. There is 1 cat.\nSpecific:\n1. The cat is orange.\nOverall:\n1. The cat is on the mat.",
        timings=StageTimings(
            stage1_seconds=0.5,
            stage2_seconds=0.3,
            stage3_seconds=1.2,
            stage4_seconds=0.4,
            total_seconds=2.4,
        ),
        success=success,
        error_message=error_message,
    )


class TestResultStoreInit:
    def test_creates_output_dir(self, tmp_path):
        out = str(tmp_path / "new_dir" / "nested")
        store = ResultStore(out)
        assert os.path.isdir(out)
        assert store.jsonl_path == os.path.join(out, "output.jsonl")
        assert store.csv_path == os.path.join(out, "output.csv")
        assert store.export_jsonl_path == os.path.join(out, "results.jsonl")

    def test_existing_dir_ok(self, tmp_path):
        store = ResultStore(str(tmp_path))
        assert os.path.isdir(str(tmp_path))

    def test_default_checkpoint_interval(self, tmp_path):
        store = ResultStore(str(tmp_path))
        assert store.checkpoint_interval == 50

    def test_custom_checkpoint_interval(self, tmp_path):
        store = ResultStore(str(tmp_path), checkpoint_interval=10)
        assert store.checkpoint_interval == 10


class TestAppend:
    def test_append_creates_file_and_writes_json_line(self, tmp_path):
        store = ResultStore(str(tmp_path))
        result = _make_sample_result()
        store.append(result)

        with open(store.jsonl_path, "r") as f:
            lines = f.readlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["image_id"] == "img_001"
        assert data["success"] is True

    def test_append_multiple_records(self, tmp_path):
        store = ResultStore(str(tmp_path))
        store.append(_make_sample_result(image_id="1"))
        store.append(_make_sample_result(image_id="2"))
        store.append(_make_sample_result(image_id="3"))

        with open(store.jsonl_path, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        assert len(lines) == 3
        ids = [json.loads(line)["image_id"] for line in lines]
        assert ids == ["1", "2", "3"]


class TestLoadCheckpoint:
    def test_empty_when_no_file(self, tmp_path):
        store = ResultStore(str(tmp_path))
        results = store.load_checkpoint()
        assert results == {}

    def test_loads_results_correctly(self, tmp_path):
        store = ResultStore(str(tmp_path))
        store.append(_make_sample_result(image_id="A"))
        store.append(_make_sample_result(image_id="B"))

        results = store.load_checkpoint()
        assert set(results.keys()) == {"A", "B"}
        assert isinstance(results["A"], SampleResult)
        assert results["A"].image_id == "A"

    def test_skips_corrupted_lines(self, tmp_path):
        store = ResultStore(str(tmp_path))
        store.append(_make_sample_result(image_id="good"))

        # Write a corrupted line
        with open(store.jsonl_path, "a") as f:
            f.write("this is not json\n")

        store.append(_make_sample_result(image_id="also_good"))

        results = store.load_checkpoint()
        assert set(results.keys()) == {"good", "also_good"}

    def test_skips_invalid_schema_lines(self, tmp_path):
        store = ResultStore(str(tmp_path))
        # Write valid JSON but invalid SampleResult schema
        with open(store.jsonl_path, "w") as f:
            f.write(json.dumps({"foo": "bar"}) + "\n")

        store.append(_make_sample_result(image_id="valid"))

        results = store.load_checkpoint()
        assert set(results.keys()) == {"valid"}

    def test_last_result_wins_for_duplicate_image_id(self, tmp_path):
        store = ResultStore(str(tmp_path))
        store.append(_make_sample_result(image_id="dup", ref_cap="first"))
        store.append(_make_sample_result(image_id="dup", ref_cap="second"))

        results = store.load_checkpoint()
        assert len(results) == 1
        assert results["dup"].ref_cap == "second"


class TestShouldSaveCheckpoint:
    def test_returns_false_for_zero(self, tmp_path):
        store = ResultStore(str(tmp_path), checkpoint_interval=50)
        assert store.should_save_checkpoint(0) is False

    def test_returns_true_at_interval(self, tmp_path):
        store = ResultStore(str(tmp_path), checkpoint_interval=50)
        assert store.should_save_checkpoint(50) is True
        assert store.should_save_checkpoint(100) is True

    def test_returns_false_between_intervals(self, tmp_path):
        store = ResultStore(str(tmp_path), checkpoint_interval=50)
        assert store.should_save_checkpoint(1) is False
        assert store.should_save_checkpoint(49) is False
        assert store.should_save_checkpoint(51) is False

    def test_custom_interval(self, tmp_path):
        store = ResultStore(str(tmp_path), checkpoint_interval=10)
        assert store.should_save_checkpoint(10) is True
        assert store.should_save_checkpoint(20) is True
        assert store.should_save_checkpoint(15) is False


class TestExportCsv:
    def test_export_creates_csv_with_correct_columns(self, tmp_path):
        store = ResultStore(str(tmp_path))
        store.append(_make_sample_result(image_id="1"))
        store.append(_make_sample_result(image_id="2"))
        store.export_csv()

        assert os.path.exists(store.csv_path)
        import pandas as pd

        df = pd.read_csv(store.csv_path)
        assert len(df) == 2
        assert "image_id" in df.columns
        assert "ref_cap" in df.columns
        assert "vkb_text" in df.columns
        assert "key_concepts" in df.columns
        assert "success" in df.columns
        assert "error_message" in df.columns

    def test_export_csv_values(self, tmp_path):
        store = ResultStore(str(tmp_path))
        store.append(_make_sample_result(image_id="test1", ref_cap="hello world"))
        store.export_csv()

        import pandas as pd

        df = pd.read_csv(store.csv_path)
        assert df.iloc[0]["image_id"] == "test1"
        assert df.iloc[0]["ref_cap"] == "hello world"
        assert df.iloc[0]["key_concepts"] == "cat, mat"
        assert df.iloc[0]["success"] == True  # noqa: E712 (pandas returns np.bool_)

    def test_export_empty_jsonl(self, tmp_path):
        store = ResultStore(str(tmp_path))
        store.export_csv()
        assert os.path.exists(store.csv_path)

    def test_export_csv_with_error(self, tmp_path):
        store = ResultStore(str(tmp_path))
        store.append(
            _make_sample_result(image_id="err", success=False, error_message="stage 1 failed")
        )
        store.export_csv()

        import pandas as pd

        df = pd.read_csv(store.csv_path)
        assert df.iloc[0]["success"] == False  # noqa: E712 (pandas returns np.bool_)
        assert df.iloc[0]["error_message"] == "stage 1 failed"


class TestExportJsonl:
    def test_export_creates_jsonl_file(self, tmp_path):
        store = ResultStore(str(tmp_path))
        store.append(_make_sample_result(image_id="1"))
        store.append(_make_sample_result(image_id="2"))
        store.export_jsonl()

        assert os.path.exists(store.export_jsonl_path)
        with open(store.export_jsonl_path, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        assert len(lines) == 2

        # Verify full SampleResult data is present
        data = json.loads(lines[0])
        assert data["image_id"] == "1"
        assert "key_concepts" in data
        assert "visual_knowledge_base" in data
        assert "timings" in data

    def test_export_jsonl_round_trip(self, tmp_path):
        store = ResultStore(str(tmp_path))
        original = _make_sample_result(image_id="rt")
        store.append(original)
        store.export_jsonl()

        with open(store.export_jsonl_path, "r") as f:
            line = f.readline().strip()
        restored = SampleResult.model_validate_json(line)
        assert restored == original

    def test_export_empty_jsonl(self, tmp_path):
        store = ResultStore(str(tmp_path))
        store.export_jsonl()
        assert os.path.exists(store.export_jsonl_path)
        with open(store.export_jsonl_path, "r") as f:
            assert f.read().strip() == ""
