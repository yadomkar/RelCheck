"""Unit tests for EvalPipeline."""

import json
import os

import pytest

from relcheck_v3.eval.config import EvalConfig
from relcheck_v3.eval.models import (
    CESample,
    CaptionEditingScores,
    POPEDomain,
    POPEQuestion,
    POPEScores,
    POPESetting,
)
from relcheck_v3.eval.pipeline import EvalPipeline, _extract_yes_no


# ---------------------------------------------------------------------------
# Helpers: mock Caption_Editor and POPE_Responder
# ---------------------------------------------------------------------------


class MockCaptionEditor:
    """Returns ref_cap uppercased as the 'edited' caption."""

    def __init__(self, fail_on: set[str] | None = None):
        self._fail_on = fail_on or set()
        self.call_count = 0

    def edit_caption(self, image: str, ref_cap: str) -> str:
        self.call_count += 1
        if image in self._fail_on:
            raise RuntimeError(f"Simulated failure for {image}")
        return ref_cap.upper()


class MockPOPEResponder:
    """Returns 'yes' for every question."""

    def __init__(self, fail_on: set[str] | None = None):
        self._fail_on = fail_on or set()
        self.call_count = 0

    def answer_pope(self, image: str, question: str) -> str:
        self.call_count += 1
        if image in self._fail_on:
            raise RuntimeError(f"Simulated failure for {image}")
        return "yes"


# ---------------------------------------------------------------------------
# _extract_yes_no
# ---------------------------------------------------------------------------


class TestExtractYesNo:
    def test_yes(self):
        assert _extract_yes_no("yes") == "yes"

    def test_no(self):
        assert _extract_yes_no("no") == "no"

    def test_yes_with_trailing(self):
        assert _extract_yes_no("Yes, there is a cat.") == "yes"

    def test_no_with_trailing(self):
        assert _extract_yes_no("No, I don't see one.") == "no"

    def test_ambiguous_falls_back_to_no(self):
        assert _extract_yes_no("Maybe there is something") == "no"

    def test_empty_string_falls_back_to_no(self):
        assert _extract_yes_no("") == "no"

    def test_whitespace_only_falls_back_to_no(self):
        assert _extract_yes_no("   ") == "no"


# ---------------------------------------------------------------------------
# Path validation
# ---------------------------------------------------------------------------


class TestPathValidation:
    def test_empty_paths_pass_validation(self, tmp_path):
        """All-empty paths should not trigger validation errors."""
        config = EvalConfig(
            model_name="test",
            output_dir=str(tmp_path / "out"),
        )
        # Should not raise
        EvalPipeline(config)

    def test_existing_paths_pass(self, tmp_path):
        ce_file = tmp_path / "coco_ce.json"
        ce_file.write_text("{}")
        config = EvalConfig(
            model_name="test",
            coco_ce_path=str(ce_file),
            output_dir=str(tmp_path / "out"),
        )
        EvalPipeline(config)

    def test_single_missing_path_raises(self, tmp_path):
        config = EvalConfig(
            model_name="test",
            coco_ce_path="/nonexistent/path/coco.json",
            output_dir=str(tmp_path / "out"),
        )
        with pytest.raises(ValueError, match="/nonexistent/path/coco.json"):
            EvalPipeline(config)

    def test_multiple_missing_paths_all_listed(self, tmp_path):
        config = EvalConfig(
            model_name="test",
            coco_ce_path="/missing/a",
            flickr_ce_path="/missing/b",
            pope_data_dir="/missing/c",
            output_dir=str(tmp_path / "out"),
        )
        with pytest.raises(ValueError) as exc_info:
            EvalPipeline(config)
        msg = str(exc_info.value)
        assert "/missing/a" in msg
        assert "/missing/b" in msg
        assert "/missing/c" in msg


# ---------------------------------------------------------------------------
# Caption Editing
# ---------------------------------------------------------------------------


def _make_samples(n: int, tmp_path) -> list[CESample]:
    """Create n dummy CESample objects."""
    return [
        CESample(
            image_id=f"img_{i:04d}",
            gt_cap=f"ground truth {i}",
            ref_cap=f"reference {i}",
            image_path=str(tmp_path / f"img_{i}.jpg"),
        )
        for i in range(n)
    ]


class TestRunCaptionEditing:
    def test_no_editor_raises(self, tmp_path):
        config = EvalConfig(model_name="test", output_dir=str(tmp_path / "out"))
        pipeline = EvalPipeline(config)
        with pytest.raises(RuntimeError, match="Caption_Editor"):
            pipeline.run_caption_editing("coco_ce", [])

    def test_basic_run(self, tmp_path, monkeypatch):
        """Run with a mock editor and verify scores are returned + files saved."""
        # Monkeypatch CaptionMetrics.compute to avoid pycocoevalcap dependency
        from relcheck_v3.eval import metrics

        monkeypatch.setattr(
            metrics.CaptionMetrics,
            "compute",
            staticmethod(
                lambda preds: CaptionEditingScores(
                    bleu_1=50.0, bleu_4=30.0, rouge_l=60.0, cider=400.0, spice=40.0
                )
            ),
        )

        editor = MockCaptionEditor()
        config = EvalConfig(model_name="test", output_dir=str(tmp_path / "out"))
        pipeline = EvalPipeline(config, caption_editor=editor)

        samples = _make_samples(5, tmp_path)
        scores = pipeline.run_caption_editing("coco_ce", samples)

        assert isinstance(scores, CaptionEditingScores)
        assert editor.call_count == 5

        # Check CSV was saved
        csv_path = tmp_path / "out" / "test_coco_ce_predictions.csv"
        assert csv_path.exists()

        # Check JSON was saved
        json_path = tmp_path / "out" / "test_coco_ce_scores.json"
        assert json_path.exists()

    def test_max_samples_limits_processing(self, tmp_path, monkeypatch):
        from relcheck_v3.eval import metrics

        monkeypatch.setattr(
            metrics.CaptionMetrics,
            "compute",
            staticmethod(
                lambda preds: CaptionEditingScores(
                    bleu_1=0, bleu_4=0, rouge_l=0, cider=0, spice=0
                )
            ),
        )

        editor = MockCaptionEditor()
        config = EvalConfig(
            model_name="test",
            output_dir=str(tmp_path / "out"),
            max_samples=3,
        )
        pipeline = EvalPipeline(config, caption_editor=editor)

        samples = _make_samples(10, tmp_path)
        pipeline.run_caption_editing("coco_ce", samples)

        assert editor.call_count == 3

    def test_error_fallback_uses_ref_cap(self, tmp_path, monkeypatch):
        from relcheck_v3.eval import metrics

        captured_preds = []

        def mock_compute(preds):
            captured_preds.extend(preds)
            return CaptionEditingScores(
                bleu_1=0, bleu_4=0, rouge_l=0, cider=0, spice=0
            )

        monkeypatch.setattr(
            metrics.CaptionMetrics, "compute", staticmethod(mock_compute)
        )

        samples = _make_samples(3, tmp_path)
        # Fail on the second sample's image path
        fail_path = samples[1].image_path
        editor = MockCaptionEditor(fail_on={fail_path})

        config = EvalConfig(model_name="test", output_dir=str(tmp_path / "out"))
        pipeline = EvalPipeline(config, caption_editor=editor)
        pipeline.run_caption_editing("coco_ce", samples)

        # The failed sample should have ref_cap as edited_cap
        failed_pred = [p for p in captured_preds if p.image_id == "img_0001"][0]
        assert failed_pred.edited_cap == failed_pred.ref_cap

        # The others should have uppercased ref_cap
        ok_pred = [p for p in captured_preds if p.image_id == "img_0000"][0]
        assert ok_pred.edited_cap == ok_pred.ref_cap.upper()


# ---------------------------------------------------------------------------
# POPE Evaluation
# ---------------------------------------------------------------------------


def _make_pope_data(
    n: int, tmp_path
) -> dict[tuple[POPEDomain, POPESetting], list[POPEQuestion]]:
    """Create a small POPE dataset with 1 domain×setting combo."""
    questions = [
        POPEQuestion(
            image_id=f"img_{i:04d}",
            question=f"Is there a cat in the image?",
            ground_truth="yes" if i % 2 == 0 else "no",
            image_path=str(tmp_path / f"img_{i}.jpg"),
            domain=POPEDomain.COCO,
            setting=POPESetting.RANDOM,
        )
        for i in range(n)
    ]
    return {(POPEDomain.COCO, POPESetting.RANDOM): questions}


class TestRunPope:
    def test_no_responder_raises(self, tmp_path):
        config = EvalConfig(model_name="test", output_dir=str(tmp_path / "out"))
        pipeline = EvalPipeline(config)
        with pytest.raises(RuntimeError, match="POPE_Responder"):
            pipeline.run_pope({})

    def test_basic_run(self, tmp_path):
        responder = MockPOPEResponder()
        config = EvalConfig(model_name="test", output_dir=str(tmp_path / "out"))
        pipeline = EvalPipeline(config, pope_responder=responder)

        pope_data = _make_pope_data(6, tmp_path)
        result = pipeline.run_pope(pope_data)

        assert (POPEDomain.COCO, POPESetting.RANDOM) in result
        scores = result[(POPEDomain.COCO, POPESetting.RANDOM)]
        assert isinstance(scores, POPEScores)
        assert responder.call_count == 6

        # Check CSV was saved
        csv_path = (
            tmp_path / "out" / "test_pope_coco_random_predictions.csv"
        )
        assert csv_path.exists()

        # Check JSON was saved
        json_path = tmp_path / "out" / "test_pope_scores.json"
        assert json_path.exists()

    def test_max_samples_limits_pope(self, tmp_path):
        responder = MockPOPEResponder()
        config = EvalConfig(
            model_name="test",
            output_dir=str(tmp_path / "out"),
            max_samples=2,
        )
        pipeline = EvalPipeline(config, pope_responder=responder)

        pope_data = _make_pope_data(10, tmp_path)
        pipeline.run_pope(pope_data)

        assert responder.call_count == 2

    def test_pope_error_fallback_records_no(self, tmp_path):
        pope_data = _make_pope_data(4, tmp_path)
        questions = pope_data[(POPEDomain.COCO, POPESetting.RANDOM)]
        fail_path = questions[1].image_path
        responder = MockPOPEResponder(fail_on={fail_path})

        config = EvalConfig(model_name="test", output_dir=str(tmp_path / "out"))
        pipeline = EvalPipeline(config, pope_responder=responder)
        result = pipeline.run_pope(pope_data)

        # Read the CSV to check the failed sample got "no"
        import pandas as pd

        csv_path = (
            tmp_path / "out" / "test_pope_coco_random_predictions.csv"
        )
        df = pd.read_csv(csv_path)
        failed_row = df[df["image_id"] == "img_0001"].iloc[0]
        assert failed_row["predicted"] == "no"

    def test_pope_scores_json_structure(self, tmp_path):
        responder = MockPOPEResponder()
        config = EvalConfig(model_name="test", output_dir=str(tmp_path / "out"))
        pipeline = EvalPipeline(config, pope_responder=responder)

        pope_data = _make_pope_data(4, tmp_path)
        pipeline.run_pope(pope_data)

        json_path = tmp_path / "out" / "test_pope_scores.json"
        with open(json_path) as f:
            data = json.load(f)

        assert "test" in data
        assert "coco_random" in data["test"]
        assert "accuracy" in data["test"]["coco_random"]
        assert "f1" in data["test"]["coco_random"]
