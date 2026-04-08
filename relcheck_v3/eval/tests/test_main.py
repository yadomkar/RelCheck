"""Tests for relcheck_v3.eval.__main__ — CLI parsing and main() wiring."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from relcheck_v3.eval.__main__ import _build_model_wrappers, _parse_args, main
from relcheck_v3.eval.config import EvalConfig
from relcheck_v3.eval.models import EvalType


# ── CLI argument parsing ──────────────────────────────────────────────


class TestParseArgs:
    """Verify _parse_args produces correct EvalConfig from CLI tokens."""

    def test_minimal_args(self) -> None:
        config = _parse_args(["--model", "passthrough"])
        assert config.model_name == "passthrough"
        assert config.eval_type == EvalType.BOTH
        assert config.output_dir == "relcheck_v3/output/eval"
        assert config.max_samples is None
        assert config.checkpoint_interval == 500

    def test_all_args(self, tmp_path: object) -> None:
        config = _parse_args([
            "--model", "llava-1.5",
            "--eval-type", "caption-editing",
            "--coco-ce-path", "/data/coco_ce.json",
            "--flickr-ce-path", "/data/flickr_ce.json",
            "--pope-data-dir", "/data/pope",
            "--coco-image-dir", "/imgs/coco",
            "--flickr-image-dir", "/imgs/flickr",
            "--aokvqa-image-dir", "/imgs/aokvqa",
            "--gqa-image-dir", "/imgs/gqa",
            "--output-dir", "/out",
            "--max-samples", "100",
            "--checkpoint-interval", "250",
        ])
        assert config.model_name == "llava-1.5"
        assert config.eval_type == EvalType.CAPTION_EDITING
        assert config.coco_ce_path == "/data/coco_ce.json"
        assert config.flickr_ce_path == "/data/flickr_ce.json"
        assert config.pope_data_dir == "/data/pope"
        assert config.coco_image_dir == "/imgs/coco"
        assert config.flickr_image_dir == "/imgs/flickr"
        assert config.aokvqa_image_dir == "/imgs/aokvqa"
        assert config.gqa_image_dir == "/imgs/gqa"
        assert config.output_dir == "/out"
        assert config.max_samples == 100
        assert config.checkpoint_interval == 250

    def test_pope_eval_type(self) -> None:
        config = _parse_args(["--model", "mplug-owl2", "--eval-type", "pope"])
        assert config.eval_type == EvalType.POPE

    def test_missing_model_exits(self) -> None:
        with pytest.raises(SystemExit):
            _parse_args([])

    def test_invalid_model_exits(self) -> None:
        with pytest.raises(SystemExit):
            _parse_args(["--model", "gpt-4"])


# ── Model wrapper dispatch ────────────────────────────────────────────


class TestBuildModelWrappers:
    """Verify _build_model_wrappers dispatches correctly."""

    def test_passthrough(self) -> None:
        from relcheck_v3.eval.baselines.passthrough import PassthroughCaptionEditor

        editor, responder = _build_model_wrappers("passthrough")
        assert isinstance(editor, PassthroughCaptionEditor)
        assert responder is None

    def test_unknown_model_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown model"):
            _build_model_wrappers("gpt-4")


# ── End-to-end main() with passthrough ────────────────────────────────


class TestMainPassthrough:
    """Run main() with passthrough model on tiny synthetic data."""

    def test_caption_editing_end_to_end(self, tmp_path: object) -> None:
        """Wire data → passthrough editor → pipeline → export."""
        import pathlib

        tmp = pathlib.Path(str(tmp_path))

        # Create a tiny CE test set JSON
        img_dir = tmp / "images"
        img_dir.mkdir()
        # Create a dummy image file
        img_file = img_dir / "img_001.jpg"
        img_file.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        ce_data = [
            {
                "image_id": "img_001",
                "gt_cap": "A cat sitting on a mat.",
                "ref_cap": "A dog sitting on a mat.",
                "image_path": str(img_file),
            }
        ]
        ce_path = tmp / "coco_ce.json"
        ce_path.write_text(json.dumps(ce_data))

        output_dir = tmp / "output"

        config = EvalConfig(
            model_name="passthrough",
            eval_type=EvalType.CAPTION_EDITING,
            coco_ce_path=str(ce_path),
            coco_image_dir=str(img_dir),
            output_dir=str(output_dir),
            max_samples=10,
            checkpoint_interval=100,
        )

        results = main(config)

        assert "caption_editing" in results
        assert "COCO-CE" in results["caption_editing"]
        scores = results["caption_editing"]["COCO-CE"]
        # Passthrough returns ref_cap unchanged, so scores should be computed
        assert hasattr(scores, "bleu_1")

        # Verify output files were created
        assert output_dir.exists()

    def test_pope_skipped_for_passthrough(self, tmp_path: object) -> None:
        """Passthrough has no POPE responder — POPE track should be skipped."""
        import pathlib

        tmp = pathlib.Path(str(tmp_path))
        output_dir = tmp / "output"

        config = EvalConfig(
            model_name="passthrough",
            eval_type=EvalType.POPE,
            pope_data_dir="",
            output_dir=str(output_dir),
        )

        results = main(config)
        # No POPE results since passthrough has no responder
        assert "pope" not in results
