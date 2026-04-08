"""Tests for the CLI/Colab entry point (run.py)."""

import json
import os
import tempfile

import pytest

from relcheck_v3.hallucination_generation.run import _parse_args, main


class TestParseArgs:
    """Test CLI argument parsing."""

    def test_required_args(self):
        args = _parse_args(["--annotation-path", "/tmp/ann.json", "--image-dir", "/tmp/imgs"])
        assert args.annotation_path == "/tmp/ann.json"
        assert args.image_dir == "/tmp/imgs"

    def test_defaults(self):
        args = _parse_args(["--annotation-path", "a.json", "--image-dir", "imgs"])
        assert args.dataset_name == "coco-ee"
        assert args.openai_api_key == ""
        assert args.output_dir == "relcheck_v3/output"
        assert args.max_samples is None
        assert args.dry_run is False
        assert args.max_retries == 3

    def test_all_args(self):
        args = _parse_args([
            "--dataset-name", "flickr30k-ee",
            "--annotation-path", "/data/ann.json",
            "--image-dir", "/data/images",
            "--openai-api-key", "sk-test",
            "--output-dir", "/out",
            "--max-samples", "50",
            "--dry-run",
            "--max-retries", "5",
        ])
        assert args.dataset_name == "flickr30k-ee"
        assert args.annotation_path == "/data/ann.json"
        assert args.image_dir == "/data/images"
        assert args.openai_api_key == "sk-test"
        assert args.output_dir == "/out"
        assert args.max_samples == 50
        assert args.dry_run is True
        assert args.max_retries == 5

    def test_missing_required_args(self):
        with pytest.raises(SystemExit):
            _parse_args([])


class TestMainKwargs:
    """Test main() called with keyword arguments (Colab usage)."""

    def test_dry_run_with_kwargs(self, tmp_path):
        """main() should accept kwargs and run the pipeline in dry-run mode."""
        # Create a minimal annotation file and image
        ann_file = tmp_path / "annotations.json"
        img_dir = tmp_path / "images"
        img_dir.mkdir()

        image_id = 42
        img_file = img_dir / f"COCO_val2014_{image_id:012d}.jpg"
        img_file.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        ann_file.write_text(json.dumps([
            {"image_id": image_id, "caption": "a cat on a mat"}
        ]))

        output_dir = tmp_path / "output"

        main(
            annotation_path=str(ann_file),
            image_dir=str(img_dir),
            output_dir=str(output_dir),
            dry_run=True,
            max_samples=1,
        )

        # Verify output was created
        assert (output_dir / "output.jsonl").exists()

    def test_api_key_from_env(self, tmp_path, monkeypatch):
        """main() should read OPENAI_API_KEY from env when not provided."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")

        ann_file = tmp_path / "annotations.json"
        img_dir = tmp_path / "images"
        img_dir.mkdir()

        image_id = 1
        img_file = img_dir / f"COCO_val2014_{image_id:012d}.jpg"
        img_file.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        ann_file.write_text(json.dumps([
            {"image_id": image_id, "caption": "a dog"}
        ]))

        output_dir = tmp_path / "output"

        # dry_run=True so we don't actually call OpenAI
        main(
            annotation_path=str(ann_file),
            image_dir=str(img_dir),
            output_dir=str(output_dir),
            dry_run=True,
        )

        assert (output_dir / "output.jsonl").exists()
