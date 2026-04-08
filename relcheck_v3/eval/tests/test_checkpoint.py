"""Unit tests for CheckpointManager."""

import json
import os

import pytest

from relcheck_v3.eval.checkpoint import CheckpointManager
from relcheck_v3.eval.models import CheckpointData


class TestCheckpointManagerInit:
    def test_checkpoint_path_format(self, tmp_path):
        mgr = CheckpointManager(str(tmp_path), "llava", "coco_ce")
        expected = os.path.join(str(tmp_path), "llava_coco_ce_checkpoint.json")
        assert mgr.get_checkpoint_path() == expected

    def test_custom_interval(self, tmp_path):
        mgr = CheckpointManager(str(tmp_path), "m", "t", interval=100)
        assert mgr.should_save(100)
        assert not mgr.should_save(99)


class TestCheckpointLoad:
    def test_load_no_file_returns_empty(self, tmp_path):
        mgr = CheckpointManager(str(tmp_path), "llava", "coco_ce")
        assert mgr.load() == {}

    def test_load_valid_checkpoint(self, tmp_path):
        data = CheckpointData(
            model_name="llava",
            test_set_name="coco_ce",
            predictions={"img_001": "a cat on a mat", "img_002": "a dog in a park"},
        )
        path = tmp_path / "llava_coco_ce_checkpoint.json"
        path.write_text(json.dumps(data.model_dump()))

        mgr = CheckpointManager(str(tmp_path), "llava", "coco_ce")
        preds = mgr.load()
        assert preds == {"img_001": "a cat on a mat", "img_002": "a dog in a park"}

    def test_load_corrupted_json_returns_empty(self, tmp_path):
        path = tmp_path / "llava_coco_ce_checkpoint.json"
        path.write_text("not valid json {{{")

        mgr = CheckpointManager(str(tmp_path), "llava", "coco_ce")
        preds = mgr.load()
        assert preds == {}

    def test_load_wrong_structure_returns_empty(self, tmp_path):
        path = tmp_path / "llava_coco_ce_checkpoint.json"
        path.write_text(json.dumps({"foo": "bar"}))

        mgr = CheckpointManager(str(tmp_path), "llava", "coco_ce")
        preds = mgr.load()
        assert preds == {}

    def test_load_mismatched_keys_returns_empty(self, tmp_path):
        data = CheckpointData(
            model_name="mplug",
            test_set_name="flickr_ce",
            predictions={"img_001": "hello"},
        )
        path = tmp_path / "llava_coco_ce_checkpoint.json"
        path.write_text(json.dumps(data.model_dump()))

        mgr = CheckpointManager(str(tmp_path), "llava", "coco_ce")
        preds = mgr.load()
        assert preds == {}


class TestCheckpointSave:
    def test_save_creates_file(self, tmp_path):
        mgr = CheckpointManager(str(tmp_path), "llava", "coco_ce")
        mgr.save({"img_001": "edited caption"})

        path = tmp_path / "llava_coco_ce_checkpoint.json"
        assert path.exists()

        raw = json.loads(path.read_text())
        assert raw["model_name"] == "llava"
        assert raw["test_set_name"] == "coco_ce"
        assert raw["predictions"] == {"img_001": "edited caption"}

    def test_save_creates_directory(self, tmp_path):
        nested = tmp_path / "sub" / "dir"
        mgr = CheckpointManager(str(nested), "llava", "coco_ce")
        mgr.save({"img_001": "cap"})
        assert (nested / "llava_coco_ce_checkpoint.json").exists()

    def test_save_overwrites_existing(self, tmp_path):
        mgr = CheckpointManager(str(tmp_path), "llava", "coco_ce")
        mgr.save({"img_001": "first"})
        mgr.save({"img_001": "first", "img_002": "second"})

        raw = json.loads(
            (tmp_path / "llava_coco_ce_checkpoint.json").read_text()
        )
        assert len(raw["predictions"]) == 2


class TestCheckpointRoundTrip:
    def test_save_then_load(self, tmp_path):
        predictions = {f"img_{i:04d}": f"caption {i}" for i in range(50)}
        mgr = CheckpointManager(str(tmp_path), "llava", "coco_ce")
        mgr.save(predictions)

        mgr2 = CheckpointManager(str(tmp_path), "llava", "coco_ce")
        loaded = mgr2.load()
        assert loaded == predictions


class TestShouldSave:
    def test_zero_returns_false(self, tmp_path):
        mgr = CheckpointManager(str(tmp_path), "m", "t", interval=500)
        assert not mgr.should_save(0)

    def test_multiples_return_true(self, tmp_path):
        mgr = CheckpointManager(str(tmp_path), "m", "t", interval=500)
        assert mgr.should_save(500)
        assert mgr.should_save(1000)
        assert mgr.should_save(1500)

    def test_non_multiples_return_false(self, tmp_path):
        mgr = CheckpointManager(str(tmp_path), "m", "t", interval=500)
        assert not mgr.should_save(1)
        assert not mgr.should_save(499)
        assert not mgr.should_save(501)
