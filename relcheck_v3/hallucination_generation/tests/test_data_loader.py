"""Tests for DataLoader."""

import json
import os
import logging

import pytest

from relcheck_v3.hallucination_generation.data_loader import DataLoader


@pytest.fixture
def loader():
    return DataLoader()


class TestCocoEE:
    """Tests for COCO-EE annotation loading."""

    def test_loads_coco_records(self, tmp_path, loader):
        """Valid COCO-EE annotations with existing images produce CaptionRecords."""
        image_dir = tmp_path / "images"
        image_dir.mkdir()

        annotations = [
            {"image_id": 42, "caption": "A cat on a mat"},
            {"image_id": 100, "caption": "A dog in a park"},
        ]
        for ann in annotations:
            img_name = f"COCO_val2014_{ann['image_id']:012d}.jpg"
            (image_dir / img_name).write_bytes(b"\xff")

        ann_file = tmp_path / "annotations.json"
        ann_file.write_text(json.dumps(annotations))

        records = loader.load("coco-ee", str(ann_file), str(image_dir))

        assert len(records) == 2
        assert records[0].image_id == "42"
        assert records[0].caption == "A cat on a mat"
        assert records[0].image_path == str(image_dir / "COCO_val2014_000000000042.jpg")
        assert records[1].image_id == "100"

    def test_image_id_converted_to_string(self, tmp_path, loader):
        """Integer image_id in COCO annotations is converted to string."""
        image_dir = tmp_path / "images"
        image_dir.mkdir()

        annotations = [{"image_id": 7, "caption": "test"}]
        (image_dir / "COCO_val2014_000000000007.jpg").write_bytes(b"\xff")

        ann_file = tmp_path / "annotations.json"
        ann_file.write_text(json.dumps(annotations))

        records = loader.load("coco-ee", str(ann_file), str(image_dir))
        assert records[0].image_id == "7"
        assert isinstance(records[0].image_id, str)


class TestFlickr30kEE:
    """Tests for Flickr30K-EE annotation loading."""

    def test_loads_flickr_records(self, tmp_path, loader):
        """Valid Flickr30K-EE annotations with existing images produce CaptionRecords."""
        image_dir = tmp_path / "images"
        image_dir.mkdir()

        annotations = [
            {"image_id": "12345", "caption": "People walking"},
            {"image_id": "67890", "caption": "A sunset"},
        ]
        for ann in annotations:
            (image_dir / f"{ann['image_id']}.jpg").write_bytes(b"\xff")

        ann_file = tmp_path / "annotations.json"
        ann_file.write_text(json.dumps(annotations))

        records = loader.load("flickr30k-ee", str(ann_file), str(image_dir))

        assert len(records) == 2
        assert records[0].image_id == "12345"
        assert records[0].image_path == str(image_dir / "12345.jpg")


class TestErrorHandling:
    """Tests for error conditions."""

    def test_missing_annotation_file_raises(self, loader):
        """FileNotFoundError raised when annotation file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="not_here.json"):
            loader.load("coco-ee", "not_here.json", "/tmp")

    def test_unsupported_dataset_raises(self, tmp_path, loader):
        """ValueError raised for unsupported dataset name."""
        ann_file = tmp_path / "ann.json"
        ann_file.write_text("[]")

        with pytest.raises(ValueError, match="Unsupported dataset"):
            loader.load("imagenet", str(ann_file), str(tmp_path))

    def test_missing_image_skipped_with_warning(self, tmp_path, loader, caplog):
        """Records with missing image files are skipped and a warning is logged."""
        image_dir = tmp_path / "images"
        image_dir.mkdir()

        annotations = [
            {"image_id": 1, "caption": "exists"},
            {"image_id": 2, "caption": "missing"},
        ]
        # Only create image for id=1
        (image_dir / "COCO_val2014_000000000001.jpg").write_bytes(b"\xff")

        ann_file = tmp_path / "annotations.json"
        ann_file.write_text(json.dumps(annotations))

        with caplog.at_level(logging.WARNING):
            records = loader.load("coco-ee", str(ann_file), str(image_dir))

        assert len(records) == 1
        assert records[0].image_id == "1"
        assert "COCO_val2014_000000000002" in caplog.text

    def test_all_images_missing_returns_empty(self, tmp_path, loader):
        """When no images exist, an empty list is returned."""
        image_dir = tmp_path / "images"
        image_dir.mkdir()

        annotations = [{"image_id": 99, "caption": "no image"}]
        ann_file = tmp_path / "annotations.json"
        ann_file.write_text(json.dumps(annotations))

        records = loader.load("coco-ee", str(ann_file), str(image_dir))
        assert records == []

    def test_empty_annotation_file(self, tmp_path, loader):
        """Empty annotation list returns empty records."""
        ann_file = tmp_path / "annotations.json"
        ann_file.write_text("[]")

        records = loader.load("coco-ee", str(ann_file), str(tmp_path))
        assert records == []
