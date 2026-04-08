"""Tests for CEDataLoader and POPEDataLoader."""

import json
import logging

import pytest

from relcheck_v3.eval.data_loaders import CEDataLoader, POPEDataLoader
from relcheck_v3.eval.models import POPEDomain, POPESetting


@pytest.fixture
def loader():
    return CEDataLoader()


class TestCEDataLoaderLoad:
    """Tests for CEDataLoader.load()."""

    def test_loads_coco_ce_records(self, tmp_path, loader):
        """Valid COCO-CE entries with existing images produce CESamples."""
        image_dir = tmp_path / "images"
        image_dir.mkdir()

        entries = [
            {"image_id": 42, "gt_cap": "A cat on a mat", "ref_cap": "A dog on a mat"},
            {"image_id": 100, "gt_cap": "A park scene", "ref_cap": "A garden scene"},
        ]
        for e in entries:
            img_name = f"COCO_val2014_{int(e['image_id']):012d}.jpg"
            (image_dir / img_name).write_bytes(b"\xff")

        test_file = tmp_path / "coco_ce.json"
        test_file.write_text(json.dumps(entries))

        samples = loader.load(str(test_file), str(image_dir))

        assert len(samples) == 2
        assert samples[0].image_id == "42"
        assert samples[0].gt_cap == "A cat on a mat"
        assert samples[0].ref_cap == "A dog on a mat"
        assert samples[0].image_path == str(
            image_dir / "COCO_val2014_000000000042.jpg"
        )

    def test_loads_flickr_ce_records(self, tmp_path, loader):
        """Valid Flickr30K-CE entries with simple image names produce CESamples."""
        image_dir = tmp_path / "images"
        image_dir.mkdir()

        entries = [
            {"image_id": "12345", "gt_cap": "People walking", "ref_cap": "People running"},
        ]
        (image_dir / "12345.jpg").write_bytes(b"\xff")

        test_file = tmp_path / "flickr_ce.json"
        test_file.write_text(json.dumps(entries))

        samples = loader.load(str(test_file), str(image_dir))

        assert len(samples) == 1
        assert samples[0].image_id == "12345"
        assert samples[0].image_path == str(image_dir / "12345.jpg")

    def test_explicit_file_name_field(self, tmp_path, loader):
        """Entries with a file_name field use that for image path resolution."""
        image_dir = tmp_path / "images"
        image_dir.mkdir()

        entries = [
            {
                "image_id": "abc",
                "gt_cap": "A sunset",
                "ref_cap": "A sunrise",
                "file_name": "sunset_001.jpg",
            },
        ]
        (image_dir / "sunset_001.jpg").write_bytes(b"\xff")

        test_file = tmp_path / "test.json"
        test_file.write_text(json.dumps(entries))

        samples = loader.load(str(test_file), str(image_dir))

        assert len(samples) == 1
        assert samples[0].image_path == str(image_dir / "sunset_001.jpg")

    def test_explicit_image_path_field(self, tmp_path, loader):
        """Entries with an image_path field use that directly."""
        image_dir = tmp_path / "images"
        image_dir.mkdir()

        img_file = image_dir / "custom.jpg"
        img_file.write_bytes(b"\xff")

        entries = [
            {
                "image_id": "x",
                "gt_cap": "Hello",
                "ref_cap": "World",
                "image_path": "custom.jpg",
            },
        ]

        test_file = tmp_path / "test.json"
        test_file.write_text(json.dumps(entries))

        samples = loader.load(str(test_file), str(image_dir))

        assert len(samples) == 1
        assert samples[0].image_path == str(image_dir / "custom.jpg")

    def test_image_id_converted_to_string(self, tmp_path, loader):
        """Integer image_id is converted to string in the output."""
        image_dir = tmp_path / "images"
        image_dir.mkdir()

        entries = [{"image_id": 7, "gt_cap": "test", "ref_cap": "test2"}]
        (image_dir / "COCO_val2014_000000000007.jpg").write_bytes(b"\xff")

        test_file = tmp_path / "test.json"
        test_file.write_text(json.dumps(entries))

        samples = loader.load(str(test_file), str(image_dir))
        assert samples[0].image_id == "7"
        assert isinstance(samples[0].image_id, str)


class TestCEDataLoaderErrors:
    """Tests for error handling in CEDataLoader."""

    def test_missing_test_set_file_raises(self, loader):
        """FileNotFoundError raised when test set file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="not_here.json"):
            loader.load("not_here.json", "/tmp")

    def test_missing_image_skipped_with_warning(self, tmp_path, loader, caplog):
        """Records with missing image files are skipped with a warning."""
        image_dir = tmp_path / "images"
        image_dir.mkdir()

        entries = [
            {"image_id": 1, "gt_cap": "exists", "ref_cap": "exists too"},
            {"image_id": 2, "gt_cap": "missing", "ref_cap": "missing too"},
        ]
        (image_dir / "COCO_val2014_000000000001.jpg").write_bytes(b"\xff")

        test_file = tmp_path / "test.json"
        test_file.write_text(json.dumps(entries))

        with caplog.at_level(logging.WARNING):
            samples = loader.load(str(test_file), str(image_dir))

        assert len(samples) == 1
        assert samples[0].image_id == "1"
        assert "Image file not found" in caplog.text

    def test_empty_gt_cap_skipped_with_warning(self, tmp_path, loader, caplog):
        """Records with empty GT-Cap are skipped with a warning."""
        image_dir = tmp_path / "images"
        image_dir.mkdir()

        entries = [
            {"image_id": "1", "gt_cap": "", "ref_cap": "some ref"},
            {"image_id": "2", "gt_cap": "valid", "ref_cap": "valid ref"},
        ]
        (image_dir / "1.jpg").write_bytes(b"\xff")
        (image_dir / "2.jpg").write_bytes(b"\xff")

        test_file = tmp_path / "test.json"
        test_file.write_text(json.dumps(entries))

        with caplog.at_level(logging.WARNING):
            samples = loader.load(str(test_file), str(image_dir))

        assert len(samples) == 1
        assert samples[0].image_id == "2"
        assert "Empty GT-Cap" in caplog.text

    def test_empty_ref_cap_skipped_with_warning(self, tmp_path, loader, caplog):
        """Records with empty Ref-Cap are skipped with a warning."""
        image_dir = tmp_path / "images"
        image_dir.mkdir()

        entries = [
            {"image_id": "1", "gt_cap": "valid gt", "ref_cap": ""},
        ]
        (image_dir / "1.jpg").write_bytes(b"\xff")

        test_file = tmp_path / "test.json"
        test_file.write_text(json.dumps(entries))

        with caplog.at_level(logging.WARNING):
            samples = loader.load(str(test_file), str(image_dir))

        assert len(samples) == 0
        assert "Empty Ref-Cap" in caplog.text

    def test_whitespace_only_captions_skipped(self, tmp_path, loader, caplog):
        """Records with whitespace-only captions are treated as empty."""
        image_dir = tmp_path / "images"
        image_dir.mkdir()

        entries = [
            {"image_id": "1", "gt_cap": "   ", "ref_cap": "valid"},
        ]
        (image_dir / "1.jpg").write_bytes(b"\xff")

        test_file = tmp_path / "test.json"
        test_file.write_text(json.dumps(entries))

        with caplog.at_level(logging.WARNING):
            samples = loader.load(str(test_file), str(image_dir))

        assert len(samples) == 0
        assert "Empty GT-Cap" in caplog.text

    def test_empty_test_set_returns_empty(self, tmp_path, loader):
        """Empty JSON array returns empty list."""
        test_file = tmp_path / "empty.json"
        test_file.write_text("[]")

        samples = loader.load(str(test_file), str(tmp_path))
        assert samples == []

    def test_all_images_missing_returns_empty(self, tmp_path, loader):
        """When no images exist, an empty list is returned."""
        image_dir = tmp_path / "images"
        image_dir.mkdir()

        entries = [{"image_id": "99", "gt_cap": "no image", "ref_cap": "no image ref"}]
        test_file = tmp_path / "test.json"
        test_file.write_text(json.dumps(entries))

        samples = loader.load(str(test_file), str(image_dir))
        assert samples == []


# ---------------------------------------------------------------------------
# POPEDataLoader tests
# ---------------------------------------------------------------------------

def _make_pope_line(image: str, text: str, label: str) -> str:
    """Create a single POPE JSON line."""
    return json.dumps({"image": image, "text": text, "label": label})


def _create_pope_files(pope_dir, domains=None, settings=None, num_questions=2):
    """Create POPE question files for all domain×setting combos.

    Returns the image_dirs dict mapping each domain to its image directory.
    """
    if domains is None:
        domains = list(POPEDomain)
    if settings is None:
        settings = list(POPESetting)

    image_dirs: dict[POPEDomain, str] = {}
    for domain in domains:
        img_dir = pope_dir / f"{domain.value}_images"
        img_dir.mkdir(exist_ok=True)
        image_dirs[domain] = str(img_dir)

        for setting in settings:
            filename = f"{domain.value}_pope_{setting.value}.json"
            lines = []
            for i in range(num_questions):
                img_name = f"{domain.value}_img_{i}.jpg"
                (img_dir / img_name).write_bytes(b"\xff")
                label = "yes" if i % 2 == 0 else "no"
                lines.append(
                    _make_pope_line(
                        img_name,
                        f"Is there a cat in the image?",
                        label,
                    )
                )
            (pope_dir / filename).write_text("\n".join(lines))

    return image_dirs


@pytest.fixture
def pope_loader():
    return POPEDataLoader()


class TestPOPEDataLoaderLoad:
    """Tests for POPEDataLoader.load()."""

    def test_loads_all_9_combinations(self, tmp_path, pope_loader):
        """All 9 domain×setting combos are loaded."""
        image_dirs = _create_pope_files(tmp_path)
        result = pope_loader.load(str(tmp_path), image_dirs)

        assert len(result) == 9
        for domain in POPEDomain:
            for setting in POPESetting:
                assert (domain, setting) in result

    def test_parses_question_fields(self, tmp_path, pope_loader):
        """Each question has correct image_id, question, ground_truth, domain, setting."""
        image_dirs = _create_pope_files(tmp_path, num_questions=1)
        result = pope_loader.load(str(tmp_path), image_dirs)

        questions = result[(POPEDomain.COCO, POPESetting.RANDOM)]
        assert len(questions) == 1
        q = questions[0]
        assert q.image_id == "coco_img_0"
        assert q.question == "Is there a cat in the image?"
        assert q.ground_truth == "yes"
        assert q.domain == POPEDomain.COCO
        assert q.setting == POPESetting.RANDOM

    def test_resolves_image_path_with_domain_dir(self, tmp_path, pope_loader):
        """Image paths are resolved using the per-domain image directory."""
        image_dirs = _create_pope_files(tmp_path, num_questions=1)
        result = pope_loader.load(str(tmp_path), image_dirs)

        q = result[(POPEDomain.GQA, POPESetting.ADVERSARIAL)][0]
        expected = str(tmp_path / "gqa_images" / "gqa_img_0.jpg")
        assert q.image_path == expected

    def test_multiple_questions_per_file(self, tmp_path, pope_loader):
        """Multiple questions in a single file are all loaded."""
        image_dirs = _create_pope_files(tmp_path, num_questions=5)
        result = pope_loader.load(str(tmp_path), image_dirs)

        for key, questions in result.items():
            assert len(questions) == 5

    def test_label_case_insensitive(self, tmp_path, pope_loader):
        """Labels like 'Yes', 'NO' are normalized to lowercase."""
        img_dir = tmp_path / "coco_images"
        img_dir.mkdir()
        (img_dir / "img.jpg").write_bytes(b"\xff")

        image_dirs = _create_pope_files(tmp_path, num_questions=0)
        # Overwrite one file with mixed-case labels
        lines = [
            _make_pope_line("img.jpg", "Is there a dog?", "Yes"),
            _make_pope_line("img.jpg", "Is there a cat?", "NO"),
        ]
        (tmp_path / "coco_pope_random.json").write_text("\n".join(lines))

        result = pope_loader.load(str(tmp_path), image_dirs)
        questions = result[(POPEDomain.COCO, POPESetting.RANDOM)]
        assert len(questions) == 2
        assert questions[0].ground_truth == "yes"
        assert questions[1].ground_truth == "no"

    def test_image_id_strips_extension(self, tmp_path, pope_loader):
        """image_id is the filename without extension."""
        image_dirs = _create_pope_files(tmp_path, num_questions=0)
        img_dir = tmp_path / "coco_images"
        (img_dir / "COCO_val2014_000000123456.jpg").write_bytes(b"\xff")

        line = _make_pope_line(
            "COCO_val2014_000000123456.jpg",
            "Is there a person?",
            "yes",
        )
        (tmp_path / "coco_pope_random.json").write_text(line)

        result = pope_loader.load(str(tmp_path), image_dirs)
        q = result[(POPEDomain.COCO, POPESetting.RANDOM)][0]
        assert q.image_id == "COCO_val2014_000000123456"


class TestPOPEDataLoaderErrors:
    """Tests for error handling in POPEDataLoader."""

    def test_missing_question_file_raises(self, tmp_path, pope_loader):
        """FileNotFoundError raised when a question file is missing."""
        image_dirs = {d: str(tmp_path) for d in POPEDomain}
        with pytest.raises(FileNotFoundError, match="coco_pope_random.json"):
            pope_loader.load(str(tmp_path), image_dirs)

    def test_missing_file_message_contains_path(self, tmp_path, pope_loader):
        """The error message includes the full expected file path."""
        image_dirs = {d: str(tmp_path) for d in POPEDomain}
        with pytest.raises(FileNotFoundError) as exc_info:
            pope_loader.load(str(tmp_path), image_dirs)
        assert str(tmp_path) in str(exc_info.value)

    def test_malformed_json_line_skipped(self, tmp_path, pope_loader, caplog):
        """Malformed JSON lines are skipped with a warning."""
        image_dirs = _create_pope_files(tmp_path, num_questions=0)
        img_dir = tmp_path / "coco_images"
        (img_dir / "good.jpg").write_bytes(b"\xff")

        lines = [
            "not valid json",
            _make_pope_line("good.jpg", "Is there a cat?", "yes"),
        ]
        (tmp_path / "coco_pope_random.json").write_text("\n".join(lines))

        with caplog.at_level(logging.WARNING):
            result = pope_loader.load(str(tmp_path), image_dirs)

        questions = result[(POPEDomain.COCO, POPESetting.RANDOM)]
        assert len(questions) == 1
        assert "malformed JSON" in caplog.text

    def test_invalid_label_skipped(self, tmp_path, pope_loader, caplog):
        """Records with labels other than yes/no are skipped."""
        image_dirs = _create_pope_files(tmp_path, num_questions=0)
        img_dir = tmp_path / "coco_images"
        (img_dir / "img.jpg").write_bytes(b"\xff")

        lines = [
            _make_pope_line("img.jpg", "Is there a cat?", "maybe"),
            _make_pope_line("img.jpg", "Is there a dog?", "yes"),
        ]
        (tmp_path / "coco_pope_random.json").write_text("\n".join(lines))

        with caplog.at_level(logging.WARNING):
            result = pope_loader.load(str(tmp_path), image_dirs)

        questions = result[(POPEDomain.COCO, POPESetting.RANDOM)]
        assert len(questions) == 1
        assert questions[0].ground_truth == "yes"

    def test_empty_question_file_returns_empty_list(self, tmp_path, pope_loader):
        """An empty question file produces an empty list for that combo."""
        image_dirs = _create_pope_files(tmp_path, num_questions=0)
        result = pope_loader.load(str(tmp_path), image_dirs)

        for key, questions in result.items():
            assert questions == []

    def test_blank_lines_ignored(self, tmp_path, pope_loader):
        """Blank lines in question files are silently skipped."""
        image_dirs = _create_pope_files(tmp_path, num_questions=0)
        img_dir = tmp_path / "coco_images"
        (img_dir / "img.jpg").write_bytes(b"\xff")

        lines = [
            "",
            _make_pope_line("img.jpg", "Is there a cat?", "yes"),
            "",
            "",
        ]
        (tmp_path / "coco_pope_random.json").write_text("\n".join(lines))

        result = pope_loader.load(str(tmp_path), image_dirs)
        questions = result[(POPEDomain.COCO, POPESetting.RANDOM)]
        assert len(questions) == 1
