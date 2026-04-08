"""Tests for ResultsExporter — Table 2 and Table 3 formatted outputs."""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from relcheck_v3.eval.export import (
    ResultsExporter,
    TABLE2_BASELINES,
    TABLE3_BASELINES,
    TABLE2_METRICS,
    TABLE3_COLUMN_ORDER,
    _domain_setting_label,
)
from relcheck_v3.eval.models import (
    CaptionEditingScores,
    POPEDomain,
    POPEScores,
    POPESetting,
)


@pytest.fixture
def tmp_output_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def exporter(tmp_output_dir):
    return ResultsExporter(tmp_output_dir)


@pytest.fixture
def sample_t2_results():
    return {
        "TestModel": {
            "COCO-CE": CaptionEditingScores(
                bleu_1=75.0, bleu_4=55.0, rouge_l=76.0, cider=490.0, spice=62.0
            ),
            "Flickr30K-CE": CaptionEditingScores(
                bleu_1=66.0, bleu_4=44.0, rouge_l=66.0, cider=330.0, spice=50.0
            ),
        }
    }


@pytest.fixture
def sample_t3_results():
    return {
        "TestModel": {
            (d, s): POPEScores(accuracy=80.0 + i, f1=79.0 + i)
            for i, (d, s) in enumerate(TABLE3_COLUMN_ORDER)
        }
    }


class TestTable2Export:
    """Tests for export_table2."""

    def test_creates_csv_and_txt_for_both_test_sets(self, exporter, sample_t2_results, tmp_output_dir):
        exporter.export_table2(sample_t2_results)
        for name in ["table2_coco_ce", "table2_flickr30k_ce"]:
            assert (Path(tmp_output_dir) / f"{name}.csv").exists()
            assert (Path(tmp_output_dir) / f"{name}.txt").exists()

    def test_csv_has_correct_columns(self, exporter, sample_t2_results, tmp_output_dir):
        exporter.export_table2(sample_t2_results)
        df = pd.read_csv(Path(tmp_output_dir) / "table2_coco_ce.csv")
        assert list(df.columns) == ["Model"] + TABLE2_METRICS

    def test_includes_paper_baselines(self, exporter, sample_t2_results, tmp_output_dir):
        exporter.export_table2(sample_t2_results)
        df = pd.read_csv(Path(tmp_output_dir) / "table2_coco_ce.csv")
        models = list(df["Model"])
        assert "Ref-Caps (paper)" in models
        assert "LLaVA-1.5 (paper)" in models
        assert "mPLUG-Owl2 (paper)" in models

    def test_includes_computed_results(self, exporter, sample_t2_results, tmp_output_dir):
        exporter.export_table2(sample_t2_results)
        df = pd.read_csv(Path(tmp_output_dir) / "table2_coco_ce.csv")
        assert "TestModel" in list(df["Model"])

    def test_baseline_values_match_paper(self, exporter, tmp_output_dir):
        exporter.export_table2({})
        df = pd.read_csv(Path(tmp_output_dir) / "table2_coco_ce.csv")
        ref_row = df[df["Model"] == "Ref-Caps (paper)"].iloc[0]
        assert ref_row["B-1"] == 77.3
        assert ref_row["B-4"] == 58.2
        assert ref_row["C"] == 507.6

    def test_txt_contains_title(self, exporter, sample_t2_results, tmp_output_dir):
        exporter.export_table2(sample_t2_results)
        txt = (Path(tmp_output_dir) / "table2_coco_ce.txt").read_text()
        assert "Table 2" in txt
        assert "COCO-CE" in txt

    def test_empty_results_still_exports_baselines(self, exporter, tmp_output_dir):
        exporter.export_table2({})
        df = pd.read_csv(Path(tmp_output_dir) / "table2_coco_ce.csv")
        assert len(df) == 3  # 3 paper baselines


class TestTable3Export:
    """Tests for export_table3."""

    def test_creates_csv_and_txt(self, exporter, sample_t3_results, tmp_output_dir):
        exporter.export_table3(sample_t3_results)
        assert (Path(tmp_output_dir) / "table3_pope.csv").exists()
        assert (Path(tmp_output_dir) / "table3_pope.txt").exists()

    def test_csv_has_acc_and_f1_for_all_9_combos(self, exporter, sample_t3_results, tmp_output_dir):
        exporter.export_table3(sample_t3_results)
        df = pd.read_csv(Path(tmp_output_dir) / "table3_pope.csv")
        for domain, setting in TABLE3_COLUMN_ORDER:
            label = _domain_setting_label(domain, setting)
            assert f"{label} Acc" in df.columns
            assert f"{label} F1" in df.columns

    def test_includes_paper_baselines(self, exporter, sample_t3_results, tmp_output_dir):
        exporter.export_table3(sample_t3_results)
        df = pd.read_csv(Path(tmp_output_dir) / "table3_pope.csv")
        assert "LLaVA-1.5 (paper)" in list(df["Model"])

    def test_includes_computed_results(self, exporter, sample_t3_results, tmp_output_dir):
        exporter.export_table3(sample_t3_results)
        df = pd.read_csv(Path(tmp_output_dir) / "table3_pope.csv")
        assert "TestModel" in list(df["Model"])

    def test_txt_contains_title(self, exporter, sample_t3_results, tmp_output_dir):
        exporter.export_table3(sample_t3_results)
        txt = (Path(tmp_output_dir) / "table3_pope.txt").read_text()
        assert "Table 3" in txt
        assert "POPE" in txt

    def test_empty_results_still_exports_baselines(self, exporter, tmp_output_dir):
        exporter.export_table3({})
        df = pd.read_csv(Path(tmp_output_dir) / "table3_pope.csv")
        assert len(df) == 1  # 1 paper baseline (LLaVA-1.5)


class TestAggregateJson:
    """Tests for export_aggregate_json."""

    def test_creates_json_file(self, exporter, sample_t2_results, tmp_output_dir):
        exporter.export_aggregate_json(table2_results=sample_t2_results)
        assert (Path(tmp_output_dir) / "aggregate_scores.json").exists()

    def test_json_contains_caption_editing_key(self, exporter, sample_t2_results, tmp_output_dir):
        exporter.export_aggregate_json(table2_results=sample_t2_results)
        data = json.loads((Path(tmp_output_dir) / "aggregate_scores.json").read_text())
        assert "caption_editing" in data
        assert "TestModel" in data["caption_editing"]

    def test_json_contains_pope_key(self, exporter, sample_t3_results, tmp_output_dir):
        exporter.export_aggregate_json(table3_results=sample_t3_results)
        data = json.loads((Path(tmp_output_dir) / "aggregate_scores.json").read_text())
        assert "pope" in data
        assert "TestModel" in data["pope"]

    def test_json_score_values_match(self, exporter, sample_t2_results, tmp_output_dir):
        exporter.export_aggregate_json(table2_results=sample_t2_results)
        data = json.loads((Path(tmp_output_dir) / "aggregate_scores.json").read_text())
        coco = data["caption_editing"]["TestModel"]["COCO-CE"]
        assert coco["bleu_1"] == 75.0
        assert coco["cider"] == 490.0


class TestOutputDirCreation:
    """Tests for output directory handling."""

    def test_creates_nested_output_dir(self):
        with tempfile.TemporaryDirectory() as base:
            nested = str(Path(base) / "a" / "b" / "c")
            exporter = ResultsExporter(nested)
            assert Path(nested).is_dir()
