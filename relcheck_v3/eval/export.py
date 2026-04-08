"""Results export in paper-comparable format.

Produces Table 2 (Caption Editing) and Table 3 (POPE) formatted outputs
matching Kim et al.'s ICCV 2025 Workshop paper layout. Includes the paper's
reported baseline values for side-by-side comparison.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from relcheck_v3.eval.models import (
    CaptionEditingScores,
    POPEDomain,
    POPEScores,
    POPESetting,
)

# Paper's reported baseline values for Table 2 (Caption Editing)
TABLE2_BASELINES: dict[str, dict[str, CaptionEditingScores]] = {
    "Ref-Caps (paper)": {
        "COCO-CE": CaptionEditingScores(
            bleu_1=77.3, bleu_4=58.2, rouge_l=77.9, cider=507.6, spice=64.7
        ),
        "Flickr30K-CE": CaptionEditingScores(
            bleu_1=68.3, bleu_4=46.1, rouge_l=68.3, cider=345.1, spice=52.1
        ),
    },
    "LLaVA-1.5 (paper)": {
        "COCO-CE": CaptionEditingScores(
            bleu_1=72.1, bleu_4=48.2, rouge_l=72.3, cider=413.1, spice=55.1
        ),
        "Flickr30K-CE": CaptionEditingScores(
            bleu_1=63.2, bleu_4=37.2, rouge_l=63.2, cider=262.1, spice=42.3
        ),
    },
    "mPLUG-Owl2 (paper)": {
        "COCO-CE": CaptionEditingScores(
            bleu_1=72.3, bleu_4=48.5, rouge_l=72.5, cider=416.2, spice=55.4
        ),
        "Flickr30K-CE": CaptionEditingScores(
            bleu_1=63.5, bleu_4=37.5, rouge_l=63.5, cider=265.3, spice=42.7
        ),
    },
}

# Paper's reported baseline values for Table 3 (POPE)
TABLE3_BASELINES: dict[str, dict[tuple[POPEDomain, POPESetting], POPEScores]] = {
    "LLaVA-1.5 (paper)": {
        (POPEDomain.COCO, POPESetting.ADVERSARIAL): POPEScores(accuracy=83.0, f1=82.3),
        (POPEDomain.COCO, POPESetting.POPULAR): POPEScores(accuracy=85.0, f1=84.5),
        (POPEDomain.COCO, POPESetting.RANDOM): POPEScores(accuracy=87.0, f1=86.5),
        (POPEDomain.AOKVQA, POPESetting.ADVERSARIAL): POPEScores(accuracy=79.0, f1=78.2),
        (POPEDomain.AOKVQA, POPESetting.POPULAR): POPEScores(accuracy=81.0, f1=80.3),
        (POPEDomain.AOKVQA, POPESetting.RANDOM): POPEScores(accuracy=84.0, f1=83.4),
        (POPEDomain.GQA, POPESetting.ADVERSARIAL): POPEScores(accuracy=78.0, f1=77.1),
        (POPEDomain.GQA, POPESetting.POPULAR): POPEScores(accuracy=80.0, f1=79.2),
        (POPEDomain.GQA, POPESetting.RANDOM): POPEScores(accuracy=83.0, f1=82.3),
    },
}

# Canonical column order for Table 3
TABLE3_COLUMN_ORDER: list[tuple[POPEDomain, POPESetting]] = [
    (POPEDomain.COCO, POPESetting.ADVERSARIAL),
    (POPEDomain.COCO, POPESetting.POPULAR),
    (POPEDomain.COCO, POPESetting.RANDOM),
    (POPEDomain.AOKVQA, POPESetting.ADVERSARIAL),
    (POPEDomain.AOKVQA, POPESetting.POPULAR),
    (POPEDomain.AOKVQA, POPESetting.RANDOM),
    (POPEDomain.GQA, POPESetting.ADVERSARIAL),
    (POPEDomain.GQA, POPESetting.POPULAR),
    (POPEDomain.GQA, POPESetting.RANDOM),
]

# Metric column names for Table 2
TABLE2_METRICS = ["B-1", "B-4", "R", "C", "S"]

# Test set names for Table 2
TABLE2_TEST_SETS = ["COCO-CE", "Flickr30K-CE"]


def _scores_to_row(scores: CaptionEditingScores) -> list[float]:
    """Convert CaptionEditingScores to a list in Table 2 column order."""
    return [scores.bleu_1, scores.bleu_4, scores.rouge_l, scores.cider, scores.spice]


def _domain_setting_label(domain: POPEDomain, setting: POPESetting) -> str:
    """Create a human-readable column label for a domain-setting pair."""
    return f"{domain.value.upper()} {setting.value.capitalize()}"


class ResultsExporter:
    """Export evaluation results in paper-comparable Table 2 and Table 3 formats.

    Saves results as both CSV files (for programmatic use) and formatted text
    tables (for visual inspection). Includes the paper's reported baseline
    values for side-by-side comparison.
    """

    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_table2(
        self, results: dict[str, dict[str, CaptionEditingScores]]
    ) -> None:
        """Export caption editing metrics in Table 2 format.

        Args:
            results: Keyed by model_name -> test_set_name -> CaptionEditingScores.
        """
        # Merge paper baselines with computed results
        all_results: dict[str, dict[str, CaptionEditingScores]] = {}
        all_results.update(TABLE2_BASELINES)
        all_results.update(results)

        for test_set in TABLE2_TEST_SETS:
            rows: list[dict[str, object]] = []
            for model_name, test_sets in all_results.items():
                if test_set not in test_sets:
                    continue
                scores = test_sets[test_set]
                row: dict[str, object] = {"Model": model_name}
                values = _scores_to_row(scores)
                for col, val in zip(TABLE2_METRICS, values):
                    row[col] = val
                rows.append(row)

            df = pd.DataFrame(rows)
            if df.empty:
                continue

            # Save CSV
            csv_path = self.output_dir / f"table2_{test_set.lower().replace('-', '_')}.csv"
            df.to_csv(csv_path, index=False)

            # Save formatted text table
            txt_path = self.output_dir / f"table2_{test_set.lower().replace('-', '_')}.txt"
            txt_path.write_text(
                _format_text_table(f"Table 2 — {test_set}", df)
            )

    def export_table3(
        self,
        results: dict[str, dict[tuple[POPEDomain, POPESetting], POPEScores]],
    ) -> None:
        """Export POPE metrics in Table 3 format.

        Args:
            results: Keyed by model_name -> (domain, setting) -> POPEScores.
        """
        # Merge paper baselines with computed results
        all_results: dict[str, dict[tuple[POPEDomain, POPESetting], POPEScores]] = {}
        all_results.update(TABLE3_BASELINES)
        all_results.update(results)

        rows: list[dict[str, object]] = []
        for model_name, combos in all_results.items():
            row: dict[str, object] = {"Model": model_name}
            for domain, setting in TABLE3_COLUMN_ORDER:
                label = _domain_setting_label(domain, setting)
                if (domain, setting) in combos:
                    pope = combos[(domain, setting)]
                    row[f"{label} Acc"] = pope.accuracy
                    row[f"{label} F1"] = pope.f1
                else:
                    row[f"{label} Acc"] = ""
                    row[f"{label} F1"] = ""
            rows.append(row)

        df = pd.DataFrame(rows)
        if df.empty:
            return

        # Save CSV
        csv_path = self.output_dir / "table3_pope.csv"
        df.to_csv(csv_path, index=False)

        # Save formatted text table
        txt_path = self.output_dir / "table3_pope.txt"
        txt_path.write_text(_format_text_table("Table 3 — POPE", df))

    def export_aggregate_json(
        self,
        table2_results: dict[str, dict[str, CaptionEditingScores]] | None = None,
        table3_results: dict[str, dict[tuple[POPEDomain, POPESetting], POPEScores]] | None = None,
    ) -> None:
        """Save aggregate scores as a JSON file for programmatic use."""
        data: dict[str, object] = {}

        if table2_results:
            t2: dict[str, dict[str, dict[str, float]]] = {}
            for model, test_sets in table2_results.items():
                t2[model] = {}
                for ts_name, scores in test_sets.items():
                    t2[model][ts_name] = scores.model_dump()
            data["caption_editing"] = t2

        if table3_results:
            t3: dict[str, dict[str, dict[str, float]]] = {}
            for model, combos in table3_results.items():
                t3[model] = {}
                for (domain, setting), scores in combos.items():
                    key = f"{domain.value}_{setting.value}"
                    t3[model][key] = scores.model_dump()
            data["pope"] = t3

        json_path = self.output_dir / "aggregate_scores.json"
        json_path.write_text(json.dumps(data, indent=2))


def _format_text_table(title: str, df: pd.DataFrame) -> str:
    """Format a DataFrame as a readable text table with a title."""
    lines: list[str] = []
    lines.append(title)
    lines.append("=" * len(title))
    lines.append("")

    # Compute column widths
    col_widths: list[int] = []
    for col in df.columns:
        max_val_width = df[col].astype(str).str.len().max()
        col_widths.append(max(len(str(col)), max_val_width))

    # Header
    header = " | ".join(
        str(col).ljust(w) for col, w in zip(df.columns, col_widths)
    )
    lines.append(header)
    lines.append("-+-".join("-" * w for w in col_widths))

    # Rows
    for _, row in df.iterrows():
        line = " | ".join(
            str(row[col]).ljust(w) for col, w in zip(df.columns, col_widths)
        )
        lines.append(line)

    lines.append("")
    return "\n".join(lines)
