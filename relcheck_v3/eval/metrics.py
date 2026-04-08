"""Metrics computation for caption editing and POPE evaluation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from relcheck_v3.eval.models import CEPrediction

from relcheck_v3.eval.models import CaptionEditingScores, POPEScores

logger = logging.getLogger(__name__)


class CaptionMetrics:
    """Wraps pycocoevalcap to compute caption editing metrics.

    Computes BLEU-1, BLEU-4, ROUGE-L, CIDEr, and SPICE.
    All scores are scaled to the paper's convention (×100).
    """

    @staticmethod
    def _check_imports() -> None:
        """Raise ImportError with install instructions if pycocoevalcap is missing."""
        try:
            import pycocoevalcap  # noqa: F401
        except ImportError:
            raise ImportError(
                "pycocoevalcap is required for caption editing metrics. "
                "Install it with: pip install pycocoevalcap"
            )

    @staticmethod
    def format_inputs(
        predictions: list[CEPrediction],
    ) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        """Convert predictions to the COCO eval API format.

        Args:
            predictions: List of CEPrediction objects.

        Returns:
            Tuple of (references, hypotheses) dicts, each mapping
            sample id to a list containing the caption string.
        """
        refs: dict[str, list[str]] = {}
        hyps: dict[str, list[str]] = {}
        for pred in predictions:
            refs[pred.image_id] = [pred.gt_cap]
            hyps[pred.image_id] = [pred.edited_cap]
        return refs, hyps

    @staticmethod
    def compute(predictions: list[CEPrediction]) -> CaptionEditingScores:
        """Compute all caption editing metrics.

        Args:
            predictions: List of CEPrediction objects with gt_cap and edited_cap.

        Returns:
            CaptionEditingScores with all metrics scaled to paper convention (×100).

        Raises:
            ImportError: If pycocoevalcap is not installed.
            ValueError: If predictions list is empty.
        """
        CaptionMetrics._check_imports()

        if not predictions:
            raise ValueError("Cannot compute metrics on empty predictions list.")

        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.spice.spice import Spice

        refs, hyps = CaptionMetrics.format_inputs(predictions)

        # Each scorer returns (aggregate_score, per_image_scores)
        # Bleu returns lists of 4 scores [bleu1, bleu2, bleu3, bleu4]
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            (Spice(), "SPICE"),
        ]

        raw_scores: dict[str, float] = {}
        for scorer, method in scorers:
            try:
                score, _ = scorer.compute_score(refs, hyps)
            except Exception as e:
                method_name = method if isinstance(method, str) else method[0]
                logger.warning(
                    "Scorer %s failed: %s. Setting score to 0.", method_name, e
                )
                if isinstance(method, list):
                    for m in method:
                        raw_scores[m] = 0.0
                else:
                    raw_scores[method] = 0.0
                continue

            if isinstance(method, list):
                # Bleu returns a list of scores for each n-gram level
                for m, s in zip(method, score):
                    raw_scores[m] = s
            else:
                raw_scores[method] = score

        # Scale all scores ×100 to match paper convention
        # Raw BLEU/ROUGE/SPICE are 0-1, CIDEr is 0-10 scale
        # All get multiplied by 100
        return CaptionEditingScores(
            bleu_1=raw_scores["Bleu_1"] * 100,
            bleu_4=raw_scores["Bleu_4"] * 100,
            rouge_l=raw_scores["ROUGE_L"] * 100,
            cider=raw_scores["CIDEr"] * 100,
            spice=raw_scores["SPICE"] * 100,
        )


class POPEMetrics:
    """Computes binary Accuracy and F1 for POPE evaluation.

    "yes" is the positive class for precision/recall/F1 computation.
    Handles edge cases (all-same-class predictions) without division-by-zero.
    """

    @staticmethod
    def compute(predicted: list[str], ground_truth: list[str]) -> POPEScores:
        """Compute POPE Accuracy and F1.

        Args:
            predicted: List of predicted answers ("yes" or "no").
            ground_truth: List of ground-truth answers ("yes" or "no").

        Returns:
            POPEScores with accuracy and f1 as percentages (0–100).

        Raises:
            ValueError: If lists are empty or have different lengths.
        """
        if not predicted or not ground_truth:
            raise ValueError("Cannot compute metrics on empty lists.")
        if len(predicted) != len(ground_truth):
            raise ValueError(
                f"Predicted ({len(predicted)}) and ground_truth ({len(ground_truth)}) "
                "lists must have the same length."
            )

        total = len(predicted)
        correct = sum(p == g for p, g in zip(predicted, ground_truth))
        accuracy = (correct / total) * 100

        # "yes" is the positive class
        tp = sum(p == "yes" and g == "yes" for p, g in zip(predicted, ground_truth))
        fp = sum(p == "yes" and g == "no" for p, g in zip(predicted, ground_truth))
        fn = sum(p == "no" and g == "yes" for p, g in zip(predicted, ground_truth))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if (precision + recall) > 0:
            f1 = (2 * precision * recall / (precision + recall)) * 100
        else:
            f1 = 0.0

        return POPEScores(accuracy=accuracy, f1=f1)
