"""Evaluation pipeline orchestration."""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING

import pandas as pd
from tqdm import tqdm

from relcheck_v3.eval.checkpoint import CheckpointManager
from relcheck_v3.eval.config import EvalConfig
from relcheck_v3.eval.metrics import CaptionMetrics, POPEMetrics
from relcheck_v3.eval.models import (
    CaptionEditingScores,
    CEPrediction,
    POPEDomain,
    POPEPrediction,
    POPEScores,
    POPESetting,
)

if TYPE_CHECKING:
    from relcheck_v3.eval.interfaces import Caption_Editor, POPE_Responder
    from relcheck_v3.eval.models import CESample, POPEQuestion

logger = logging.getLogger(__name__)


def _extract_yes_no(response: str) -> str:
    """Extract a yes/no answer from a model response.

    Takes the first whitespace-delimited token, lowercases it, strips
    trailing punctuation, and checks if it is "yes" or "no".
    If neither, logs a warning and returns "no".
    """
    raw = response.strip().split()[0].lower() if response.strip() else ""
    # Strip common trailing punctuation (commas, periods, etc.)
    token = raw.rstrip(".,!?;:")
    if token in ("yes", "no"):
        return token
    logger.warning(
        "Could not extract yes/no from response (first token=%r). "
        "Falling back to 'no'.",
        token,
    )
    return "no"


class EvalPipeline:
    """Top-level orchestrator for caption editing and POPE evaluation.

    Accepts an EvalConfig and optional pluggable Caption_Editor / POPE_Responder.
    Validates configured paths before any processing begins.
    Uses CheckpointManager for resumable caption editing inference.
    Falls back to Ref-Cap (CE) or "no" (POPE) on per-sample errors.
    Saves per-sample predictions to CSV and aggregate scores to JSON.
    """

    def __init__(
        self,
        config: EvalConfig,
        caption_editor: Caption_Editor | None = None,
        pope_responder: POPE_Responder | None = None,
    ) -> None:
        self._config = config
        self._caption_editor = caption_editor
        self._pope_responder = pope_responder

        self._validate_paths()
        os.makedirs(config.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Path validation
    # ------------------------------------------------------------------

    def _validate_paths(self) -> None:
        """Check that all configured file/directory paths exist.

        Collects every missing path and raises a single ValueError listing
        them all (Req 13.3).
        """
        missing: list[str] = []

        path_fields = [
            ("coco_ce_path", self._config.coco_ce_path),
            ("flickr_ce_path", self._config.flickr_ce_path),
            ("pope_data_dir", self._config.pope_data_dir),
            ("coco_image_dir", self._config.coco_image_dir),
            ("flickr_image_dir", self._config.flickr_image_dir),
            ("aokvqa_image_dir", self._config.aokvqa_image_dir),
            ("gqa_image_dir", self._config.gqa_image_dir),
        ]

        for name, path in path_fields:
            if path and not os.path.exists(path):
                missing.append(path)

        if missing:
            paths_str = ", ".join(missing)
            raise ValueError(
                f"Configuration error: the following paths do not exist: {paths_str}"
            )

    # ------------------------------------------------------------------
    # Caption Editing
    # ------------------------------------------------------------------

    def run_caption_editing(
        self,
        test_set_name: str,
        samples: list[CESample],
    ) -> CaptionEditingScores:
        """Run caption editing evaluation on a list of CE samples.

        Args:
            test_set_name: Identifier for this test set (e.g. "coco_ce").
            samples: List of CESample objects to evaluate.

        Returns:
            CaptionEditingScores with aggregate metrics.

        Raises:
            RuntimeError: If no Caption_Editor was provided.
        """
        if self._caption_editor is None:
            raise RuntimeError(
                "Cannot run caption editing without a Caption_Editor."
            )

        # Apply max_samples limit (Req 13.2, 13.4)
        if self._config.max_samples is not None:
            samples = samples[: self._config.max_samples]

        # Set up checkpoint manager
        ckpt = CheckpointManager(
            checkpoint_dir=os.path.join(self._config.output_dir, "checkpoints"),
            model_name=self._config.model_name,
            test_set_name=test_set_name,
            interval=self._config.checkpoint_interval,
        )
        existing_preds = ckpt.load()

        predictions: dict[str, str] = dict(existing_preds)
        processed_count = len(existing_preds)

        for sample in tqdm(samples, desc=f"Caption editing ({test_set_name})"):
            # Skip already-checkpointed samples
            if sample.image_id in predictions:
                continue

            try:
                edited = self._caption_editor.edit_caption(
                    sample.image_path, sample.ref_cap
                )
            except Exception as exc:
                logger.error(
                    "Caption_Editor failed for %s: %s. "
                    "Using Ref-Cap as fallback.",
                    sample.image_id,
                    exc,
                )
                edited = sample.ref_cap

            predictions[sample.image_id] = edited
            processed_count += 1

            if ckpt.should_save(processed_count):
                ckpt.save(predictions)

        # Final checkpoint save
        ckpt.save(predictions)

        # Build CEPrediction list for metrics
        ce_predictions: list[CEPrediction] = []
        for sample in samples:
            edited_cap = predictions.get(sample.image_id, sample.ref_cap)
            ce_predictions.append(
                CEPrediction(
                    image_id=sample.image_id,
                    ref_cap=sample.ref_cap,
                    edited_cap=edited_cap,
                    gt_cap=sample.gt_cap,
                )
            )

        # Compute metrics
        scores = CaptionMetrics.compute(ce_predictions)

        # Save per-sample predictions to CSV (Req 9.3)
        self._save_ce_predictions_csv(test_set_name, ce_predictions)

        # Save aggregate scores to JSON (Req 9.4)
        self._save_ce_scores_json(test_set_name, scores)

        return scores

    # ------------------------------------------------------------------
    # POPE Evaluation
    # ------------------------------------------------------------------

    def run_pope(
        self,
        pope_data: dict[tuple[POPEDomain, POPESetting], list[POPEQuestion]],
    ) -> dict[tuple[POPEDomain, POPESetting], POPEScores]:
        """Run POPE evaluation across all domain×setting combinations.

        Args:
            pope_data: Dict mapping (domain, setting) to list of POPEQuestion.

        Returns:
            Dict mapping (domain, setting) to POPEScores.

        Raises:
            RuntimeError: If no POPE_Responder was provided.
        """
        if self._pope_responder is None:
            raise RuntimeError(
                "Cannot run POPE evaluation without a POPE_Responder."
            )

        all_scores: dict[tuple[POPEDomain, POPESetting], POPEScores] = {}

        for (domain, setting), questions in pope_data.items():
            # Apply max_samples limit (Req 13.2, 13.4)
            if self._config.max_samples is not None:
                questions = questions[: self._config.max_samples]

            predicted: list[str] = []
            ground_truth: list[str] = []
            pope_predictions: list[POPEPrediction] = []

            desc = f"POPE ({domain.value}/{setting.value})"
            for q in tqdm(questions, desc=desc):
                try:
                    raw_response = self._pope_responder.answer_pope(
                        q.image_path, q.question
                    )
                    answer = _extract_yes_no(raw_response)
                except Exception as exc:
                    logger.error(
                        "POPE_Responder failed for %s (%s): %s. "
                        "Recording 'no' as fallback.",
                        q.image_id,
                        q.question,
                        exc,
                    )
                    answer = "no"

                predicted.append(answer)
                ground_truth.append(q.ground_truth)
                pope_predictions.append(
                    POPEPrediction(
                        image_id=q.image_id,
                        question=q.question,
                        predicted=answer,
                        ground_truth=q.ground_truth,
                    )
                )

            # Compute metrics for this combo
            scores = POPEMetrics.compute(predicted, ground_truth)
            all_scores[(domain, setting)] = scores

            # Save per-question predictions to CSV (Req 10.3)
            self._save_pope_predictions_csv(domain, setting, pope_predictions)

        # Save aggregate POPE scores to JSON (Req 10.4)
        self._save_pope_scores_json(all_scores)

        return all_scores

    # ------------------------------------------------------------------
    # CSV / JSON export helpers
    # ------------------------------------------------------------------

    def _save_ce_predictions_csv(
        self,
        test_set_name: str,
        predictions: list[CEPrediction],
    ) -> None:
        """Save per-sample caption editing predictions to CSV."""
        rows = [p.model_dump() for p in predictions]
        df = pd.DataFrame(rows)
        path = os.path.join(
            self._config.output_dir,
            f"{self._config.model_name}_{test_set_name}_predictions.csv",
        )
        df.to_csv(path, index=False)
        logger.info("Saved %d CE predictions to %s", len(rows), path)

    def _save_ce_scores_json(
        self,
        test_set_name: str,
        scores: CaptionEditingScores,
    ) -> None:
        """Save aggregate caption editing scores to JSON."""
        payload = {
            self._config.model_name: {
                test_set_name: scores.model_dump(),
            }
        }
        path = os.path.join(
            self._config.output_dir,
            f"{self._config.model_name}_{test_set_name}_scores.json",
        )
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info("Saved CE scores to %s", path)

    def _save_pope_predictions_csv(
        self,
        domain: POPEDomain,
        setting: POPESetting,
        predictions: list[POPEPrediction],
    ) -> None:
        """Save per-question POPE predictions to CSV for one domain×setting."""
        rows = [p.model_dump() for p in predictions]
        df = pd.DataFrame(rows)
        path = os.path.join(
            self._config.output_dir,
            f"{self._config.model_name}_pope_{domain.value}_{setting.value}_predictions.csv",
        )
        df.to_csv(path, index=False)
        logger.info(
            "Saved %d POPE predictions (%s/%s) to %s",
            len(rows),
            domain.value,
            setting.value,
            path,
        )

    def _save_pope_scores_json(
        self,
        all_scores: dict[tuple[POPEDomain, POPESetting], POPEScores],
    ) -> None:
        """Save aggregate POPE scores for all combos to a single JSON."""
        payload: dict[str, dict[str, dict[str, float]]] = {}
        for (domain, setting), scores in all_scores.items():
            key = f"{domain.value}_{setting.value}"
            payload.setdefault(self._config.model_name, {})[key] = (
                scores.model_dump()
            )

        path = os.path.join(
            self._config.output_dir,
            f"{self._config.model_name}_pope_scores.json",
        )
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info("Saved POPE scores to %s", path)
