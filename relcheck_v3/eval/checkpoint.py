"""Checkpoint manager for resumable inference."""

import json
import logging
import os

from relcheck_v3.eval.models import CheckpointData

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Saves and loads prediction checkpoints for resumable inference.

    Predictions are saved to a JSON checkpoint file every N samples (configurable).
    On restart, existing predictions are loaded so already-processed samples are skipped.
    Keyed by (model_name, test_set_name). Corrupted checkpoint files are handled
    gracefully — a warning is logged and inference starts fresh.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        model_name: str,
        test_set_name: str,
        interval: int = 500,
    ) -> None:
        self._checkpoint_dir = checkpoint_dir
        self._model_name = model_name
        self._test_set_name = test_set_name
        self._interval = interval

        filename = f"{model_name}_{test_set_name}_checkpoint.json"
        self._checkpoint_path = os.path.join(checkpoint_dir, filename)

    def load(self) -> dict[str, str]:
        """Load existing predictions from checkpoint file.

        Returns an empty dict if no checkpoint exists or the file is corrupted.
        """
        if not os.path.exists(self._checkpoint_path):
            return {}

        try:
            with open(self._checkpoint_path, "r") as f:
                raw = json.load(f)

            data = CheckpointData(**raw)

            # Verify the checkpoint belongs to this model/test_set combo
            if (
                data.model_name != self._model_name
                or data.test_set_name != self._test_set_name
            ):
                logger.warning(
                    "Checkpoint file %s has mismatched keys "
                    "(expected model=%s test_set=%s, got model=%s test_set=%s). "
                    "Starting fresh.",
                    self._checkpoint_path,
                    self._model_name,
                    self._test_set_name,
                    data.model_name,
                    data.test_set_name,
                )
                return {}

            logger.info(
                "Loaded %d predictions from checkpoint %s",
                len(data.predictions),
                self._checkpoint_path,
            )
            return data.predictions

        except (json.JSONDecodeError, TypeError, KeyError, ValueError) as exc:
            logger.warning(
                "Corrupted checkpoint file %s (%s). Starting fresh.",
                self._checkpoint_path,
                exc,
            )
            return {}

    def save(self, predictions: dict[str, str]) -> None:
        """Save predictions to the checkpoint file."""
        os.makedirs(self._checkpoint_dir, exist_ok=True)

        data = CheckpointData(
            model_name=self._model_name,
            test_set_name=self._test_set_name,
            predictions=predictions,
        )

        with open(self._checkpoint_path, "w") as f:
            json.dump(data.model_dump(), f)

        logger.debug(
            "Saved %d predictions to checkpoint %s",
            len(predictions),
            self._checkpoint_path,
        )

    def should_save(self, count: int) -> bool:
        """Return True if count is a multiple of the checkpoint interval."""
        return count > 0 and count % self._interval == 0

    def get_checkpoint_path(self) -> str:
        """Return the checkpoint file path."""
        return self._checkpoint_path
