"""Harness metrics for POPE, MME, and AMBER benchmarks.

Implements official metric computation logic matching the reference
repositories for each benchmark:

- POPE: https://github.com/RUCAIBox/POPE (evaluate.py)
- MME: https://github.com/DAILtech/Evaluation-benchmark-MME (calculation.py)
- AMBER: https://github.com/junyangwang0410/AMBER (inference.py)

Uses ``sklearn.metrics`` for all classification metrics rather than
manual TP/FP counting, consistent with the project's "libraries over
hand-rolled code" design principle.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# POPE metrics — match RUCAIBox/POPE/evaluate.py exactly
# ---------------------------------------------------------------------------


def pope_extract_yesno(response: str) -> str:
    """Extract yes/no matching POPE official evaluate.py logic.

    The official POPE extraction:
    1. Take the first sentence (split on ``'.'``).
    2. Strip commas from the sentence.
    3. Split into words.
    4. If ``'No'``, ``'not'``, or ``'no'`` appears anywhere in the words,
       return ``"no"``; otherwise return ``"yes"``.

    Args:
        response: Raw model response string.

    Returns:
        ``"yes"`` or ``"no"``.
    """
    # First sentence
    first_sentence = response.split(".")[0]
    # Strip commas
    first_sentence = first_sentence.replace(",", "")
    # Split into words and check for negative indicators
    words = first_sentence.split()
    if "No" in words or "not" in words or "no" in words:
        return "no"
    return "yes"


def pope_metrics(predicted: list[str], ground_truth: list[str]) -> dict:
    """Compute accuracy, precision, recall, F1, yes-ratio per POPE official script.

    Uses ``sklearn.metrics`` with ``"yes"`` as the positive class
    (``pos_label=1`` after mapping ``yes → 1``, ``no → 0``).

    Args:
        predicted: List of predicted answers (``"yes"`` or ``"no"``).
        ground_truth: List of ground-truth answers (``"yes"`` or ``"no"``).

    Returns:
        Dict with keys: ``accuracy``, ``precision``, ``recall``, ``f1``,
        ``yes_ratio``.

    Raises:
        ValueError: If either list is empty or lengths differ.
    """
    if not predicted or not ground_truth:
        raise ValueError("Cannot compute metrics on empty lists.")
    if len(predicted) != len(ground_truth):
        raise ValueError(
            f"Predicted ({len(predicted)}) and ground_truth "
            f"({len(ground_truth)}) lists must have the same length."
        )

    label_map = {"yes": 1, "no": 0}
    y_pred = [label_map.get(p, 0) for p in predicted]
    y_true = [label_map.get(g, 0) for g in ground_truth]

    yes_count = sum(1 for p in predicted if p == "yes")

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "yes_ratio": yes_count / len(predicted),
    }


# ---------------------------------------------------------------------------
# MME metrics — match DAILtech/MME calculation.py exactly
# ---------------------------------------------------------------------------


def mme_extract_yesno(response: str) -> str:
    """Extract yes/no/other matching MME official calculation.py logic.

    The official MME extraction checks the first 4 characters of the
    lowercased response for ``'yes'`` or ``'no'``.  If neither is found,
    returns ``"other"``.

    Args:
        response: Raw model response string.

    Returns:
        ``"yes"``, ``"no"``, or ``"other"``.
    """
    prefix = response.lower()[:4]
    if "yes" in prefix:
        return "yes"
    if "no" in prefix:
        return "no"
    return "other"


def mme_metrics(predictions: list[dict]) -> dict:
    """Compute per-subtask accuracy, accuracy+, and score per MME official script.

    Each prediction dict must have keys:
    ``image_name``, ``question``, ``predicted``, ``ground_truth``, ``subtask``.

    Questions come in pairs per image (positive + negative).

    - ``accuracy``: standard accuracy via ``sklearn.metrics.accuracy_score``
    - ``accuracy_plus``: fraction of image pairs where BOTH questions correct
    - ``score``: ``(accuracy + accuracy_plus) × 100``

    Args:
        predictions: List of prediction dicts.

    Returns:
        Dict with keys: ``accuracy``, ``accuracy_plus``, ``score``.

    Raises:
        ValueError: If predictions list is empty.
    """
    if not predictions:
        raise ValueError("Cannot compute metrics on empty predictions list.")

    y_pred = [p["predicted"] for p in predictions]
    y_true = [p["ground_truth"] for p in predictions]

    # Standard accuracy across all questions
    acc = float(accuracy_score(y_true, y_pred))

    # accuracy+ : group by image_name, check if ALL questions for that
    # image are correct (typically a pair: positive + negative)
    image_results: dict[str, list[bool]] = defaultdict(list)
    for p in predictions:
        correct = p["predicted"] == p["ground_truth"]
        image_results[p["image_name"]].append(correct)

    if image_results:
        acc_plus = sum(
            1 for results in image_results.values() if all(results)
        ) / len(image_results)
    else:
        acc_plus = 0.0

    score = (acc + acc_plus) * 100

    return {
        "accuracy": acc,
        "accuracy_plus": acc_plus,
        "score": score,
    }


# ---------------------------------------------------------------------------
# AMBER metrics
# ---------------------------------------------------------------------------


def amber_discriminative_metrics(
    predicted: list[str], ground_truth: list[str]
) -> dict:
    """Compute accuracy, precision, recall, F1 with ``'no'`` as positive class.

    Uses ``sklearn.metrics`` with ``pos_label=0`` (mapping ``no → 0``,
    ``yes → 1``), matching AMBER's convention where ``"no"`` (i.e. the
    model correctly identifying a hallucination) is the positive class.

    Args:
        predicted: List of predicted answers (``"yes"`` or ``"no"``).
        ground_truth: List of ground-truth answers (``"yes"`` or ``"no"``).

    Returns:
        Dict with keys: ``accuracy``, ``precision``, ``recall``, ``f1``.

    Raises:
        ValueError: If either list is empty or lengths differ.
    """
    if not predicted or not ground_truth:
        raise ValueError("Cannot compute metrics on empty lists.")
    if len(predicted) != len(ground_truth):
        raise ValueError(
            f"Predicted ({len(predicted)}) and ground_truth "
            f"({len(ground_truth)}) lists must have the same length."
        )

    label_map = {"yes": 1, "no": 0}
    y_pred = [label_map.get(p, 0) for p in predicted]
    y_true = [label_map.get(g, 0) for g in ground_truth]

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
    }


def amber_generative_metrics(
    descriptions: list[str], amber_data: list[dict]
) -> dict:
    """Compute CHAIR, Cover, Hal, Cog using AMBER's spacy-based evaluation.

    Wrapping approach: attempts to import from a cloned AMBER repo
    (``/content/AMBER`` or ``./AMBER``).  If direct import fails, falls
    back to writing a temp JSON file and invoking AMBER's CLI.

    Args:
        descriptions: List of generated descriptions (one per image).
        amber_data: List of AMBER annotation dicts (from generative JSON).

    Returns:
        Dict with keys: ``chair``, ``cover``, ``hal``, ``cog``.

    Raises:
        ImportError: If spacy model ``en_core_web_lg`` is not available.
        ValueError: If descriptions list is empty.
    """
    if not descriptions:
        raise ValueError("Cannot compute metrics on empty descriptions list.")

    # Build inference data in AMBER's expected format
    inference_data = []
    for desc, ann in zip(descriptions, amber_data):
        entry = dict(ann)
        entry["response"] = desc
        inference_data.append(entry)

    # Try direct import from cloned AMBER repo
    amber_paths = [
        Path("/content/AMBER"),
        Path("./AMBER"),
        Path("../AMBER"),
    ]

    for amber_path in amber_paths:
        if (amber_path / "inference.py").exists():
            return _amber_generative_via_import(inference_data, amber_path)

    # Fall back to subprocess invocation
    for amber_path in amber_paths:
        if (amber_path / "inference.py").exists():
            return _amber_generative_via_subprocess(inference_data, amber_path)

    raise ImportError(
        "AMBER repository not found. Clone it with: "
        "git clone https://github.com/junyangwang0410/AMBER "
        "and ensure spacy en_core_web_lg is installed: "
        "python -m spacy download en_core_web_lg"
    )


def _amber_generative_via_import(
    inference_data: list[dict], amber_path: Path
) -> dict:
    """Compute AMBER generative metrics via direct import.

    Args:
        inference_data: List of dicts with AMBER annotation + ``response`` key.
        amber_path: Path to cloned AMBER repository.

    Returns:
        Dict with keys: ``chair``, ``cover``, ``hal``, ``cog``.

    Raises:
        ImportError: If import fails or spacy model is unavailable.
    """
    amber_str = str(amber_path)
    if amber_str not in sys.path:
        sys.path.insert(0, amber_str)

    try:
        # AMBER's inference.py contains the evaluation logic
        # Import and call the scoring functions directly
        import importlib

        spec = importlib.util.spec_from_file_location(
            "amber_inference", str(amber_path / "inference.py")
        )
        if spec is None or spec.loader is None:
            raise ImportError("Cannot load AMBER inference.py")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]

        # Write temp file for AMBER's expected input format
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(inference_data, f)
            tmp_path = f.name

        # Call AMBER's evaluation function
        if hasattr(module, "evaluate_generative"):
            results = module.evaluate_generative(tmp_path)
        elif hasattr(module, "main"):
            # Some versions use main() with args
            results = module.main(
                inference_data=tmp_path, evaluation_type="g"
            )
        else:
            raise ImportError(
                "AMBER inference.py does not expose expected evaluation functions. "
                "Falling back to subprocess."
            )

        Path(tmp_path).unlink(missing_ok=True)

        return {
            "chair": float(results.get("CHAIR", results.get("chair", 0.0))),
            "cover": float(results.get("Cover", results.get("cover", 0.0))),
            "hal": float(results.get("Hal", results.get("hal", 0.0))),
            "cog": float(results.get("Cog", results.get("cog", 0.0))),
        }

    except Exception as exc:
        logger.warning(
            "Direct AMBER import failed (%s), falling back to subprocess.", exc
        )
        return _amber_generative_via_subprocess(inference_data, amber_path)
    finally:
        if amber_str in sys.path:
            sys.path.remove(amber_str)


def _amber_generative_via_subprocess(
    inference_data: list[dict], amber_path: Path
) -> dict:
    """Compute AMBER generative metrics via subprocess invocation.

    Writes inference data to a temp JSON file, invokes AMBER's
    ``inference.py`` CLI, and parses stdout for metric values.

    Args:
        inference_data: List of dicts with AMBER annotation + ``response`` key.
        amber_path: Path to cloned AMBER repository.

    Returns:
        Dict with keys: ``chair``, ``cover``, ``hal``, ``cog``.

    Raises:
        ImportError: If AMBER CLI invocation fails.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(inference_data, f)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [
                sys.executable,
                str(amber_path / "inference.py"),
                "--inference_data",
                tmp_path,
                "--evaluation_type",
                "g",
            ],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )

        if result.returncode != 0:
            raise ImportError(
                f"AMBER CLI failed (exit {result.returncode}): {result.stderr}"
            )

        return _parse_amber_stdout(result.stdout)

    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _parse_amber_stdout(stdout: str) -> dict:
    """Parse AMBER CLI stdout for generative metric values.

    Args:
        stdout: Raw stdout from AMBER's inference.py.

    Returns:
        Dict with keys: ``chair``, ``cover``, ``hal``, ``cog``.
    """
    metrics: dict[str, float] = {
        "chair": 0.0,
        "cover": 0.0,
        "hal": 0.0,
        "cog": 0.0,
    }

    for line in stdout.splitlines():
        line_lower = line.lower().strip()
        for key in metrics:
            if key in line_lower and ":" in line:
                try:
                    value_str = line.split(":")[-1].strip()
                    metrics[key] = float(value_str)
                except (ValueError, IndexError):
                    logger.warning("Could not parse AMBER metric from line: %s", line)

    return metrics
