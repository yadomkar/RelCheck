"""Evaluation runner for the multi-benchmark evaluation harness.

Orchestrates the full evaluation pipeline: load benchmark → run MLLM →
run correction → extract answers → compute metrics → save results JSON.

Exposed both as a Python API (``run_eval()``) and as a CLI entry point::

    python -m relcheck_v3.eval.eval_runner \\
        --benchmark pope --system full \\
        --mllm llava-hf/llava-1.5-7b-hf --corrector gpt-5.4

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 12.4, 12.6, 13.1, 13.2, 13.3
"""

from __future__ import annotations

import argparse
import logging
import os
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from relcheck_v3.benchmarks.amber import AMBERLoader
from relcheck_v3.benchmarks.mme import MMELoader
from relcheck_v3.benchmarks.models import BenchmarkSample
from relcheck_v3.benchmarks.pope import POPELoader
from relcheck_v3.eval.answer_extractor import AnswerExtractor
from relcheck_v3.eval.harness_metrics import (
    amber_discriminative_metrics,
    amber_generative_metrics,
    mme_extract_yesno,
    mme_metrics,
    pope_extract_yesno,
    pope_metrics,
)
from relcheck_v3.eval.results_aggregator import RunResult, SplitResult
from relcheck_v3.mllm.cache import InferenceCache
from relcheck_v3.mllm.wrapper import VALID_MODEL_IDS, MLLMWrapper
from relcheck_v3.systems.base import CorrectionSystem
from relcheck_v3.systems.raw_mllm import RawMLLM

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Valid model identifiers
# ---------------------------------------------------------------------------

VALID_CORRECTOR_MODELS: set[str] = {
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "gpt-3.5-turbo",
}

_VALID_BENCHMARKS: set[str] = {"pope", "mme", "amber"}
_VALID_SYSTEMS: set[str] = {"raw", "woodpecker", "claim", "claim+geom", "full"}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_model_names(mllm: str, corrector: str) -> None:
    """Raise ``ValueError`` if either model name is unknown.

    Args:
        mllm: MLLM model identifier.
        corrector: Corrector model identifier.

    Raises:
        ValueError: With a message listing valid options.
    """
    if mllm not in VALID_MODEL_IDS:
        valid = ", ".join(sorted(VALID_MODEL_IDS))
        raise ValueError(
            f"Unknown MLLM model {mllm!r}. Valid options: {valid}"
        )
    if corrector not in VALID_CORRECTOR_MODELS:
        valid = ", ".join(sorted(VALID_CORRECTOR_MODELS))
        raise ValueError(
            f"Unknown corrector model {corrector!r}. Valid options: {valid}"
        )


# ---------------------------------------------------------------------------
# Benchmark loading
# ---------------------------------------------------------------------------


def _load_samples(
    benchmark: str,
    data_dir: str,
    image_dir: str,
    coco_instances_path: str | None,
) -> list[BenchmarkSample]:
    """Load all samples for the specified benchmark.

    Args:
        benchmark: One of ``"pope"``, ``"mme"``, ``"amber"``.
        data_dir: Root directory containing benchmark data files.
        image_dir: Directory containing benchmark images.
        coco_instances_path: Path to COCO instances JSON (POPE only).

    Returns:
        List of :class:`BenchmarkSample` objects.
    """
    if benchmark == "pope":
        loader = POPELoader()
        return list(
            loader.iter_samples(data_dir, image_dir, coco_instances_path)
        )
    if benchmark == "mme":
        loader_mme = MMELoader()
        return list(loader_mme.iter_samples(data_dir))
    if benchmark == "amber":
        loader_amber = AMBERLoader()
        return list(loader_amber.iter_samples(data_dir, image_dir))

    raise ValueError(f"Unknown benchmark: {benchmark!r}")


# ---------------------------------------------------------------------------
# System instantiation
# ---------------------------------------------------------------------------


def _build_system(
    system: str,
    corrector: str,
    cache_dir: str,
    coco_instances_path: str | None,
) -> CorrectionSystem:
    """Instantiate the requested correction system.

    Args:
        system: One of ``"raw"``, ``"woodpecker"``, ``"claim"``,
            ``"claim+geom"``, ``"full"``.
        corrector: Model identifier for the correction LLM.
        cache_dir: Root cache directory.
        coco_instances_path: COCO instances path (used by ``"full"``
            variant for RelTR).

    Returns:
        A :class:`CorrectionSystem` instance.
    """
    openai_key = os.environ.get("OPENAI_API_KEY", "")

    if system == "raw":
        return RawMLLM()

    if system == "woodpecker":
        from relcheck_v3.systems.woodpecker import WoodpeckerBaseline
        return WoodpeckerBaseline(
            openai_api_key=openai_key,
            corrector_model=corrector,
            cache_dir=str(Path(cache_dir) / "systems" / "woodpecker"),
        )

    if system == "claim":
        from relcheck_v3.systems.relcheck_claim import RelCheckClaimOnly
        return RelCheckClaimOnly(
            openai_api_key=openai_key,
            corrector_model=corrector,
            cache_dir=str(Path(cache_dir) / "systems" / "claim"),
        )

    if system == "claim+geom":
        from relcheck_v3.systems.relcheck_claim_geom import RelCheckClaimGeom
        return RelCheckClaimGeom(
            openai_api_key=openai_key,
            corrector_model=corrector,
            cache_dir=str(Path(cache_dir) / "systems" / "claim_geom"),
        )

    if system == "full":
        from relcheck_v3.systems.relcheck_full import RelCheckFull
        return RelCheckFull(
            openai_api_key=openai_key,
            corrector_model=corrector,
            cache_dir=str(Path(cache_dir) / "systems" / "full"),
        )

    raise ValueError(f"Unknown system: {system!r}")


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------


def _compute_metrics(
    benchmark: str,
    system_id: str,
    samples: list[BenchmarkSample],
    predictions: dict[str, str],
) -> list[SplitResult]:
    """Compute per-split metrics for the given benchmark.

    Args:
        benchmark: Benchmark name.
        system_id: System identifier.
        samples: All benchmark samples.
        predictions: Mapping from ``sample_id`` to predicted answer or
            description.

    Returns:
        List of :class:`SplitResult` objects, one per split.
    """
    # Group samples by split.
    by_split: dict[str, list[BenchmarkSample]] = defaultdict(list)
    for s in samples:
        by_split[s.split].append(s)

    results: list[SplitResult] = []

    for split, split_samples in sorted(by_split.items()):
        n = len(split_samples)

        if benchmark == "pope":
            predicted = [
                pope_extract_yesno(predictions.get(s.sample_id, ""))
                for s in split_samples
            ]
            ground_truth = [s.label for s in split_samples]
            m = pope_metrics(predicted, ground_truth)
            results.append(
                SplitResult(
                    split=split,
                    benchmark=benchmark,
                    system_id=system_id,
                    n_samples=n,
                    accuracy=m["accuracy"],
                    precision=m["precision"],
                    recall=m["recall"],
                    f1=m["f1"],
                    yes_ratio=m["yes_ratio"],
                )
            )

        elif benchmark == "mme":
            preds_list: list[dict] = []
            for s in split_samples:
                pred_raw = predictions.get(s.sample_id, "")
                pred_yn = mme_extract_yesno(pred_raw)
                preds_list.append(
                    {
                        "image_name": s.metadata.get("image_name", s.sample_id),
                        "question": s.question,
                        "predicted": pred_yn,
                        "ground_truth": s.label,
                        "subtask": split,
                    }
                )
            m = mme_metrics(preds_list)
            results.append(
                SplitResult(
                    split=split,
                    benchmark=benchmark,
                    system_id=system_id,
                    n_samples=n,
                    accuracy=m["accuracy"],
                    accuracy_plus=m["accuracy_plus"],
                )
            )

        elif benchmark == "amber":
            if split == "g":
                # Generative — collect descriptions.
                descs = [
                    predictions.get(s.sample_id, "") for s in split_samples
                ]
                amber_data = [
                    {"id": s.metadata.get("amber_id", s.sample_id)}
                    for s in split_samples
                ]
                try:
                    m = amber_generative_metrics(descs, amber_data)
                except Exception:
                    logger.warning(
                        "AMBER generative metrics failed — skipping split 'g'",
                        exc_info=True,
                    )
                    m = {}
                results.append(
                    SplitResult(
                        split=split,
                        benchmark=benchmark,
                        system_id=system_id,
                        n_samples=n,
                        chair=m.get("chair"),
                        cover=m.get("cover"),
                        hal=m.get("hal"),
                        cog=m.get("cog"),
                    )
                )
            else:
                # Discriminative — extract yes/no.
                predicted = [
                    predictions.get(s.sample_id, "no")
                    for s in split_samples
                ]
                ground_truth = [s.label for s in split_samples]
                m = amber_discriminative_metrics(predicted, ground_truth)
                results.append(
                    SplitResult(
                        split=split,
                        benchmark=benchmark,
                        system_id=system_id,
                        n_samples=n,
                        accuracy=m["accuracy"],
                        precision=m["precision"],
                        recall=m["recall"],
                        f1=m["f1"],
                    )
                )

    return results


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_eval(
    benchmark: str,
    system: str,
    mllm: str,
    corrector: str,
    data_dir: str,
    image_dir: str,
    output_dir: str,
    cache_dir: str,
    coco_instances_path: str | None = None,
) -> dict:
    """Load benchmark → run MLLM → run correction → extract answers → compute metrics → save results.

    This is the main entry point for the evaluation harness, usable both
    from Python and via the CLI.

    Args:
        benchmark: Benchmark name (``"pope"``, ``"mme"``, ``"amber"``).
        system: Correction system variant (``"raw"``, ``"woodpecker"``,
            ``"claim"``, ``"claim+geom"``, ``"full"``).
        mllm: Model identifier for the MLLM (must be in
            :data:`~relcheck_v3.mllm.wrapper.VALID_MODEL_IDS`).
        corrector: Model identifier for the correction LLM (must be in
            :data:`VALID_CORRECTOR_MODELS`).
        data_dir: Root directory containing benchmark data files.
        image_dir: Directory containing benchmark images (for POPE this
            is the COCO val2014 directory).
        output_dir: Directory for results JSON output.
        cache_dir: Root directory for all inference caches.
        coco_instances_path: Path to COCO instances JSON for RelTR
            tagging.  Only used by POPE; ``None`` for other benchmarks.

    Returns:
        Dictionary representation of the :class:`RunResult`.

    Raises:
        ValueError: If *benchmark*, *system*, *mllm*, or *corrector*
            is not a recognised value.
    """
    # --- Validate inputs ---
    if benchmark not in _VALID_BENCHMARKS:
        raise ValueError(
            f"Unknown benchmark {benchmark!r}. "
            f"Valid options: {', '.join(sorted(_VALID_BENCHMARKS))}"
        )
    if system not in _VALID_SYSTEMS:
        raise ValueError(
            f"Unknown system {system!r}. "
            f"Valid options: {', '.join(sorted(_VALID_SYSTEMS))}"
        )
    _validate_model_names(mllm, corrector)

    logger.info(
        "Starting eval: benchmark=%s, system=%s, mllm=%s, corrector=%s",
        benchmark,
        system,
        mllm,
        corrector,
    )

    # --- Ensure output directory exists ---
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # --- Set up caches ---
    answer_cache = InferenceCache(Path(cache_dir) / "answers")

    # --- Load benchmark samples ---
    logger.info("Loading %s benchmark samples from %s", benchmark, data_dir)
    samples = _load_samples(benchmark, data_dir, image_dir, coco_instances_path)
    logger.info("Loaded %d samples", len(samples))

    # --- Initialise MLLM wrapper ---
    mllm_wrapper = MLLMWrapper(
        model_id=mllm,
        cache_dir=str(Path(cache_dir) / "weights"),
        output_cache_dir=str(Path(cache_dir) / "mllm"),
    )

    # --- Initialise correction system ---
    correction_system = _build_system(
        system, corrector, cache_dir, coco_instances_path
    )

    # --- Initialise answer extractor (for discriminative tasks) ---
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    answer_extractor = AnswerExtractor(
        openai_api_key=openai_key,
        model="gpt-5.4-mini",
        cache=answer_cache,
    )

    # --- Process samples ---
    predictions: dict[str, str] = {}

    for sample in tqdm(samples, desc=f"{benchmark}/{system}"):
        sample_id = sample.sample_id
        is_generative = sample.question == "" and sample.label == ""

        # Step 1: Get MLLM output (describe for generative, answer_yesno
        # for discriminative).
        if is_generative:
            mllm_output = mllm_wrapper.describe(sample.image_path)
        else:
            mllm_output = mllm_wrapper.answer_yesno(
                sample.image_path, sample.question
            )

        # Step 2: Run correction system.
        try:
            corrected = correction_system.correct(
                sample.image_path, mllm_output
            )
        except Exception:
            logger.error(
                "Correction failed for %s — using raw MLLM output",
                sample_id,
                exc_info=True,
            )
            corrected = mllm_output

        # Step 3: For discriminative tasks, extract yes/no from the
        # corrected output.
        if is_generative:
            predictions[sample_id] = corrected
        else:
            # If the system is "raw", the corrected output is the raw
            # MLLM answer — use benchmark-specific extraction directly.
            if system == "raw":
                predictions[sample_id] = corrected
            else:
                # Use the LLM judge to extract yes/no from the corrected
                # description.
                try:
                    answer = answer_extractor.extract_yesno(
                        corrected, sample.question
                    )
                except Exception:
                    logger.error(
                        "Answer extraction failed for %s — defaulting to 'no'",
                        sample_id,
                        exc_info=True,
                    )
                    answer = "no"
                predictions[sample_id] = answer

    # --- Compute metrics ---
    logger.info("Computing metrics for %d predictions", len(predictions))
    split_results = _compute_metrics(benchmark, system, samples, predictions)

    # --- Build and save RunResult ---
    run_result = RunResult(
        benchmark=benchmark,
        system_id=system,
        mllm_model_id=mllm,
        corrector_model=corrector,
        splits=split_results,
    )

    result_filename = f"{benchmark}_{system}_{mllm.replace('/', '_')}.json"
    result_path = Path(output_dir) / result_filename
    result_path.write_text(
        run_result.model_dump_json(indent=2), encoding="utf-8"
    )
    logger.info("Results saved to %s", result_path)

    return run_result.model_dump()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the ``argparse`` parser for the eval runner CLI.

    Returns:
        Configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        description="Run the multi-benchmark evaluation harness.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m relcheck_v3.eval.eval_runner \\\n"
            "      --benchmark pope --system full \\\n"
            "      --mllm llava-hf/llava-1.5-7b-hf --corrector gpt-5.4\n"
        ),
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=sorted(_VALID_BENCHMARKS),
        help="Benchmark to evaluate on.",
    )
    parser.add_argument(
        "--system",
        required=True,
        choices=sorted(_VALID_SYSTEMS),
        help="Correction system variant.",
    )
    parser.add_argument(
        "--mllm",
        required=True,
        help=(
            "MLLM model identifier. Valid: "
            + ", ".join(sorted(VALID_MODEL_IDS))
        ),
    )
    parser.add_argument(
        "--corrector",
        required=True,
        help=(
            "Corrector model identifier. Valid: "
            + ", ".join(sorted(VALID_CORRECTOR_MODELS))
        ),
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Root directory containing benchmark data files.",
    )
    parser.add_argument(
        "--image-dir",
        required=True,
        help="Directory containing benchmark images.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/",
        help="Directory for results JSON output (default: results/).",
    )
    parser.add_argument(
        "--cache-dir",
        default="cache/",
        help="Root directory for all inference caches (default: cache/).",
    )
    parser.add_argument(
        "--coco-instances",
        default=None,
        help="Path to COCO instances JSON for RelTR tagging (POPE only).",
    )
    return parser


def main(argv: list[str] | None = None) -> dict:
    """CLI entry point for the evaluation runner.

    Args:
        argv: Command-line arguments.  Defaults to ``sys.argv[1:]``.

    Returns:
        Dictionary representation of the :class:`RunResult`.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # Configure logging.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Validate MLLM model name (argparse handles benchmark/system).
    _validate_model_names(args.mllm, args.corrector)

    return run_eval(
        benchmark=args.benchmark,
        system=args.system,
        mllm=args.mllm,
        corrector=args.corrector,
        data_dir=args.data_dir,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        coco_instances_path=args.coco_instances,
    )


if __name__ == "__main__":
    main()
