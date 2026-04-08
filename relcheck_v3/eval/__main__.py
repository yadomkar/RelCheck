"""Entry point for the caption evaluation pipeline.

Supports both CLI usage (``python -m relcheck_v3.eval``) and direct
import in Colab notebooks via ``from relcheck_v3.eval.__main__ import main``.

Requirements: 9.1, 10.1, 13.1
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

from relcheck_v3.eval.config import EvalConfig
from relcheck_v3.eval.data_loaders import CEDataLoader, POPEDataLoader
from relcheck_v3.eval.export import ResultsExporter
from relcheck_v3.eval.models import (
    CaptionEditingScores,
    EvalType,
    POPEDomain,
    POPEScores,
    POPESetting,
)
from relcheck_v3.eval.pipeline import EvalPipeline

logger = logging.getLogger(__name__)


def _build_model_wrappers(
    model_name: str,
) -> tuple[Any, Any]:
    """Instantiate caption editor and POPE responder for the given model.

    Imports are deferred so GPU-dependent packages are only loaded when
    actually needed.

    Returns:
        (caption_editor, pope_responder) — pope_responder may be None
        for the passthrough model.
    """
    if model_name == "passthrough":
        from relcheck_v3.eval.baselines.passthrough import PassthroughCaptionEditor

        return PassthroughCaptionEditor(), None

    if model_name == "llava-1.5":
        from relcheck_v3.eval.baselines.llava import (
            LLaVACaptionEditor,
            LLaVAPOPEResponder,
            _LLaVAModel,
        )

        shared_model = _LLaVAModel()
        return (
            LLaVACaptionEditor(shared_model),
            LLaVAPOPEResponder(shared_model),
        )

    if model_name == "mplug-owl2":
        from relcheck_v3.eval.baselines.mplug_owl2 import (
            MPLUGCaptionEditor,
            MPLUGPOPEResponder,
            _MPLUGModel,
        )

        shared_model = _MPLUGModel()
        return (
            MPLUGCaptionEditor(shared_model),
            MPLUGPOPEResponder(shared_model),
        )

    raise ValueError(
        f"Unknown model: {model_name!r}. "
        "Choose from: llava-1.5, mplug-owl2, passthrough"
    )


def _parse_args(argv: list[str] | None = None) -> EvalConfig:
    """Parse CLI arguments into an EvalConfig."""
    parser = argparse.ArgumentParser(
        prog="python -m relcheck_v3.eval",
        description="Run Kim et al. caption editing and POPE evaluation.",
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["llava-1.5", "mplug-owl2", "passthrough"],
        help="Baseline model to evaluate.",
    )
    parser.add_argument(
        "--eval-type",
        default="both",
        choices=["caption-editing", "pope", "both"],
        help="Evaluation track(s) to run (default: both).",
    )
    parser.add_argument("--coco-ce-path", default="", help="Path to COCO-CE test set JSON.")
    parser.add_argument("--flickr-ce-path", default="", help="Path to Flickr30K-CE test set JSON.")
    parser.add_argument("--pope-data-dir", default="", help="Directory with POPE question files.")
    parser.add_argument("--coco-image-dir", default="", help="COCO image directory.")
    parser.add_argument("--flickr-image-dir", default="", help="Flickr30K image directory.")
    parser.add_argument("--aokvqa-image-dir", default="", help="AOKVQA image directory.")
    parser.add_argument("--gqa-image-dir", default="", help="GQA image directory.")
    parser.add_argument(
        "--output-dir",
        default="relcheck_v3/output/eval",
        help="Output directory for results (default: relcheck_v3/output/eval).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit processing to first N samples per test set.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=500,
        help="Save checkpoint every N samples (default: 500).",
    )

    args = parser.parse_args(argv)

    return EvalConfig(
        model_name=args.model,
        eval_type=EvalType(args.eval_type),
        coco_ce_path=args.coco_ce_path,
        flickr_ce_path=args.flickr_ce_path,
        pope_data_dir=args.pope_data_dir,
        coco_image_dir=args.coco_image_dir,
        flickr_image_dir=args.flickr_image_dir,
        aokvqa_image_dir=args.aokvqa_image_dir,
        gqa_image_dir=args.gqa_image_dir,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        checkpoint_interval=args.checkpoint_interval,
    )


def main(
    config: EvalConfig | None = None,
) -> dict[str, Any]:
    """Run the evaluation pipeline.

    Can be called two ways:

    1. **CLI** — ``python -m relcheck_v3.eval --model llava-1.5 ...``
       ``config`` is ``None``; arguments are parsed from ``sys.argv``.
    2. **Colab / direct import** — pass an ``EvalConfig`` directly::

           from relcheck_v3.eval.__main__ import main
           results = main(EvalConfig(model_name="passthrough", ...))

    Args:
        config: Pre-built configuration. When *None*, CLI args are parsed.

    Returns:
        Dictionary with keys ``"caption_editing"`` and/or ``"pope"``
        containing the computed scores, suitable for further inspection
        in a notebook.
    """
    if config is None:
        config = _parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info(
        "Starting evaluation — model=%s, eval_type=%s",
        config.model_name,
        config.eval_type.value,
    )

    # --- Instantiate model wrappers (lazy GPU imports) ---
    caption_editor, pope_responder = _build_model_wrappers(config.model_name)

    # --- Create pipeline ---
    pipeline = EvalPipeline(
        config=config,
        caption_editor=caption_editor,
        pope_responder=pope_responder,
    )

    results: dict[str, Any] = {}
    table2_results: dict[str, dict[str, CaptionEditingScores]] = {}
    table3_results: dict[str, dict[tuple[POPEDomain, POPESetting], POPEScores]] = {}

    # --- Caption Editing track ---
    if config.eval_type in (EvalType.CAPTION_EDITING, EvalType.BOTH):
        ce_loader = CEDataLoader()
        model_ce_scores: dict[str, CaptionEditingScores] = {}

        if config.coco_ce_path:
            logger.info("Loading COCO-CE test set from %s", config.coco_ce_path)
            coco_samples = ce_loader.load(config.coco_ce_path, config.coco_image_dir)
            logger.info("Loaded %d COCO-CE samples", len(coco_samples))
            scores = pipeline.run_caption_editing("COCO-CE", coco_samples)
            model_ce_scores["COCO-CE"] = scores
            logger.info("COCO-CE scores: %s", scores)

        if config.flickr_ce_path:
            logger.info("Loading Flickr30K-CE test set from %s", config.flickr_ce_path)
            flickr_samples = ce_loader.load(config.flickr_ce_path, config.flickr_image_dir)
            logger.info("Loaded %d Flickr30K-CE samples", len(flickr_samples))
            scores = pipeline.run_caption_editing("Flickr30K-CE", flickr_samples)
            model_ce_scores["Flickr30K-CE"] = scores
            logger.info("Flickr30K-CE scores: %s", scores)

        if model_ce_scores:
            table2_results[config.model_name] = model_ce_scores
            results["caption_editing"] = model_ce_scores

    # --- POPE track ---
    if config.eval_type in (EvalType.POPE, EvalType.BOTH):
        if pope_responder is None:
            logger.warning(
                "Model %s does not support POPE evaluation (no POPE_Responder). "
                "Skipping POPE track.",
                config.model_name,
            )
        elif config.pope_data_dir:
            pope_loader = POPEDataLoader()
            image_dirs = {
                POPEDomain.COCO: config.coco_image_dir,
                POPEDomain.AOKVQA: config.aokvqa_image_dir,
                POPEDomain.GQA: config.gqa_image_dir,
            }
            logger.info("Loading POPE data from %s", config.pope_data_dir)
            pope_data = pope_loader.load(config.pope_data_dir, image_dirs)
            logger.info(
                "Loaded POPE data: %d domain×setting combinations",
                len(pope_data),
            )

            pope_scores = pipeline.run_pope(pope_data)
            table3_results[config.model_name] = pope_scores
            results["pope"] = {
                f"{d.value}_{s.value}": sc
                for (d, s), sc in pope_scores.items()
            }

    # --- Export results ---
    exporter = ResultsExporter(config.output_dir)

    if table2_results:
        exporter.export_table2(table2_results)
        logger.info("Exported Table 2 results to %s", config.output_dir)

    if table3_results:
        exporter.export_table3(table3_results)
        logger.info("Exported Table 3 results to %s", config.output_dir)

    if table2_results or table3_results:
        exporter.export_aggregate_json(
            table2_results=table2_results or None,
            table3_results=table3_results or None,
        )
        logger.info("Exported aggregate JSON to %s", config.output_dir)

    logger.info("Evaluation complete.")
    return results


if __name__ == "__main__":
    main()
