"""CLI/Colab entry point for the hallucination generation pipeline."""

import argparse
import logging
import os
import sys
from typing import Optional

from relcheck_v3.hallucination_generation.config import PipelineConfig
from relcheck_v3.hallucination_generation.pipeline import Pipeline


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Synthetic hallucination generation pipeline (Kim et al. Task 2).",
    )
    parser.add_argument(
        "--dataset-name",
        default="coco-ee",
        help='Dataset name: "coco-ee" or "flickr30k-ee" (default: coco-ee)',
    )
    parser.add_argument(
        "--annotation-path",
        required=True,
        help="Path to annotation JSON file",
    )
    parser.add_argument(
        "--image-dir",
        required=True,
        help="Directory containing source images",
    )
    parser.add_argument(
        "--openai-api-key",
        default="",
        help="OpenAI API key (default: reads from OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--output-dir",
        default="relcheck_v3/output",
        help="Output directory for results (default: relcheck_v3/output)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip API calls and use placeholder captions",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries for API calls (default: 3)",
    )
    return parser.parse_args(argv)


def main(**kwargs: object) -> None:
    """Run the hallucination generation pipeline.

    Can be called from CLI (via argparse) or directly from Python/Colab
    with keyword arguments::

        from relcheck_v3.hallucination_generation.run import main
        main(annotation_path="...", image_dir="...", dry_run=True)
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # If keyword arguments provided (Colab usage), build config directly
    if kwargs:
        # Resolve API key from kwargs or env var
        api_key = str(kwargs.get("openai_api_key", "")) or os.environ.get("OPENAI_API_KEY", "")
        config = PipelineConfig(
            dataset_name=str(kwargs.get("dataset_name", "coco-ee")),
            annotation_path=str(kwargs.get("annotation_path", "")),
            image_dir=str(kwargs.get("image_dir", "")),
            openai_api_key=api_key,
            output_dir=str(kwargs.get("output_dir", "relcheck_v3/output")),
            max_samples=kwargs.get("max_samples"),  # type: ignore[arg-type]
            dry_run=bool(kwargs.get("dry_run", False)),
            max_retries=int(kwargs.get("max_retries", 3)),
        )
    else:
        # CLI usage: parse sys.argv
        args = _parse_args()
        api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        config = PipelineConfig(
            dataset_name=args.dataset_name,
            annotation_path=args.annotation_path,
            image_dir=args.image_dir,
            openai_api_key=api_key,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            dry_run=args.dry_run,
            max_retries=args.max_retries,
        )

    pipeline = Pipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
