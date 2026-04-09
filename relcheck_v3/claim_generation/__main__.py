"""CLI/Colab entry point for the claim generation pipeline.

Usage:
    python -m relcheck_v3.claim_generation          # CLI
    from relcheck_v3.claim_generation.__main__ import main  # Colab
"""

import argparse
import json
import logging
import sys

from relcheck_v3.claim_generation.config import ClaimGenConfig
from relcheck_v3.claim_generation.models import InputSample
from relcheck_v3.claim_generation.pipeline import ClaimGenerationPipeline

logger = logging.getLogger(__name__)


def main(
    samples: list[InputSample] | None = None,
    config: ClaimGenConfig | None = None,
) -> list:
    """Run the claim generation pipeline.

    Args:
        samples: List of InputSample objects. If None, reads from
            --input-jsonl CLI argument.
        config: Pipeline configuration. If None, uses defaults with
            CLI overrides.

    Returns:
        List of SampleResult objects.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if config is None:
        args = _parse_args()
        config = ClaimGenConfig(
            openai_api_key=args.api_key or "",
            gpt_model_id=args.model,
            output_dir=args.output_dir,
            checkpoint_interval=args.checkpoint_interval,
            max_samples=args.max_samples,
        )

    pipeline = ClaimGenerationPipeline(config)

    if samples is None:
        args = _parse_args()
        samples = _load_samples(args.input_jsonl)

    results = pipeline.process_batch(samples)
    logger.info("Done — %d samples processed", len(results))
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Woodpecker claim generation pipeline (Stages 1–4)"
    )
    parser.add_argument(
        "--input-jsonl",
        default="",
        help="Path to input JSONL file with image_id, image_path, ref_cap fields",
    )
    parser.add_argument("--api-key", default="", help="OpenAI API key")
    parser.add_argument(
        "--model", default="gpt-5.4-mini", help="GPT model ID"
    )
    parser.add_argument(
        "--output-dir",
        default="relcheck_v3/output/claim_generation",
        help="Output directory",
    )
    parser.add_argument(
        "--checkpoint-interval", type=int, default=50, help="Checkpoint interval"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Max samples to process"
    )
    return parser.parse_args()


def _load_samples(path: str) -> list[InputSample]:
    if not path:
        logger.error("No --input-jsonl provided")
        sys.exit(1)
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(InputSample.model_validate_json(line))
    return samples


if __name__ == "__main__":
    main()
