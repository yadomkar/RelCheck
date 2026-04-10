"""Pydantic configuration models for the evaluation harness.

Defines ``EvalRunConfig``, ``MLLMConfig``, and ``SystemConfig`` — the
configuration data models used by the evaluation runner and CLI entry point.

Note:
    ``SplitResult`` and ``RunResult`` live in
    :mod:`relcheck_v3.eval.results_aggregator`.
    ``CacheEntry`` lives in :mod:`relcheck_v3.mllm.cache`.
    ``Benchmark``, ``SystemID``, and other benchmark-level models live in
    :mod:`relcheck_v3.benchmarks.models`.

Requirements: 8.1, 13.1, 13.2
"""

from __future__ import annotations

from pydantic import BaseModel

from relcheck_v3.benchmarks.models import Benchmark, SystemID


class EvalRunConfig(BaseModel):
    """Top-level configuration for a single evaluation run.

    Captures the full set of parameters needed to execute one
    benchmark × system × MLLM combination.

    Attributes:
        benchmark: Which benchmark to evaluate on.
        system_id: Which correction system variant to use.
        mllm_model_id: HuggingFace model identifier for the MLLM.
        corrector_model: Model identifier for the correction LLM.
        data_dir: Root directory containing benchmark data files.
        image_dir: Directory containing benchmark images.
        output_dir: Directory for results JSON output.
        cache_dir: Root directory for all inference caches.
        coco_instances_path: Path to COCO instances JSON for RelTR
            tagging. Only used by POPE; ``None`` for other benchmarks.
    """

    benchmark: Benchmark
    system_id: SystemID
    mllm_model_id: str = "llava-hf/llava-1.5-7b-hf"
    corrector_model: str = "gpt-5.4"
    data_dir: str = ""
    image_dir: str = ""
    output_dir: str = "results/"
    cache_dir: str = "cache/"
    coco_instances_path: str | None = None


class MLLMConfig(BaseModel):
    """Configuration for loading and running an MLLM.

    Attributes:
        model_id: HuggingFace model identifier.
        weights_dir: Directory for cached model weights (Colab Drive).
        cache_dir: Directory for cached inference outputs.
        load_in_8bit: Whether to load the model in 8-bit quantization.
    """

    model_id: str = "llava-hf/llava-1.5-7b-hf"
    weights_dir: str = "/content/weights/"
    cache_dir: str = "cache/mllm/"
    load_in_8bit: bool = True


class SystemConfig(BaseModel):
    """Configuration for a correction system variant.

    Attributes:
        system_id: Which correction system to configure.
        openai_api_key: API key for OpenAI-based correctors.
        corrector_model: Model identifier for the correction LLM.
        gdino_config: Path to GroundingDINO config file.
        gdino_checkpoint: Path to GroundingDINO checkpoint.
        reltr_checkpoint: Path to RelTR checkpoint (Full variant only).
        cache_dir: Directory for cached correction outputs.
    """

    system_id: SystemID
    openai_api_key: str = ""
    corrector_model: str = "gpt-5.4"
    gdino_config: str = ""
    gdino_checkpoint: str = ""
    reltr_checkpoint: str = ""
    cache_dir: str = "cache/systems/"
