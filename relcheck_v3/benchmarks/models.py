"""Pydantic data models for the multi-benchmark evaluation harness.

Defines the uniform sample schema shared by all benchmark loaders (POPE, MME,
AMBER) and the configuration models for benchmark setup.
"""

from enum import Enum

from pydantic import BaseModel


class Benchmark(str, Enum):
    """Supported evaluation benchmarks."""

    POPE = "pope"
    MME = "mme"
    AMBER = "amber"


class SystemID(str, Enum):
    """Correction system variant identifiers."""

    RAW = "raw"
    WOODPECKER = "woodpecker"
    CLAIM = "claim"
    CLAIM_GEOM = "claim+geom"
    FULL = "full"


class BenchmarkSample(BaseModel):
    """Uniform sample schema yielded by all benchmark loaders.

    Attributes:
        sample_id: Unique identifier across the benchmark.
        image_path: Absolute path to the image file.
        question: Discriminative question text; empty for generative tasks.
        label: Ground-truth answer ("yes"/"no") or empty for generative.
        split: Split or subtask name (e.g. "random", "existence", "de").
        benchmark: Benchmark name ("pope", "mme", "amber").
        reltr_tag: True if COCO image has RelTR vocab overlap; None for
            non-COCO benchmarks.
        metadata: Benchmark-specific extras (e.g. AMBER image_id, MME
            pair_id).
    """

    sample_id: str
    image_path: str
    question: str
    label: str
    split: str
    benchmark: str
    reltr_tag: bool | None
    metadata: dict = {}


class BenchmarkConfig(BaseModel):
    """Configuration for loading a specific benchmark.

    Attributes:
        benchmark: Which benchmark to load.
        data_dir: Root directory containing benchmark data files.
        image_dir: Directory containing benchmark images. For POPE this is
            the COCO val2014 directory.
        coco_instances_path: Path to COCO instances JSON for RelTR tagging.
            Only used by POPE; None for other benchmarks.
    """

    benchmark: Benchmark
    data_dir: str
    image_dir: str
    coco_instances_path: str | None = None
