"""Evaluation pipeline configuration."""

from pydantic import BaseModel

from relcheck_v3.eval.models import EvalType


class EvalConfig(BaseModel):
    """Configuration for the caption evaluation pipeline.

    Covers model selection, evaluation type, data paths,
    image directories per domain, output settings, and
    checkpoint/sampling controls.
    """

    model_name: str  # "llava-1.5" or "mplug-owl2"
    eval_type: EvalType = EvalType.BOTH
    coco_ce_path: str = ""
    flickr_ce_path: str = ""
    pope_data_dir: str = ""
    coco_image_dir: str = ""
    flickr_image_dir: str = ""
    aokvqa_image_dir: str = ""
    gqa_image_dir: str = ""
    output_dir: str = "relcheck_v3/output/eval"
    max_samples: int | None = None
    checkpoint_interval: int = 500
