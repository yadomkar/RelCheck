"""Pipeline configuration model."""

from typing import Optional

from pydantic import BaseModel


class PipelineConfig(BaseModel):
    """All configurable pipeline parameters."""

    dataset_name: str = "coco-ee"
    annotation_path: str = ""
    image_dir: str = ""
    openai_api_key: str = ""
    output_dir: str = "relcheck_v3/output"
    max_samples: Optional[int] = None
    dry_run: bool = False
    max_retries: int = 3
