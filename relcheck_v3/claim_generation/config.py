"""Claim generation pipeline configuration model."""

from typing import Optional

from pydantic import BaseModel


class ClaimGenConfig(BaseModel):
    """All configurable claim generation pipeline parameters."""

    openai_api_key: str = ""
    gpt_model_id: str = "gpt-5.4-mini"
    grounding_dino_model_id: str = "IDEA-Research/grounding-dino-base"
    qa2claim_model_id: str = "khhuang/zerofec-qa2claim-t5-base"
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    output_dir: str = "relcheck_v3/output/claim_generation"
    checkpoint_interval: int = 50
    max_samples: Optional[int] = None
