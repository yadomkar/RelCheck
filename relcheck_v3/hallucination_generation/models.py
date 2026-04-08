"""Pydantic data models for the hallucination generation pipeline."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel


class HallucinationType(str, Enum):
    OBJECT_EXISTENCE = "Object Existence"
    ATTRIBUTE = "Attribute"
    INTERACTION = "Interaction"
    COUNT = "Count"


class RecordStatus(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    PARSE_FAILURE = "parse_failure"
    API_ERROR = "api_error"


class CaptionRecord(BaseModel):
    image_id: str
    caption: str
    image_path: str


class AnnotatedRecord(BaseModel):
    image_id: str
    caption: str
    image_path: str
    hallucination_type: HallucinationType
    index: int


class APIResponse(BaseModel):
    raw_text: str
    success: bool
    error_message: Optional[str] = None


class ParsedRecord(BaseModel):
    image_id: str
    gt_cap: str
    ref_cap: str
    hallucination_type: str
    reason: str
    parse_success: bool
    raw_text: Optional[str] = None


class ValidationResult(BaseModel):
    edit_distance: int
    accepted: bool
    rejection_reason: Optional[str] = None


class ResultRecord(BaseModel):
    image_id: str
    gt_cap: str
    ref_cap: str
    hallucination_type: str
    reason: str
    edit_distance: int
    status: RecordStatus


class SummaryStats(BaseModel):
    total_processed: int
    accepted_count: int
    rejected_count_too_small: int
    rejected_count_too_large: int
    parse_failure_count: int
    api_error_count: int
    type_distribution: dict[str, int]
    type_percentages: dict[str, float]
    edit_distance_mean: float
    edit_distance_median: float
    edit_distance_min: int
    edit_distance_max: int
    duration_seconds: float
