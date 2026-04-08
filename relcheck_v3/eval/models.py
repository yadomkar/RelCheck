"""Pydantic data models for the caption evaluation pipeline."""

from enum import Enum

from pydantic import BaseModel


class EvalType(str, Enum):
    CAPTION_EDITING = "caption-editing"
    POPE = "pope"
    BOTH = "both"


class POPEDomain(str, Enum):
    COCO = "coco"
    AOKVQA = "aokvqa"
    GQA = "gqa"


class POPESetting(str, Enum):
    RANDOM = "random"
    POPULAR = "popular"
    ADVERSARIAL = "adversarial"


class CESample(BaseModel):
    image_id: str
    gt_cap: str
    ref_cap: str
    image_path: str


class POPEQuestion(BaseModel):
    image_id: str
    question: str
    ground_truth: str  # "yes" or "no"
    image_path: str
    domain: POPEDomain
    setting: POPESetting


class CEPrediction(BaseModel):
    image_id: str
    ref_cap: str
    edited_cap: str
    gt_cap: str


class POPEPrediction(BaseModel):
    image_id: str
    question: str
    predicted: str  # "yes" or "no"
    ground_truth: str


class CaptionEditingScores(BaseModel):
    bleu_1: float
    bleu_4: float
    rouge_l: float
    cider: float
    spice: float


class POPEScores(BaseModel):
    accuracy: float
    f1: float


class CheckpointData(BaseModel):
    model_name: str
    test_set_name: str
    predictions: dict[str, str]  # sample_id -> prediction
