"""Pydantic data models for the claim generation pipeline."""

from pydantic import BaseModel


# --- Detection ---


class Detection(BaseModel):
    bbox: list[float]  # [x_min, y_min, x_max, y_max], normalized 0-1
    confidence: float


# --- Stage 2 outputs ---


class ObjectQuestion(BaseModel):
    object_name: str
    question: str  # "Is there any {object} in the image? How many are there?"


class AttributeQuestion(BaseModel):
    question: str
    entities: list[str]  # entities involved, parsed from "&" delimiter


class FormulatedQuestions(BaseModel):
    object_questions: list[ObjectQuestion]
    attribute_questions: list[AttributeQuestion]


# --- Stage 3 outputs ---


class ObjectAnswer(BaseModel):
    object_name: str
    count: int
    bboxes: list[list[float]]  # list of [x_min, y_min, x_max, y_max]


class AttributeQA(BaseModel):
    question: str
    entities: list[str]
    answer: str


# --- Stage 4 outputs ---


class CountClaim(BaseModel):
    object_name: str
    count: int
    claim_text: str  # "There are 2 dogs." or "There is no cat."
    bboxes: list[list[float]]


class SpecificClaim(BaseModel):
    object_name: str
    instance_index: int | None = None  # e.g., "dog 1", "dog 2"
    bbox: list[float] | None = None
    claim_text: str


class OverallClaim(BaseModel):
    claim_text: str


class VisualKnowledgeBase(BaseModel):
    count_claims: list[CountClaim]
    specific_claims: list[SpecificClaim]
    overall_claims: list[OverallClaim]

    def format(self) -> str:
        """Format as labeled text string with Count/Specific/Overall sections."""
        lines: list[str] = []

        lines.append("Count:")
        for i, c in enumerate(self.count_claims, 1):
            lines.append(f"{i}. {c.claim_text}")

        lines.append("Specific:")
        for i, s in enumerate(self.specific_claims, 1):
            lines.append(f"{i}. {s.claim_text}")

        lines.append("Overall:")
        for i, o in enumerate(self.overall_claims, 1):
            lines.append(f"{i}. {o.claim_text}")

        return "\n".join(lines)


# --- Pipeline I/O ---


class InputSample(BaseModel):
    image_id: str
    image_path: str
    ref_cap: str


class StageTimings(BaseModel):
    stage1_seconds: float = 0.0
    stage2_seconds: float = 0.0
    stage3_seconds: float = 0.0
    stage4_seconds: float = 0.0
    total_seconds: float = 0.0


class SampleResult(BaseModel):
    image_id: str
    ref_cap: str
    key_concepts: list[str]
    object_questions: list[str]
    attribute_questions: list[AttributeQuestion]
    object_answers: dict[str, ObjectAnswer]
    attribute_answers: list[AttributeQA]
    visual_knowledge_base: VisualKnowledgeBase
    vkb_text: str
    timings: StageTimings
    success: bool
    error_message: str | None = None
