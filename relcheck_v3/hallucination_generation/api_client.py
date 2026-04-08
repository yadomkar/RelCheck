"""APIClient: OpenAI SDK multimodal request construction."""

import base64

import openai
from openai import OpenAI

from relcheck_v3.hallucination_generation.models import AnnotatedRecord, APIResponse


_SYSTEM_MESSAGE = (
    "You are a multimodal assistant tasked with modifying captions. "
    "Specifically, given an image and its corresponding caption, you are asked "
    "to modify the caption with the following guideline. The modified caption "
    "must include one aspect that is not consistent with the given image. "
    "The aspects are as follows: "
    "Object existence: Modify the caption by replacing an existing objects with "
    "a non-existent one, ensuring that the changes are clearly different from "
    "the image but remain plausible. "
    "Attribute: Misdescribe the attribute such as color, pose, position, and "
    "activity of one of the objects in the caption. "
    "Interaction: Modify the caption to mispresent the interactions among the "
    "objects in the image. "
    "Count: Change the caption to inaccurately represent the number of a "
    "certain object in the image while still mentioning the actual objects. "
    "The edit distance should be smaller than 50 and greater than 5."
)

_USER_MESSAGE_TEMPLATE = (
    "Based on the given image and caption, modify the caption to be "
    "inconsistent with the image based upon the given aspect. The output "
    "format should be as follows: 'image id': {id}, 'GT-Cap': image caption "
    "before modification, 'Ref-Cap': image caption after modification, "
    "'Type': type of aspect to generate Ref-Cap: {aspect}, 'Reason': the "
    "reason why the caption is inconsistent with the image. The caption "
    "(GT-Cap) is as follows: {caption}"
)


class APIClient:
    """Sends multimodal requests to GPT-4o-mini via the official OpenAI SDK."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def _build_system_message(self) -> str:
        """Return the exact system prompt from the paper (verbatim)."""
        return _SYSTEM_MESSAGE

    @staticmethod
    def _encode_image(image_path: str) -> str:
        """Read image file and return base64 data URL string."""
        with open(image_path, "rb") as f:
            base64_data = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{base64_data}"

    def _build_user_message(self, record: AnnotatedRecord) -> list[dict]:
        """Build multimodal user message with text and image content."""
        text = _USER_MESSAGE_TEMPLATE.format(
            id=record.image_id,
            aspect=record.hallucination_type.value,
            caption=record.caption,
        )
        data_url = self._encode_image(record.image_path)
        return [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]

    def generate_hallucination(self, record: AnnotatedRecord) -> APIResponse:
        """Send chat completion request via SDK, return APIResponse."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._build_system_message()},
                    {"role": "user", "content": self._build_user_message(record)},
                ],
            )
            raw_text = response.choices[0].message.content or ""
            return APIResponse(raw_text=raw_text, success=True)
        except openai.APIError as exc:
            return APIResponse(
                raw_text="",
                success=False,
                error_message=f"OpenAI API error: {exc}",
            )
        except Exception as exc:
            return APIResponse(
                raw_text="",
                success=False,
                error_message=f"Unexpected error: {exc}",
            )
