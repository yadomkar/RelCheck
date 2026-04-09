"""Centralized OpenAI SDK wrapper with retry logic."""

import base64
import logging
import os
import time
from io import BytesIO

import openai
from PIL import Image

logger = logging.getLogger(__name__)

# Transient errors that should trigger retry
_RETRYABLE_ERRORS = (openai.APIError, openai.RateLimitError, openai.APITimeoutError)

_MAX_RETRIES = 3
_BASE_DELAY = 1  # seconds


class OpenAIClient:
    """Centralized client for all GPT-5.4-mini API calls.

    Wraps the openai SDK with retry logic and supports both text-only
    and multimodal (base64 image) requests.
    """

    def __init__(self, api_key: str = "", model: str = "gpt-5.4-mini") -> None:
        """Initialize OpenAI client.

        Args:
            api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
            model: Model ID for all calls.

        Raises:
            ValueError: If no API key is provided and OPENAI_API_KEY is not set.
        """
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "OpenAI API key not configured. "
                "Pass api_key or set the OPENAI_API_KEY environment variable."
            )
        self._client = openai.OpenAI(api_key=resolved_key)
        self.model = model

    def chat(self, system: str, user: str) -> str:
        """Text-only chat completion via openai SDK.

        Args:
            system: System message content.
            user: User message content.

        Returns:
            Response text from the model.
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self._call_with_retry(messages)

    def chat_with_image(
        self, system: str, user: str, image: str | Image.Image
    ) -> str:
        """Multimodal chat completion with base64-encoded image input.

        Args:
            system: System message content.
            user: User message content.
            image: File path string or PIL Image object.

        Returns:
            Response text from the model.
        """
        image_b64 = self._encode_image(image)
        data_uri = f"data:image/png;base64,{image_b64}"

        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_uri},
                    },
                ],
            },
        ]
        return self._call_with_retry(messages)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_with_retry(self, messages: list[dict]) -> str:
        """Call the OpenAI chat completions API with exponential backoff retry.

        Retries up to _MAX_RETRIES times on transient errors. Delay between
        retry *i* and retry *i+1* is 2^i seconds (base delay 1s).

        Raises the last caught exception after exhausting all retries.
        """
        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):  # 0, 1, 2, 3 → initial + 3 retries
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                )
                return response.choices[0].message.content or ""
            except _RETRYABLE_ERRORS as exc:
                last_error = exc
                if attempt < _MAX_RETRIES:
                    delay = 2**attempt * _BASE_DELAY
                    logger.warning(
                        "OpenAI API error (attempt %d/%d): %s. "
                        "Retrying in %.1fs...",
                        attempt + 1,
                        _MAX_RETRIES + 1,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
        # All retries exhausted
        raise last_error  # type: ignore[misc]

    @staticmethod
    def _encode_image(image: str | Image.Image) -> str:
        """Convert an image to a base64-encoded string.

        Args:
            image: File path string or PIL Image object.

        Returns:
            Base64-encoded image bytes as a string.
        """
        if isinstance(image, str):
            with open(image, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")

        # PIL Image → bytes via BytesIO
        buf = BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
