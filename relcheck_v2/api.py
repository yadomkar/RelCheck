"""
RelCheck v2 — API Clients
===========================
Together.ai LLM/VLM wrappers with tenacity retry, and image encoding utility.
Single source of truth — no duplicate encode_b64 or retry logic elsewhere.
"""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Optional

from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from together import Together

from .config import LLM_MODEL, VLM_MODEL
from ._logging import log

# ── Module-level client (initialized by init_client) ─────────────────────

_client: Optional[Together] = None


def init_client(api_key: str) -> None:
    """
    Initialize the Together.ai client.

    Must be called before any API use. Stores the client in module-level
    singleton _client for use by get_client() and all API functions.

    Args:
        api_key: Together.ai API key (typically from environment variable).

    Raises:
        RuntimeError: If api_key is invalid (propagated by Together SDK).
    """
    global _client
    _client = Together(api_key=api_key)


def get_client() -> Together:
    """
    Return the initialized client.

    Raises RuntimeError if init_client() was not called first. Acts as a
    safety check to prevent accidental None usage in API calls.

    Returns:
        Initialized Together client singleton.

    Raises:
        RuntimeError: If init_client() has not been called.
    """
    if _client is None:
        raise RuntimeError("Call api.init_client(api_key) before making API calls.")
    return _client


# ── Image encoding ───────────────────────────────────────────────────────

def encode_b64(image: Image.Image, quality: int = 85) -> str:
    """
    Encode a PIL Image as a JPEG base64 string.

    Used for embedding images in Together.ai API messages. Converts to RGB
    to ensure compatibility with JPEG format.

    Args:
        image: PIL Image object to encode.
        quality: JPEG quality level (0-100, default 85). Higher values
                 produce better quality but larger strings.

    Returns:
        Base64-encoded JPEG string (without data:// prefix).
    """
    buf = BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── LLM / VLM calls ─────────────────────────────────────────────────────

@retry(
    wait=wait_exponential(multiplier=1, min=1, max=8),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _raw_llm_call(
    messages: list[dict],
    model: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """
    Low-level LLM API call with tenacity exponential backoff retry.

    Attempts up to 3 times with exponential backoff (1–8 seconds). Raises
    the final exception after retries are exhausted. Not meant to be called
    directly; use llm_call() or vlm_call() instead for error handling.

    Args:
        messages: OpenAI-format message list.
        model: Model name (e.g., "meta-llama/Llama-3.3-70B-Instruct-Turbo").
        max_tokens: Maximum response tokens.
        temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative).

    Returns:
        Response text content (stripped of whitespace).

    Raises:
        Exception: Any API error after all 3 retry attempts are exhausted.
    """
    resp = get_client().chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def llm_call(
    messages: list[dict],
    model: str = LLM_MODEL,
    max_tokens: int = 600,
    temperature: float = 0.0,
) -> Optional[str]:
    """
    Call an LLM via Together.ai with robust error handling.

    Wraps _raw_llm_call() and catches exceptions after retries, logging
    the error and returning None. This allows calling code to continue
    gracefully on API failures.

    Args:
        messages: OpenAI-format message list.
        model: Model name (defaults to LLM_MODEL from config).
        max_tokens: Maximum response tokens (default 600).
        temperature: Sampling temperature (default 0.0 for deterministic).

    Returns:
        Response text content, or None if API failed after retries.
    """
    try:
        return _raw_llm_call(messages, model, max_tokens, temperature)
    except Exception as e:
        log.error(f"llm_call: API failed after retries: {e}")
        return None


def vlm_call(
    messages: list[dict],
    max_tokens: int = 10,
    model: Optional[str] = None,
) -> Optional[str]:
    """
    Call a Vision Language Model via Together.ai.

    Convenience wrapper around llm_call() that defaults to VLM_MODEL from
    config. Useful for delegating visual understanding tasks to a stronger
    vision model (e.g., Maverick, Qwen3-VL).

    Args:
        messages: OpenAI-format message list with image_url and text content.
        max_tokens: Maximum response tokens (default 10 for short answers).
        model: Model name (defaults to VLM_MODEL from config if None).

    Returns:
        Response text content, or None if API failed after retries.
    """
    return llm_call(messages, model=model or VLM_MODEL, max_tokens=max_tokens)


def vlm_yesno(
    image: Image.Image,
    question: str,
    model: Optional[str] = None,
) -> tuple[Optional[float], Optional[float]]:
    """
    Ask a yes/no question via VLM and return (yes_ratio, confidence).

    Together.ai does not support logprobs for vision models, so we parse
    the text answer: "yes" → (1.0, 0.9), "no" → (0.0, 0.9), unclear
    response → (0.5, 0.1). Returns (None, None) if API fails.

    Args:
        image: PIL Image to ask the question about.
        question: Natural language yes/no question (e.g., "Is there a cat?").
        model: Vision model name (defaults to VLM_MODEL if None).

    Returns:
        Tuple of (yes_ratio, confidence):
            - yes_ratio: 1.0 (yes), 0.0 (no), 0.5 (unclear)
            - confidence: 0.9 (clear answer), 0.1 (unclear)
            - (None, None) if API call fails
    """
    b64 = encode_b64(image)
    resp = vlm_call(
        [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": question + " Answer with exactly one word: yes or no."},
        ]}],
        max_tokens=5,
        model=model,
    )
    if resp is None:
        return (None, None)

    answer = resp.strip().lower()
    if answer.startswith("y"):
        return (1.0, 0.9)
    elif answer.startswith("n"):
        return (0.0, 0.9)
    else:
        return (0.5, 0.1)
