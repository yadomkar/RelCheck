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

# ── Module-level client (initialized by init_client) ─────────────────────

_client: Optional[Together] = None


def init_client(api_key: str) -> None:
    """Initialize the Together.ai client. Must be called before any API use."""
    global _client
    _client = Together(api_key=api_key)


def get_client() -> Together:
    """Return the initialized client, raising if init_client was not called."""
    if _client is None:
        raise RuntimeError("Call api.init_client(api_key) before making API calls.")
    return _client


# ── Image encoding ───────────────────────────────────────────────────────

def encode_b64(image: Image.Image, quality: int = 85) -> str:
    """Encode a PIL Image as a JPEG base64 string."""
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
    """Low-level API call with tenacity retry. Not called directly."""
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
    """Call an LLM via Together.ai. Returns None on failure (after retries)."""
    try:
        return _raw_llm_call(messages, model, max_tokens, temperature)
    except Exception as e:
        print(f"  [llm_call] API failed after retries: {e}")
        return None


def vlm_call(
    messages: list[dict],
    max_tokens: int = 10,
    model: Optional[str] = None,
) -> Optional[str]:
    """Call a VLM via Together.ai. Defaults to VLM_MODEL."""
    return llm_call(messages, model=model or VLM_MODEL, max_tokens=max_tokens)


def vlm_yesno(
    image: Image.Image,
    question: str,
    model: Optional[str] = None,
) -> tuple[float, float]:
    """Ask a yes/no question via VLM. Returns (yes_ratio, confidence).

    Together.ai does not support logprobs for vision models, so we parse
    the text answer: yes → (1.0, 0.9), no → (0.0, 0.9), unclear → (0.5, 0.1).
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
