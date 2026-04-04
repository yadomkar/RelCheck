"""
RelCheck v2 — Image Captioning
================================
Unified captioning interface supporting local models (BLIP-2, LLaVA)
and API-based models (Qwen, etc.) via Together.ai.
"""

from __future__ import annotations

from typing import Optional

import torch
from PIL import Image

from .api import llm_call, encode_b64
from .config import CAPTIONER_MODELS, DESCRIBE_PROMPT, VLM_MODEL
from .models import get_llava, get_blip2
from ._logging import log


# ── API-based captioning (Together.ai) ─────────────────────────────────

def caption_image_api(
    pil_image: Image.Image,
    model: str,
    max_tokens: int = 300,
) -> str | None:
    """Generate a caption via Together.ai vision API.

    Encodes the image as base64 JPEG and sends it with DESCRIBE_PROMPT.
    Works for any vision model on Together.ai (Qwen, etc.).

    Args:
        pil_image: PIL Image to caption.
        model: Model key on Together.ai (e.g., 'Qwen/Qwen3-VL-8B-Instruct').
        max_tokens: Maximum tokens in the generated caption. Defaults to 300.

    Returns:
        Generated caption string, or None if the API call failed.
    """
    b64 = encode_b64(pil_image)
    return llm_call(
        [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": DESCRIBE_PROMPT},
        ]}],
        model=model,
        max_tokens=max_tokens,
    )


# ── Local LLaVA-1.5-7B captioning ─────────────────────────────────────

def caption_image_llava(
    pil_image: Image.Image,
    max_new_tokens: int = 300,
) -> str | None:
    """Generate a caption using locally-loaded LLaVA-1.5-7B.

    Loads the model from Hugging Face if not already in memory. Applies the
    model's chat template and generates a caption conditioned on DESCRIBE_PROMPT.

    Args:
        pil_image: PIL Image to caption.
        max_new_tokens: Maximum tokens in the generated caption. Defaults to 300.

    Returns:
        Generated caption string, or None if the model failed to load
        (e.g., insufficient GPU memory).
    """
    model, processor = get_llava()
    if model is None:
        return None

    conversation = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": DESCRIBE_PROMPT},
    ]}]
    prompt_text = processor.apply_chat_template(
        conversation, add_generation_prompt=True,
    )
    inputs = processor(
        images=pil_image, text=prompt_text, return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    gen_ids = out[:, inputs["input_ids"].shape[1]:]
    return processor.decode(gen_ids[0], skip_special_tokens=True).strip()


# ── Local BLIP-2 captioning ───────────────────────────────────────────

def caption_image_blip2(
    pil_image: Image.Image,
    max_new_tokens: int = 50,
) -> str | None:
    """Generate a caption using locally-loaded BLIP-2 (blip2-flan-t5-xl).

    Loads the model from Hugging Face if not already in memory. Generates a short
    caption using the BLIP-2 architecture with T5 decoder.

    Args:
        pil_image: PIL Image to caption.
        max_new_tokens: Maximum tokens in the generated caption. Defaults to 50.

    Returns:
        Generated caption string, or None if the model failed to load.
    """
    model, processor = get_blip2()
    if model is None:
        return None

    inputs = processor(
        images=pil_image, return_tensors="pt",
    ).to(model.device, torch.float16)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)

    return processor.decode(out[0], skip_special_tokens=True).strip()


# ── Unified captioning router ─────────────────────────────────────────

def caption_image(
    pil_image: Image.Image,
    captioner: str = "llava",
) -> str | None:
    """Generate a caption using the specified captioner.

    Routes to the appropriate captioning backend:
      - 'llava': Local LLaVA-1.5-7B via GPU
      - 'blip2': Local BLIP-2 via GPU
      - Other keys: API-based via Together.ai (e.g., 'qwen', 'internvl')

    Falls back to Together.ai API if the requested local model is not available.

    Args:
        pil_image: PIL Image to caption.
        captioner: Model identifier. One of 'blip2', 'llava', 'qwen',
                   or any Together.ai model key. Defaults to 'llava'.

    Returns:
        Generated caption string, or None if captioning failed
        (e.g., model not available, API error).
    """
    if captioner == "llava":
        model, _ = get_llava()
        if model is not None:
            return caption_image_llava(pil_image)

    if captioner == "blip2":
        model, _ = get_blip2()
        if model is not None:
            return caption_image_blip2(pil_image)
        log.warning("BLIP-2 not loaded locally. Skipping.")
        return None

    # Fallback: API-based captioning
    api_model = CAPTIONER_MODELS.get(captioner, VLM_MODEL)
    if api_model is None:
        log.warning(f"No API model for captioner '{captioner}'. Skipping.")
        return None
    return caption_image_api(pil_image, api_model)
