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


# ── API-based captioning (Together.ai) ─────────────────────────────────

def caption_image_api(
    pil_image: Image.Image,
    model: str,
    max_tokens: int = 300,
) -> str | None:
    """Generate a caption via Together.ai vision API.

    Encodes the image as base64 JPEG and sends it with DESCRIBE_PROMPT.
    Works for any vision model on Together.ai (Qwen, etc.).
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

    Returns None if the model failed to load (e.g. insufficient GPU memory).
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

    Returns None if the model failed to load.
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

    Args:
        pil_image: PIL Image to caption
        captioner: One of 'blip2', 'llava', 'qwen' (or any Together.ai model key)

    Returns:
        Caption string, or None if captioning failed.
    """
    if captioner == "llava":
        model, _ = get_llava()
        if model is not None:
            return caption_image_llava(pil_image)

    if captioner == "blip2":
        model, _ = get_blip2()
        if model is not None:
            return caption_image_blip2(pil_image)
        print("  WARNING: BLIP-2 not loaded locally. Skipping.")
        return None

    # Fallback: API-based captioning
    api_model = CAPTIONER_MODELS.get(captioner, VLM_MODEL)
    if api_model is None:
        print(f"  WARNING: No API model for captioner '{captioner}'. Skipping.")
        return None
    return caption_image_api(pil_image, api_model)
