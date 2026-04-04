"""
RelCheck v2 — Model Loading
=============================
Centralized, lazy-loaded model registry. Each model is loaded on first use
and cached for the session. Avoids redundant loading and keeps GPU memory
management in one place.
"""

from __future__ import annotations

from typing import Optional

import torch
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
)

from .config import GDINO_ID

# ── Device detection ─────────────────────────────────────────────────────

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# ── GroundingDINO ────────────────────────────────────────────────────────

_gdino_model = None
_gdino_processor = None


def get_gdino() -> tuple:
    """Return (model, processor) for GroundingDINO, loading on first call."""
    global _gdino_model, _gdino_processor
    if _gdino_model is None:
        print(f"Loading GroundingDINO on {DEVICE}...")
        _gdino_processor = AutoProcessor.from_pretrained(GDINO_ID)
        _gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            GDINO_ID
        ).to(DEVICE)
        print("  GroundingDINO ready.")
    return _gdino_model, _gdino_processor


# ── LLaVA-1.5-7B (local) ────────────────────────────────────────────────

_llava_model = None
_llava_processor = None
_LLAVA_ID = "llava-hf/llava-1.5-7b-hf"


def get_llava() -> tuple[Optional[object], Optional[object]]:
    """Return (model, processor) for LLaVA-1.5-7B, loading on first call.

    Returns (None, None) if loading fails (e.g. insufficient GPU memory).
    """
    global _llava_model, _llava_processor
    if _llava_model is None:
        try:
            from transformers import LlavaForConditionalGeneration
            from transformers import AutoProcessor as _AP

            print(f"Loading LLaVA-1.5-7B locally on {DEVICE}...")
            _llava_processor = _AP.from_pretrained(_LLAVA_ID)
            _llava_model = LlavaForConditionalGeneration.from_pretrained(
                _LLAVA_ID, torch_dtype=torch.float16, device_map="auto"
            )
            print("  LLaVA-1.5-7B ready.")
        except Exception as e:
            print(f"  LLaVA local load failed: {e}")
    return _llava_model, _llava_processor


# ── BLIP-2 (local) ──────────────────────────────────────────────────────

_blip2_model = None
_blip2_processor = None
_BLIP2_ID = "Salesforce/blip2-flan-t5-xl"


def get_blip2() -> tuple[Optional[object], Optional[object]]:
    """Return (model, processor) for BLIP-2, loading on first call.

    Returns (None, None) if loading fails.
    """
    global _blip2_model, _blip2_processor
    if _blip2_model is None:
        try:
            from transformers import Blip2ForConditionalGeneration, Blip2Processor

            print(f"Loading BLIP-2 locally on {DEVICE}...")
            _blip2_processor = Blip2Processor.from_pretrained(_BLIP2_ID)
            _blip2_model = Blip2ForConditionalGeneration.from_pretrained(
                _BLIP2_ID, torch_dtype=torch.float16, device_map="auto"
            )
            print("  BLIP-2 ready.")
        except Exception as e:
            print(f"  BLIP-2 local load failed: {e}")
    return _blip2_model, _blip2_processor
