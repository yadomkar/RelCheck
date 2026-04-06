"""
RelCheck v2 — Model Loading
=============================
Centralized, lazy-loaded model registry. Each model is loaded on first use
and cached for the session. Avoids redundant loading and keeps GPU memory
management in one place.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
)

from ._logging import log
from .config import GDINO_ID

# ── Device detection ─────────────────────────────────────────────────────

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# ── GroundingDINO ────────────────────────────────────────────────────────

_gdino_model: Optional[Any] = None
_gdino_processor: Optional[Any] = None


def get_gdino() -> tuple[Any, Any]:
    """Load and return GroundingDINO model and processor.

    Uses lazy loading and caching: on first call, loads the model and processor
    from HuggingFace and caches them in module globals. Subsequent calls return
    the cached instances.

    Returns:
        tuple[Any, Any]: (model, processor) pair for GroundingDINO object detection.
            Model is moved to the detected device (cuda or cpu).

    Raises:
        Exception: If model loading from HuggingFace fails (e.g., network error,
            invalid model ID, or insufficient disk space).
    """
    global _gdino_model, _gdino_processor
    if _gdino_model is None:
        log.info(f"Loading GroundingDINO on {DEVICE}...")
        _gdino_processor = AutoProcessor.from_pretrained(GDINO_ID)
        _gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            GDINO_ID
        ).to(DEVICE)
        log.info("  GroundingDINO ready.")
    return _gdino_model, _gdino_processor


# ── LLaVA-1.5-7B (local) ────────────────────────────────────────────────

_llava_model: Optional[Any] = None
_llava_processor: Optional[Any] = None
_LLAVA_ID = "llava-hf/llava-1.5-7b-hf"


def get_llava() -> tuple[Optional[Any], Optional[Any]]:
    """Load and return LLaVA-1.5-7B model and processor (graceful failure).

    Uses lazy loading and caching: on first call, attempts to load the model
    and processor from HuggingFace. If loading fails (e.g., insufficient GPU
    memory), returns (None, None) and logs the error.

    Returns:
        tuple[Optional[Any], Optional[Any]]: (model, processor) pair on success,
            or (None, None) if loading fails. Model uses float16 precision and
            device_map="auto" for efficient GPU placement.

    Note:
        This function gracefully degrades: callers should check for None returns
        and either raise an informative error or fall back to an alternative model.
    """
    global _llava_model, _llava_processor
    if _llava_model is None:
        try:
            from transformers import LlavaForConditionalGeneration
            from transformers import AutoProcessor as _AP

            log.info(f"Loading LLaVA-1.5-7B locally on {DEVICE}...")
            _llava_processor = _AP.from_pretrained(_LLAVA_ID)
            _llava_model = LlavaForConditionalGeneration.from_pretrained(
                _LLAVA_ID, torch_dtype=torch.float16, device_map="auto"
            )
            log.info("  LLaVA-1.5-7B ready.")
        except Exception as e:
            log.error(f"  LLaVA local load failed: {e}")
    return _llava_model, _llava_processor


# ── BLIP-2 (local) ──────────────────────────────────────────────────────

_blip2_model: Optional[Any] = None
_blip2_processor: Optional[Any] = None
_BLIP2_ID = "Salesforce/blip2-flan-t5-xl"


def get_blip2() -> tuple[Optional[Any], Optional[Any]]:
    """Load and return BLIP-2 model and processor (graceful failure).

    Uses lazy loading and caching: on first call, attempts to load the model
    and processor from HuggingFace. If loading fails (e.g., insufficient GPU
    memory, network error), returns (None, None) and logs the error.

    Returns:
        tuple[Optional[Any], Optional[Any]]: (model, processor) pair on success,
            or (None, None) if loading fails. Model uses float16 precision and
            device_map="auto" for efficient GPU placement.

    Note:
        This function gracefully degrades: callers should check for None returns
        and either raise an informative error or fall back to an alternative model.
    """
    global _blip2_model, _blip2_processor
    if _blip2_model is None:
        try:
            from transformers import Blip2ForConditionalGeneration, Blip2Processor

            log.info(f"Loading BLIP-2 locally on {DEVICE}...")
            _blip2_processor = Blip2Processor.from_pretrained(_BLIP2_ID)
            _blip2_model = Blip2ForConditionalGeneration.from_pretrained(
                _BLIP2_ID, torch_dtype=torch.float16, device_map="auto"
            )
            log.info("  BLIP-2 ready.")
        except Exception as e:
            log.error(f"  BLIP-2 local load failed: {e}")
    return _blip2_model, _blip2_processor


# ── RelTR (scene graph generation) ──────────────────────────────────────

_reltr_model: Optional[Any] = None
_reltr_transform: Optional[Any] = None


def get_reltr() -> tuple[Optional[Any], Optional[Any]]:
    """Load and return RelTR model and image transform (graceful failure).

    Uses lazy loading and caching. Imports RelTR's ``build_model`` from the
    cloned repo via sys.path. Returns ``(None, None)`` on any failure.

    Returns:
        (model, transform) on success, (None, None) on failure.
    """
    global _reltr_model, _reltr_transform
    if _reltr_model is None:
        try:
            import sys
            import os
            import argparse
            import torchvision.transforms as T
            from .config import RELTR_CHECKPOINT_PATH

            # RelTR repo must be on sys.path (cloned in Colab setup)
            reltr_dir = os.path.join(os.getcwd(), "RelTR")
            if reltr_dir not in sys.path:
                sys.path.insert(0, reltr_dir)

            from models import build_model  # type: ignore[import-untyped]

            args = argparse.Namespace(
                backbone="resnet50", dilation=False, lr_backbone=0,
                return_interm_layers=False, position_embedding="sine",
                enc_layers=6, dec_layers=6, dim_feedforward=2048,
                hidden_dim=256, dropout=0.1, nheads=8,
                num_entities=100, num_triplets=200, pre_norm=False,
                set_cost_class=1, set_cost_bbox=5, set_cost_giou=2,
                set_iou_threshold=0.7, bbox_loss_coef=5, giou_loss_coef=2,
                rel_loss_coef=1, eos_coef=0.1, aux_loss=False,
                dataset="vg", device=DEVICE,
            )

            model, _, _ = build_model(args)
            ckpt = torch.load(RELTR_CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ckpt["model"])
            model.eval().to(DEVICE)

            ckpt_size = os.path.getsize(RELTR_CHECKPOINT_PATH) / 1e6
            log.info("RelTR loaded on %s (%.1f MB checkpoint)", DEVICE, ckpt_size)

            _reltr_transform = T.Compose([
                T.Resize(800),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            _reltr_model = model
        except Exception as e:
            log.error("RelTR load failed: %s", e)
    return _reltr_model, _reltr_transform
