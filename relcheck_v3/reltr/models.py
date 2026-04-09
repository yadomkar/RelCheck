"""
RelTR Model Loading — carried over from relcheck_v2/models.py (RelTR section only).

Lazy-loaded, cached model registry. Identical to v2's get_reltr + DEVICE.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch

from relcheck_v3.reltr import config as _cfg

log = logging.getLogger(__name__)

# ── Device detection (resolved at import time, same as v2) ───────────────

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

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
            ckpt = torch.load(
                _cfg.RELTR_CHECKPOINT_PATH,
                map_location=DEVICE,
                weights_only=False,
            )
            model.load_state_dict(ckpt["model"])
            model.eval().to(DEVICE)

            ckpt_size = os.path.getsize(_cfg.RELTR_CHECKPOINT_PATH) / 1e6
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
