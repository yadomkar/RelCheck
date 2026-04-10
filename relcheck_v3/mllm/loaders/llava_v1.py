"""LLaVA v1 (13B) model loader.

Loads ``liuhaotian/llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3`` using
the original LLaVA codebase (``liuhaotian/LLaVA`` repo).  The repo must
be cloned and on ``sys.path`` before calling :meth:`load_model` — this is
handled by :func:`relcheck_v3.mllm.setup.setup_llava_v1`.

Requirements: 2.4, 2.7, 2.8
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

_MODEL_ID = "liuhaotian/llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3"
_LLAVA_REPO_DIR = Path("/content/LLaVA")


class LLaVAV1Loader:
    """Loader for LLaVA v1 13B using the original LLaVA codebase.

    Implements the :class:`~relcheck_v3.mllm.wrapper.ModelLoader` protocol.

    The ``liuhaotian/LLaVA`` repository must be cloned to
    ``/content/LLaVA`` and added to ``sys.path`` before use.  Call
    :func:`relcheck_v3.mllm.setup.setup_llava_v1` to set this up.

    Attributes:
        model: The loaded LLaVA v1 model, or ``None`` before loading.
        tokenizer: The loaded tokenizer, or ``None`` before loading.
        image_processor: The loaded image processor, or ``None``.
        conv_mode: Conversation template mode for this model variant.
    """

    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.conv_mode: str = "llava_v1"

    def load_model(self, cache_dir: str) -> None:
        """Download weights (if needed) and instantiate the model on GPU.

        Ensures the LLaVA repo is on ``sys.path``, then uses the repo's
        custom model loading utilities.  Uses 8-bit quantization.
        Subsequent calls are no-ops if the model is already loaded.

        Args:
            cache_dir: Directory for caching model weights.

        Raises:
            RuntimeError: If the LLaVA repo is not found or the model
                fails to load.
        """
        if self.model is not None:
            logger.debug("LLaVA v1 already loaded — skipping.")
            return

        # Ensure LLaVA repo is on sys.path.
        repo_str = str(_LLAVA_REPO_DIR)
        if not _LLAVA_REPO_DIR.exists():
            raise RuntimeError(
                f"LLaVA repo not found at {_LLAVA_REPO_DIR}. "
                "Run setup_llava_v1() first."
            )
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)
            logger.info("Added %s to sys.path.", repo_str)

        logger.info("Loading LLaVA v1 from %s …", _MODEL_ID)
        try:
            from llava.model import LlavaLlamaForCausalLM
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                _MODEL_ID,
                cache_dir=cache_dir,
                use_fast=False,
            )
            self.model = LlavaLlamaForCausalLM.from_pretrained(
                _MODEL_ID,
                cache_dir=cache_dir,
                torch_dtype="float16",
                device_map="auto",
            )

            # The LLaVA v1 repo stores the vision tower config on the model.
            vision_tower = self.model.get_vision_tower()
            if vision_tower is not None and not vision_tower.is_loaded:
                vision_tower.load_model()
            self.image_processor = vision_tower.image_processor if vision_tower else None

            logger.info("LLaVA v1 loaded successfully.")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load LLaVA v1 ({_MODEL_ID!r}): {exc}"
            ) from exc

    def generate(self, image_path: str, prompt: str) -> str:
        """Run inference on a single image with the given prompt.

        Uses the LLaVA v1 conversation template and inference utilities
        from the cloned repository.

        Args:
            image_path: Absolute path to the input image.
            prompt: Text prompt for the model.

        Returns:
            The model's decoded text output.

        Raises:
            RuntimeError: If the model has not been loaded yet.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "Model not loaded. Call load_model() before generate()."
            )

        import torch
        from llava.constants import (
            DEFAULT_IMAGE_TOKEN,
            IMAGE_TOKEN_INDEX,
        )
        from llava.conversation import conv_templates
        from llava.mm_utils import (
            process_images,
            tokenizer_image_token,
        )

        image = Image.open(image_path).convert("RGB")

        # Build conversation using the LLaVA v1 template.
        conv = conv_templates[self.conv_mode].copy()
        image_prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        conv.append_message(conv.roles[0], image_prompt)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(
                full_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(self.model.device)
        )

        # Process image through the vision tower's processor.
        if self.image_processor is not None:
            image_tensor = process_images(
                [image], self.image_processor, self.model.config
            ).to(dtype=self.model.dtype, device=self.model.device)
        else:
            image_tensor = None

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                max_new_tokens=512,
                do_sample=False,
            )

        result = self.tokenizer.decode(
            output_ids[0, input_ids.shape[1] :], skip_special_tokens=True
        ).strip()

        logger.debug("LLaVA v1 output (%d chars): %.100s…", len(result), result)
        return result
