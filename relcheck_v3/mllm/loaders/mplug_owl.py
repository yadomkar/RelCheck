"""mPLUG-Owl v1 (7B) model loader.

Loads ``MAGAer13/mplug-owl-llama-7b`` using the ``X-PLUG/mPLUG-Owl``
repository's custom model classes.  The repo must be cloned and installed
before calling :meth:`load_model` — this is handled by
:func:`relcheck_v3.mllm.setup.setup_mplug_owl`.

Requirements: 2.4, 2.7, 2.8
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

_MODEL_ID = "MAGAer13/mplug-owl-llama-7b"
_MPLUG_OWL_REPO_DIR = Path("/content/mPLUG-Owl")


class MplugOwlLoader:
    """Loader for mPLUG-Owl v1 7B using the X-PLUG/mPLUG-Owl codebase.

    Implements the :class:`~relcheck_v3.mllm.wrapper.ModelLoader` protocol.

    The ``X-PLUG/mPLUG-Owl`` repository must be cloned to
    ``/content/mPLUG-Owl`` and pip-installed before use.  Call
    :func:`relcheck_v3.mllm.setup.setup_mplug_owl` to set this up.

    Attributes:
        model: The loaded ``MplugOwlForConditionalGeneration`` instance,
            or ``None`` before loading.
        tokenizer: The loaded tokenizer, or ``None`` before loading.
        image_processor: The loaded image processor, or ``None``.
    """

    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.image_processor = None

    def load_model(self, cache_dir: str) -> None:
        """Download weights (if needed) and instantiate the model on GPU.

        Imports from the mPLUG-Owl package and loads the model with 8-bit
        quantization.  Subsequent calls are no-ops if the model is already
        loaded.

        Args:
            cache_dir: Directory for caching model weights.

        Raises:
            RuntimeError: If the mPLUG-Owl package is not installed or
                the model fails to load.
        """
        if self.model is not None:
            logger.debug("mPLUG-Owl already loaded — skipping.")
            return

        # Ensure the mPLUG-Owl inner package directory is on sys.path
        # as a fallback if the pip install didn't fully register.
        inner_pkg = _MPLUG_OWL_REPO_DIR / "mPLUG-Owl"
        inner_str = str(inner_pkg)
        if inner_pkg.exists() and inner_str not in sys.path:
            sys.path.insert(0, inner_str)
            logger.info("Added %s to sys.path.", inner_str)

        logger.info("Loading mPLUG-Owl from %s …", _MODEL_ID)
        try:
            from mplug_owl.modeling_mplug_owl import (
                MplugOwlForConditionalGeneration,
            )
            from mplug_owl.processing_mplug_owl import (
                MplugOwlImageProcessor,
                MplugOwlProcessor,
            )
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                _MODEL_ID,
                cache_dir=cache_dir,
            )
            self.image_processor = MplugOwlImageProcessor.from_pretrained(
                _MODEL_ID,
                cache_dir=cache_dir,
            )
            self.model = MplugOwlForConditionalGeneration.from_pretrained(
                _MODEL_ID,
                cache_dir=cache_dir,
                load_in_8bit=True,
                device_map="auto",
            )
            self._processor = MplugOwlProcessor(
                self.image_processor, self.tokenizer
            )

            logger.info("mPLUG-Owl loaded successfully.")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load mPLUG-Owl ({_MODEL_ID!r}): {exc}"
            ) from exc

    def generate(self, image_path: str, prompt: str) -> str:
        """Run inference on a single image with the given prompt.

        Uses the mPLUG-Owl conversation format and processor.

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

        image = Image.open(image_path).convert("RGB")

        # mPLUG-Owl uses a specific prompt template with <image> token.
        owl_prompt = (
            "The following is a conversation between a curious human and AI "
            "assistant. The assistant gives helpful, detailed, and polite "
            "answers to the human's questions.\n"
            f"Human: <image>\n{prompt}\n"
            "AI: "
        )

        inputs = self._processor(
            text=[owl_prompt], images=[image], return_tensors="pt"
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                num_beams=1,
            )

        result = self.tokenizer.decode(
            output_ids[0], skip_special_tokens=True
        ).strip()

        # Strip the prompt prefix from the output if present.
        ai_marker = "AI: "
        if ai_marker in result:
            result = result.split(ai_marker)[-1].strip()

        logger.debug(
            "mPLUG-Owl output (%d chars): %.100s…", len(result), result
        )
        return result
