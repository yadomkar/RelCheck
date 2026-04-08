"""mPLUG-Owl2-7B baseline wrapper for caption editing and POPE.

Loads MAGAer13/mplug-owl2-llama2-7b with 8-bit quantization via
the custom load_pretrained_model builder.
Provides MPLUGCaptionEditor (Caption_Editor protocol) and
MPLUGPOPEResponder (POPE_Responder protocol).

Requires the mPLUG-Owl2 package installed from the official repo:
    pip install -e . (from mPLUG-Owl/mPLUG-Owl2/)

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from PIL import Image as PILImage
from PIL.Image import Image

logger = logging.getLogger(__name__)

# Same caption editing prompt as LLaVA (Req 6.5)
CAPTION_EDIT_PROMPT = (
    "Please edit the following sentence to be consistent with the given "
    "image, making only the minimal necessary changes: {ref_cap}"
)

# Default model identifier
_MODEL_ID = "MAGAer13/mplug-owl2-llama2-7b"


class _MPLUGModel:
    """Shared mPLUG-Owl2-7B model loader.

    Loads the model once and is referenced by both MPLUGCaptionEditor
    and MPLUGPOPEResponder to avoid duplicate GPU memory usage.
    """

    def __init__(self, model_id: str = _MODEL_ID) -> None:
        """Load mPLUG-Owl2-7B with 8-bit quantization.

        Args:
            model_id: HuggingFace model identifier.

        Raises:
            RuntimeError: If CUDA is not available.
        """
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. GPU is required for mPLUG-Owl2 model "
                "inference. Please run this on a machine with a CUDA-capable GPU "
                "(e.g., Google Colab with GPU runtime)."
            )

        logger.info("Loading mPLUG-Owl2-7B model: %s", model_id)

        # Import mPLUG-Owl2 components (only available after package install)
        from mplug_owl2.constants import (  # type: ignore[import-untyped]
            DEFAULT_IMAGE_TOKEN,
            IMAGE_TOKEN_INDEX,
        )
        from mplug_owl2.conversation import conv_templates  # type: ignore[import-untyped]
        from mplug_owl2.mm_utils import (  # type: ignore[import-untyped]
            get_model_name_from_path,
            process_images,
            tokenizer_image_token,
            KeywordsStoppingCriteria,
        )
        from mplug_owl2.model.builder import load_pretrained_model  # type: ignore[import-untyped]

        # Store references for use in generate()
        self._DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self._IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self._conv_templates = conv_templates
        self._process_images = process_images
        self._tokenizer_image_token = tokenizer_image_token
        self._KeywordsStoppingCriteria = KeywordsStoppingCriteria

        # Load model with 8-bit quantization (Req 6.1, 6.2)
        model_name = get_model_name_from_path(model_id)
        self.tokenizer, self.model, self.image_processor, self.context_len = (
            load_pretrained_model(model_id, None, model_name, load_8bit=True)
        )

        # Report GPU memory usage (Req 6.3)
        mem_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        mem_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        logger.info(
            "mPLUG-Owl2 model loaded. GPU memory — allocated: %.1f MB, reserved: %.1f MB",
            mem_allocated,
            mem_reserved,
        )

    def generate(self, image: Image, prompt: str, max_new_tokens: int = 512) -> str:
        """Run inference with a single image and text prompt.

        Args:
            image: PIL Image to condition on.
            prompt: Text prompt (will be wrapped in mPLUG-Owl2 conversation format).
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Generated text string.
        """
        # Prepare conversation using mplug_owl2 template (Req 6.3)
        conv = self._conv_templates["mplug_owl2"].copy()

        # Prepend DEFAULT_IMAGE_TOKEN to query (Req 6.3)
        query = self._DEFAULT_IMAGE_TOKEN + "\n" + prompt
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        formatted_prompt = conv.get_prompt()

        # Process image: resize to square, then run through image processor
        max_edge = max(image.size)
        image_resized = image.resize((max_edge, max_edge))
        image_tensor = self._process_images([image_resized], self.image_processor)
        image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

        # Tokenize prompt
        input_ids = (
            self._tokenizer_image_token(
                formatted_prompt,
                self.tokenizer,
                self._IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            )
            .unsqueeze(0)
            .to(self.model.device)
        )

        # Generate
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                max_new_tokens=max_new_tokens,
            )

        # Decode only the generated tokens (skip the input)
        response = self.tokenizer.decode(
            output_ids[0, input_ids.shape[1]:], skip_special_tokens=True
        ).strip()

        return response


def _load_image(image: str | Image) -> Image:
    """Load an image from a file path or return a PIL Image directly.

    Args:
        image: File path string or PIL Image.

    Returns:
        PIL Image in RGB mode.

    Raises:
        FileNotFoundError: If the image path does not exist.
        TypeError: If image is neither a string nor a PIL Image.
    """
    if isinstance(image, Image):
        return image.convert("RGB")
    if isinstance(image, str):
        path = Path(image)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image}")
        return PILImage.open(path).convert("RGB")
    raise TypeError(f"Expected str or PIL Image, got {type(image)}")


class MPLUGCaptionEditor:
    """mPLUG-Owl2-7B caption editor implementing the Caption_Editor protocol.

    Uses the exact test prompt from Kim et al.:
    "Please edit the following sentence to be consistent with the given image,
    making only the minimal necessary changes: {ref_cap}"

    Requirements: 6.1, 6.2, 6.3, 6.4, 6.5
    """

    def __init__(self, mplug_model: _MPLUGModel | None = None) -> None:
        """Initialize the caption editor.

        Args:
            mplug_model: Shared mPLUG-Owl2 model instance. If None, creates a new one.

        Raises:
            RuntimeError: If CUDA is not available.
        """
        self._model = mplug_model or _MPLUGModel()

    def edit_caption(self, image: str | Image, ref_cap: str) -> str:
        """Edit a caption to be consistent with the given image.

        Args:
            image: Image file path or PIL Image.
            ref_cap: The reference caption to edit.

        Returns:
            Edited caption string.
        """
        pil_image = _load_image(image)
        prompt = CAPTION_EDIT_PROMPT.format(ref_cap=ref_cap)
        return self._model.generate(pil_image, prompt)


class MPLUGPOPEResponder:
    """mPLUG-Owl2-7B POPE responder implementing the POPE_Responder protocol.

    Passes the POPE question directly as the text prompt alongside the image.

    Requirements: 6.1, 6.2, 6.3, 6.4, 6.6
    """

    def __init__(self, mplug_model: _MPLUGModel | None = None) -> None:
        """Initialize the POPE responder.

        Args:
            mplug_model: Shared mPLUG-Owl2 model instance. If None, creates a new one.

        Raises:
            RuntimeError: If CUDA is not available.
        """
        self._model = mplug_model or _MPLUGModel()

    def answer_pope(self, image: str | Image, question: str) -> str:
        """Answer a POPE yes/no question about an image.

        Args:
            image: Image file path or PIL Image.
            question: POPE question (e.g., "Is there a cat in the image?").

        Returns:
            Model response string (pipeline handles yes/no extraction).
        """
        pil_image = _load_image(image)
        return self._model.generate(pil_image, question, max_new_tokens=64)
