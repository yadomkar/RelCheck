"""LLaVA-1.5-7B baseline wrapper for caption editing and POPE.

Loads liuhaotian/llava-v1.5-7b in 8-bit quantization via bitsandbytes.
Provides LLaVACaptionEditor (Caption_Editor protocol) and
LLaVAPOPEResponder (POPE_Responder protocol).

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from PIL import Image as PILImage
from PIL.Image import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
)

logger = logging.getLogger(__name__)

# Exact prompt template from Kim et al. for caption editing (Req 5.5)
CAPTION_EDIT_PROMPT = (
    "Please edit the following sentence to be consistent with the given "
    "image, making only the minimal necessary changes: {ref_cap}"
)

# LLaVA-1.5 conversation format
_LLAVA_PROMPT_TEMPLATE = "USER: <image>\n{prompt}\nASSISTANT:"

# Default model identifier
_MODEL_ID = "liuhaotian/llava-v1.5-7b"


class _LLaVAModel:
    """Shared LLaVA-1.5-7B model loader.

    Loads the model once and is referenced by both LLaVACaptionEditor
    and LLaVAPOPEResponder to avoid duplicate GPU memory usage.
    """

    def __init__(self, model_id: str = _MODEL_ID) -> None:
        """Load LLaVA-1.5-7B with 8-bit quantization.

        Args:
            model_id: HuggingFace model identifier.

        Raises:
            RuntimeError: If CUDA is not available.
        """
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. GPU is required for LLaVA model inference. "
                "Please run this on a machine with a CUDA-capable GPU "
                "(e.g., Google Colab with GPU runtime)."
            )

        logger.info("Loading LLaVA-1.5-7B model: %s", model_id)

        # 8-bit quantization config via bitsandbytes (Req 5.1)
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        # Load processor (image processor + tokenizer) (Req 5.2)
        self.processor = AutoProcessor.from_pretrained(model_id)

        # Load model in 8-bit (Req 5.1)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        # Report GPU memory usage (Req 5.3)
        mem_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        mem_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        logger.info(
            "LLaVA model loaded. GPU memory — allocated: %.1f MB, reserved: %.1f MB",
            mem_allocated,
            mem_reserved,
        )

    def generate(self, image: Image, prompt: str, max_new_tokens: int = 512) -> str:
        """Run inference with a single image and text prompt.

        Args:
            image: PIL Image to condition on.
            prompt: Text prompt (will be wrapped in LLaVA conversation format).
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Generated text string.
        """
        # Format prompt in LLaVA-1.5 conversation template
        formatted_prompt = _LLAVA_PROMPT_TEMPLATE.format(prompt=prompt)

        # Process inputs
        inputs = self.processor(
            text=formatted_prompt,
            images=image,
            return_tensors="pt",
        ).to(self.model.device)

        # Generate
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # Decode only the generated tokens (skip the input)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        response = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

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


class LLaVACaptionEditor:
    """LLaVA-1.5-7B caption editor implementing the Caption_Editor protocol.

    Uses the exact test prompt from Kim et al.:
    "Please edit the following sentence to be consistent with the given image,
    making only the minimal necessary changes: {ref_cap}"

    Requirements: 5.1, 5.2, 5.3, 5.4, 5.5
    """

    def __init__(self, llava_model: _LLaVAModel | None = None) -> None:
        """Initialize the caption editor.

        Args:
            llava_model: Shared LLaVA model instance. If None, creates a new one.

        Raises:
            RuntimeError: If CUDA is not available.
        """
        self._model = llava_model or _LLaVAModel()

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


class LLaVAPOPEResponder:
    """LLaVA-1.5-7B POPE responder implementing the POPE_Responder protocol.

    Passes the POPE question directly as the text prompt alongside the image.

    Requirements: 5.1, 5.2, 5.3, 5.4, 5.6
    """

    def __init__(self, llava_model: _LLaVAModel | None = None) -> None:
        """Initialize the POPE responder.

        Args:
            llava_model: Shared LLaVA model instance. If None, creates a new one.

        Raises:
            RuntimeError: If CUDA is not available.
        """
        self._model = llava_model or _LLaVAModel()

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
