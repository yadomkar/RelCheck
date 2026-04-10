"""LLaVA-1.5-7B model loader.

Loads ``llava-hf/llava-1.5-7b-hf`` via the standard ``transformers`` API
with 8-bit quantization.  This is the default and primary evaluation model.

Requirements: 2.3, 2.7
"""

from __future__ import annotations

import logging

from PIL import Image

logger = logging.getLogger(__name__)

_MODEL_ID = "llava-hf/llava-1.5-7b-hf"


class LLaVA15Loader:
    """Loader for LLaVA-1.5-7B using the HuggingFace ``transformers`` API.

    Implements the :class:`~relcheck_v3.mllm.wrapper.ModelLoader` protocol.

    Attributes:
        model: The loaded ``LlavaForConditionalGeneration`` instance, or
            ``None`` before :meth:`load_model` is called.
        processor: The loaded ``AutoProcessor`` instance, or ``None``
            before :meth:`load_model` is called.
    """

    def __init__(self) -> None:
        self.model = None
        self.processor = None

    def load_model(self, cache_dir: str) -> None:
        """Download weights (if needed) and instantiate the model on GPU.

        Uses 8-bit quantization via ``bitsandbytes`` to fit within a
        single T4/A100 GPU.  Subsequent calls are no-ops if the model
        is already loaded.

        Args:
            cache_dir: Directory for caching model weights.

        Raises:
            RuntimeError: If the model fails to load.
        """
        if self.model is not None:
            logger.debug("LLaVA-1.5-7B already loaded — skipping.")
            return

        logger.info("Loading LLaVA-1.5-7B from %s …", _MODEL_ID)
        try:
            from transformers import AutoProcessor, LlavaForConditionalGeneration

            self.processor = AutoProcessor.from_pretrained(
                _MODEL_ID,
                cache_dir=cache_dir,
            )
            self.model = LlavaForConditionalGeneration.from_pretrained(
                _MODEL_ID,
                cache_dir=cache_dir,
                load_in_8bit=True,
                device_map="auto",
            )
            logger.info("LLaVA-1.5-7B loaded successfully.")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load LLaVA-1.5-7B ({_MODEL_ID!r}): {exc}"
            ) from exc

    def generate(self, image_path: str, prompt: str) -> str:
        """Run inference on a single image with the given prompt.

        Args:
            image_path: Absolute path to the input image.
            prompt: Text prompt for the model.

        Returns:
            The model's decoded text output.

        Raises:
            RuntimeError: If the model has not been loaded yet.
        """
        if self.model is None or self.processor is None:
            raise RuntimeError(
                "Model not loaded. Call load_model() before generate()."
            )

        image = Image.open(image_path).convert("RGB")

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        text_prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )

        inputs = self.processor(
            text=text_prompt, images=image, return_tensors="pt"
        ).to(self.model.device)

        output_ids = self.model.generate(**inputs, max_new_tokens=512)

        # Decode only the newly generated tokens (skip the input tokens).
        generated_ids = output_ids[:, inputs["input_ids"].shape[-1] :]
        result = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

        logger.debug("LLaVA-1.5 output (%d chars): %.100s…", len(result), result)
        return result
