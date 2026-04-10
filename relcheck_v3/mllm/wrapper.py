"""Swappable MLLM inference wrapper with disk-backed caching.

Provides a unified interface for generating image descriptions and answering
yes/no questions across four supported MLLMs.  Uses a strategy pattern to
dispatch to model-specific loaders and integrates :class:`InferenceCache` so
every output is persisted to disk keyed by ``(model_id, image_id, prompt_hash)``.

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Protocol, runtime_checkable

from relcheck_v3.mllm.cache import InferenceCache

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Valid model identifiers
# ---------------------------------------------------------------------------

VALID_MODEL_IDS: set[str] = {
    "llava-hf/llava-1.5-7b-hf",
    "liuhaotian/llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3",
    "MAGAer13/mplug-owl-llama-7b",
    "Vision-CAIR/MiniGPT-4",
}

# ---------------------------------------------------------------------------
# Prompt constants
# ---------------------------------------------------------------------------

DESCRIBE_PROMPT = "Describe this image in detail."
YESNO_PROMPT_TEMPLATE = "Answer the following question with yes or no: {question}"


# ---------------------------------------------------------------------------
# ModelLoader protocol — loaders created in tasks 1.5, 1.5b, 1.6, 1.7
# ---------------------------------------------------------------------------


@runtime_checkable
class ModelLoader(Protocol):
    """Protocol that model-specific loaders must implement.

    Each loader handles weight downloading, model instantiation, and
    inference for a single MLLM architecture.
    """

    def load_model(self, cache_dir: str) -> None:
        """Download weights (if needed) and instantiate the model on GPU.

        Args:
            cache_dir: Directory for caching model weights.

        Raises:
            RuntimeError: If the model fails to load.
        """
        ...

    def generate(self, image_path: str, prompt: str) -> str:
        """Run inference on a single image with the given prompt.

        Args:
            image_path: Absolute path to the input image.
            prompt: Text prompt for the model.

        Returns:
            The model's text output.
        """
        ...


# ---------------------------------------------------------------------------
# Loader factory — lazy imports to avoid pulling in heavy deps at module level
# ---------------------------------------------------------------------------


def _get_loader(model_id: str) -> ModelLoader:
    """Return the appropriate :class:`ModelLoader` for *model_id*.

    Uses lazy imports so that model-specific dependencies (custom repos,
    heavy libraries) are only loaded when actually needed.

    Args:
        model_id: One of the identifiers in :data:`VALID_MODEL_IDS`.

    Returns:
        An instance of the loader for the requested model.

    Raises:
        ValueError: If *model_id* is not recognised.
    """
    if model_id == "llava-hf/llava-1.5-7b-hf":
        from relcheck_v3.mllm.loaders.llava15 import LLaVA15Loader

        return LLaVA15Loader()

    if model_id == "liuhaotian/llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3":
        from relcheck_v3.mllm.loaders.llava_v1 import LLaVAV1Loader

        return LLaVAV1Loader()

    if model_id == "MAGAer13/mplug-owl-llama-7b":
        from relcheck_v3.mllm.loaders.mplug_owl import MplugOwlLoader

        return MplugOwlLoader()

    if model_id == "Vision-CAIR/MiniGPT-4":
        from relcheck_v3.mllm.loaders.minigpt4 import MiniGPT4Loader

        return MiniGPT4Loader()

    valid = ", ".join(sorted(VALID_MODEL_IDS))
    raise ValueError(
        f"Unknown model_id {model_id!r}. Valid options: {valid}"
    )


# ---------------------------------------------------------------------------
# MLLMWrapper
# ---------------------------------------------------------------------------


class MLLMWrapper:
    """Unified MLLM inference wrapper with disk-backed caching.

    Supports four MLLMs via a strategy pattern that dispatches to
    model-specific loaders.  All outputs are cached to disk so that
    repeated calls with the same inputs are free.

    Args:
        model_id: HuggingFace model identifier.  Must be one of
            :data:`VALID_MODEL_IDS`.  Defaults to LLaVA-1.5-7B.
        cache_dir: Directory for storing downloaded model weights.
        output_cache_dir: Directory for the :class:`InferenceCache`
            that persists inference outputs.

    Raises:
        ValueError: If *model_id* is not in :data:`VALID_MODEL_IDS`.
        RuntimeError: If CUDA is not available or the model fails to load.
    """

    def __init__(
        self,
        model_id: str = "llava-hf/llava-1.5-7b-hf",
        cache_dir: str = "/content/weights/",
        output_cache_dir: str = "cache/mllm/",
    ) -> None:
        if model_id not in VALID_MODEL_IDS:
            valid = ", ".join(sorted(VALID_MODEL_IDS))
            raise ValueError(
                f"Unknown model_id {model_id!r}. Valid options: {valid}"
            )

        self._model_id = model_id
        self._cache_dir = cache_dir
        self._output_cache = InferenceCache(Path(output_cache_dir))
        self._loader: ModelLoader | None = None

        # Validate CUDA availability eagerly so callers get a clear error.
        self._check_cuda()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def describe(self, image_path: str) -> str:
        """Generate a captioning-style description of *image_path*.

        The result is cached by ``(model_id, image_id, prompt_hash)`` so
        subsequent calls with the same image return instantly.

        Args:
            image_path: Path to the input image file.

        Returns:
            A text description of the image, or an empty string if
            inference fails for this sample.
        """
        image_id = _extract_image_id(image_path)
        prompt_hash = _hash_prompt(DESCRIBE_PROMPT)
        cache_key = InferenceCache.make_key(self._model_id, image_id, prompt_hash)

        cached = self._output_cache.get(cache_key)
        if cached is not None:
            logger.debug("Cache hit for describe(%s)", image_path)
            return cached

        return self._run_inference(image_path, DESCRIBE_PROMPT, cache_key, image_id)

    def answer_yesno(self, image_path: str, question: str) -> str:
        """Answer a yes/no *question* about the image at *image_path*.

        The result is cached by ``(model_id, image_id, question_hash)``
        so subsequent calls with the same inputs return instantly.

        Args:
            image_path: Path to the input image file.
            question: The yes/no question to answer.

        Returns:
            The model's response string, or an empty string if inference
            fails for this sample.
        """
        image_id = _extract_image_id(image_path)
        prompt = YESNO_PROMPT_TEMPLATE.format(question=question)
        prompt_hash = _hash_prompt(prompt)
        cache_key = InferenceCache.make_key(self._model_id, image_id, prompt_hash)

        cached = self._output_cache.get(cache_key)
        if cached is not None:
            logger.debug("Cache hit for answer_yesno(%s, ...)", image_path)
            return cached

        return self._run_inference(image_path, prompt, cache_key, image_id)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _check_cuda() -> None:
        """Raise :class:`RuntimeError` if CUDA is not available."""
        try:
            import torch  # noqa: WPS433 — lazy import intentional
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch is required but not installed."
            ) from exc

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. A GPU is required for MLLM inference."
            )

    def _ensure_model_loaded(self) -> ModelLoader:
        """Lazily instantiate and load the model on first use.

        Returns:
            The initialised :class:`ModelLoader`.

        Raises:
            RuntimeError: If the model fails to load.
        """
        if self._loader is not None:
            return self._loader

        try:
            loader = _get_loader(self._model_id)
            loader.load_model(self._cache_dir)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load model {self._model_id!r}: {exc}"
            ) from exc

        self._loader = loader
        return self._loader

    def _run_inference(
        self,
        image_path: str,
        prompt: str,
        cache_key: str,
        image_id: str,
    ) -> str:
        """Run model inference, cache the result, and return it.

        On per-sample failure the error is logged and an empty string is
        returned so that the evaluation loop can continue.

        Args:
            image_path: Path to the input image.
            prompt: Text prompt for the model.
            cache_key: Pre-computed cache key.
            image_id: Image identifier for cache metadata.

        Returns:
            Model output string, or ``""`` on failure.
        """
        try:
            loader = self._ensure_model_loaded()
            result = loader.generate(image_path, prompt)
        except RuntimeError:
            # Model-load failures are fatal — re-raise.
            raise
        except Exception:
            logger.error(
                "Inference failed for model=%s image=%s",
                self._model_id,
                image_path,
                exc_info=True,
            )
            return ""

        self._output_cache.put(
            cache_key, result, model_id=self._model_id, image_id=image_id
        )
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_image_id(image_path: str) -> str:
    """Derive a short image identifier from a file path.

    Args:
        image_path: Absolute or relative path to an image file.

    Returns:
        The filename stem (no directory, no extension).
    """
    return os.path.splitext(os.path.basename(image_path))[0]


def _hash_prompt(prompt: str) -> str:
    """Return the SHA-256 hex digest of *prompt*.

    Args:
        prompt: The prompt text to hash.

    Returns:
        Full 64-character hex digest.
    """
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()
