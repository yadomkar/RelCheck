"""MiniGPT-4 model loader.

Loads ``Vision-CAIR/MiniGPT-4`` using the original repository's model
loading utilities.  Requires Vicuna-13B v0 weights and a custom MiniGPT-4
checkpoint.  The repo must be cloned and configured before calling
:meth:`load_model` — this is handled by
:func:`relcheck_v3.mllm.setup.setup_minigpt4`.

Requirements: 2.4, 2.7, 2.8
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

_MODEL_ID = "Vision-CAIR/MiniGPT-4"
_MINIGPT4_REPO_DIR = Path("/content/MiniGPT-4")
_EVAL_YAML = _MINIGPT4_REPO_DIR / "eval_configs" / "minigpt4_eval.yaml"


class MiniGPT4Loader:
    """Loader for MiniGPT-4 using the Vision-CAIR/MiniGPT-4 codebase.

    Implements the :class:`~relcheck_v3.mllm.wrapper.ModelLoader` protocol.

    The ``Vision-CAIR/MiniGPT-4`` repository must be cloned to
    ``/content/MiniGPT-4`` with Vicuna-13B v0 weights and the MiniGPT-4
    checkpoint downloaded and configured.  Call
    :func:`relcheck_v3.mllm.setup.setup_minigpt4` to set this up.

    Attributes:
        model: The loaded MiniGPT-4 model, or ``None`` before loading.
        chat: The MiniGPT-4 ``Chat`` instance for inference, or ``None``.
        vis_processor: The visual processor, or ``None`` before loading.
    """

    def __init__(self) -> None:
        self.model = None
        self.chat = None
        self.vis_processor = None

    def load_model(self, cache_dir: str) -> None:
        """Download weights (if needed) and instantiate the model on GPU.

        Adds the MiniGPT-4 repo to ``sys.path``, imports their model
        loading utilities, and loads the model using the eval config YAML.
        Subsequent calls are no-ops if the model is already loaded.

        Args:
            cache_dir: Directory for caching model weights (used by
                setup, not directly by this loader since MiniGPT-4 uses
                its own config-driven weight paths).

        Raises:
            RuntimeError: If the MiniGPT-4 repo is not found or the
                model fails to load.
        """
        if self.model is not None:
            logger.debug("MiniGPT-4 already loaded — skipping.")
            return

        if not _MINIGPT4_REPO_DIR.exists():
            raise RuntimeError(
                f"MiniGPT-4 repo not found at {_MINIGPT4_REPO_DIR}. "
                "Run setup_minigpt4() first."
            )

        # Add repo to sys.path for custom imports.
        repo_str = str(_MINIGPT4_REPO_DIR)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)
            logger.info("Added %s to sys.path.", repo_str)

        logger.info("Loading MiniGPT-4 …")
        try:
            from minigpt4.common.config import Config
            from minigpt4.common.registry import registry
            from minigpt4.conversation.conversation import Chat

            # Build config from the eval YAML (paths already configured
            # by setup_minigpt4).
            cfg = Config(
                _EVAL_YAML,
                options=None,
            )

            model_config = cfg.model_cfg
            model_cls = registry.get_model_class(model_config.arch)
            self.model = model_cls.from_config(model_config)
            self.model = self.model.to("cuda").eval()

            vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
            vis_processor_cls = registry.get_processor_class(
                vis_processor_cfg.name
            )
            self.vis_processor = vis_processor_cls.from_config(vis_processor_cfg)

            self.chat = Chat(self.model, self.vis_processor, device="cuda:0")

            logger.info("MiniGPT-4 loaded successfully.")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load MiniGPT-4 ({_MODEL_ID!r}): {exc}"
            ) from exc

    def generate(self, image_path: str, prompt: str) -> str:
        """Run inference on a single image with the given prompt.

        Uses MiniGPT-4's ``Chat`` interface for multi-turn conversation.
        Each call creates a fresh conversation context.

        Args:
            image_path: Absolute path to the input image.
            prompt: Text prompt for the model.

        Returns:
            The model's decoded text output.

        Raises:
            RuntimeError: If the model has not been loaded yet.
        """
        if self.chat is None:
            raise RuntimeError(
                "Model not loaded. Call load_model() before generate()."
            )

        try:
            from minigpt4.conversation.conversation import CONV_VISION

            image = Image.open(image_path).convert("RGB")

            # Create a fresh conversation for each inference call.
            chat_state = CONV_VISION.copy()
            img_list = []

            # Upload image to the model's internal state.
            self.chat.upload_img(image, chat_state, img_list)

            # Ask the question.
            self.chat.ask(prompt, chat_state)

            # Generate the response.
            result = self.chat.answer(
                conv=chat_state,
                img_list=img_list,
                max_new_tokens=512,
                max_length=2000,
            )[0]

            logger.debug(
                "MiniGPT-4 output (%d chars): %.100s…", len(result), result
            )
            return result
        except Exception as exc:
            logger.error("MiniGPT-4 inference failed: %s", exc, exc_info=True)
            raise
