"""Model setup and download helpers for supported MLLMs.

Provides idempotent setup functions for each of the four supported MLLMs.
Each function downloads model weights, clones required repositories, and
configures the environment as needed.  All functions skip work that has
already been completed and log progress via the :mod:`logging` module.

Requirements: 2.4, 2.7, 2.8
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_WEIGHTS_DIR = Path("/content/weights/")

_LLAVA15_MODEL_ID = "llava-hf/llava-1.5-7b-hf"

_LLAVA_V1_REPO_URL = "https://github.com/haotian-liu/LLaVA.git"
_LLAVA_V1_REPO_DIR = Path("/content/LLaVA")
_LLAVA_V1_MODEL_ID = (
    "liuhaotian/llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3"
)

_MPLUG_OWL_REPO_URL = "https://github.com/X-PLUG/mPLUG-Owl.git"
_MPLUG_OWL_REPO_DIR = Path("/content/mPLUG-Owl")
_MPLUG_OWL_MODEL_ID = "MAGAer13/mplug-owl-llama-7b"

_MINIGPT4_REPO_URL = "https://github.com/Vision-CAIR/MiniGPT-4.git"
_MINIGPT4_REPO_DIR = Path("/content/MiniGPT-4")
_VICUNA_V0_MODEL_ID = "lmsys/vicuna-13b-delta-v0"
_MINIGPT4_CHECKPOINT_URL = (
    "https://drive.google.com/uc?id=1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R"
)
_MINIGPT4_CHECKPOINT_PATH = _WEIGHTS_DIR / "pretrained_minigpt4_7b.pth"
_MINIGPT4_EVAL_YAML = (
    _MINIGPT4_REPO_DIR / "eval_configs" / "minigpt4_eval.yaml"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_cmd(cmd: list[str], description: str) -> None:
    """Run a shell command, raising on failure.

    Args:
        cmd: Command and arguments to execute.
        description: Human-readable description for log messages.

    Raises:
        RuntimeError: If the command exits with a non-zero return code.
    """
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        logger.error(
            "%s failed (rc=%d):\nstdout: %s\nstderr: %s",
            description,
            result.returncode,
            result.stdout[-500:] if result.stdout else "",
            result.stderr[-500:] if result.stderr else "",
        )
        raise RuntimeError(
            f"{description} failed with return code {result.returncode}: "
            f"{result.stderr[-300:] if result.stderr else 'no stderr'}"
        )
    logger.info("%s completed successfully.", description)


def _clone_repo(repo_url: str, target_dir: Path) -> None:
    """Clone a git repository if the target directory does not exist.

    Args:
        repo_url: HTTPS URL of the git repository.
        target_dir: Local directory to clone into.

    Raises:
        RuntimeError: If ``git clone`` fails.
    """
    if target_dir.exists():
        logger.info("Repository already cloned at %s — skipping.", target_dir)
        return
    _run_cmd(
        ["git", "clone", repo_url, str(target_dir)],
        f"git clone {repo_url}",
    )


def _download_hf_model(model_id: str, cache_dir: Path) -> None:
    """Download a HuggingFace model snapshot to *cache_dir*.

    Uses ``huggingface_hub.snapshot_download`` so that only missing files
    are fetched on subsequent calls (idempotent).

    Args:
        model_id: HuggingFace model identifier (e.g. ``"llava-hf/llava-1.5-7b-hf"``).
        cache_dir: Local directory used as the HF cache root.

    Raises:
        RuntimeError: If the download fails.
    """
    try:
        from huggingface_hub import snapshot_download  # noqa: WPS433
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for model downloads. "
            "Install with: pip install huggingface_hub"
        ) from exc

    logger.info(
        "Downloading HuggingFace model %s to %s …", model_id, cache_dir
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=model_id,
            cache_dir=str(cache_dir),
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download model {model_id!r}: {exc}"
        ) from exc
    logger.info("Model %s downloaded successfully.", model_id)


# ---------------------------------------------------------------------------
# Public setup functions
# ---------------------------------------------------------------------------


def setup_llava15(weights_dir: Path = _WEIGHTS_DIR) -> None:
    """Download LLaVA-1.5-7B weights to the standard HF cache.

    Uses ``huggingface_hub.snapshot_download`` which is idempotent — files
    already present are not re-downloaded.

    Args:
        weights_dir: Directory for caching model weights.
            Defaults to ``/content/weights/``.

    Raises:
        RuntimeError: If the download fails.
    """
    logger.info("Setting up LLaVA-1.5-7B …")
    _download_hf_model(_LLAVA15_MODEL_ID, weights_dir)
    logger.info("LLaVA-1.5-7B setup complete.")


def setup_llava_v1(
    weights_dir: Path = _WEIGHTS_DIR,
    repo_dir: Path = _LLAVA_V1_REPO_DIR,
) -> None:
    """Clone the LLaVA v1 repository and download model weights.

    Steps:
        1. Clone ``liuhaotian/LLaVA`` to *repo_dir* (skipped if exists).
        2. Download ``liuhaotian/llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3``
           weights to *weights_dir*.
        3. Add *repo_dir* to :data:`sys.path` so the custom model code is
           importable.

    Args:
        weights_dir: Directory for caching model weights.
            Defaults to ``/content/weights/``.
        repo_dir: Directory to clone the LLaVA repository into.
            Defaults to ``/content/LLaVA``.

    Raises:
        RuntimeError: If cloning or downloading fails.
    """
    logger.info("Setting up LLaVA v1 …")

    # 1. Clone repo
    _clone_repo(_LLAVA_V1_REPO_URL, repo_dir)

    # 2. Install the LLaVA package in editable mode WITHOUT its pinned deps
    #    (the repo pins torch==2.1.2 which is unavailable on modern Colab)
    _run_cmd(
        [sys.executable, "-m", "pip", "install", "-e", str(repo_dir),
         "--no-deps", "-q"],
        "pip install -e LLaVA (no-deps)",
    )

    # 3. Download weights
    _download_hf_model(_LLAVA_V1_MODEL_ID, weights_dir)

    # 4. Add repo to sys.path for custom imports
    repo_str = str(repo_dir)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
        logger.info("Added %s to sys.path.", repo_str)
    else:
        logger.info("%s already in sys.path.", repo_str)

    logger.info("LLaVA v1 setup complete.")


def setup_mplug_owl(
    weights_dir: Path = _WEIGHTS_DIR,
    repo_dir: Path = _MPLUG_OWL_REPO_DIR,
) -> None:
    """Clone the mPLUG-Owl repository, install it, and download weights.

    Steps:
        1. Clone ``X-PLUG/mPLUG-Owl`` to *repo_dir* (skipped if exists).
        2. Run ``pip install -e mPLUG-Owl/mPLUG-Owl`` for the editable
           install of the inner package.
        3. Download ``MAGAer13/mplug-owl-llama-7b`` weights to *weights_dir*.

    Args:
        weights_dir: Directory for caching model weights.
            Defaults to ``/content/weights/``.
        repo_dir: Directory to clone the mPLUG-Owl repository into.
            Defaults to ``/content/mPLUG-Owl``.

    Raises:
        RuntimeError: If cloning, installation, or downloading fails.
    """
    logger.info("Setting up mPLUG-Owl …")

    # 1. Clone repo
    _clone_repo(_MPLUG_OWL_REPO_URL, repo_dir)

    # 2. pip install -e the inner package
    inner_pkg = repo_dir / "mPLUG-Owl"
    if inner_pkg.exists():
        _run_cmd(
            [sys.executable, "-m", "pip", "install", "-e", str(inner_pkg)],
            "pip install -e mPLUG-Owl/mPLUG-Owl",
        )
    else:
        raise RuntimeError(
            f"Expected inner package directory not found: {inner_pkg}"
        )

    # 3. Download weights
    _download_hf_model(_MPLUG_OWL_MODEL_ID, weights_dir)

    logger.info("mPLUG-Owl setup complete.")


def setup_minigpt4(
    weights_dir: Path = _WEIGHTS_DIR,
    repo_dir: Path = _MINIGPT4_REPO_DIR,
) -> None:
    """Clone the MiniGPT-4 repository, download weights, and configure paths.

    Steps:
        1. Clone ``Vision-CAIR/MiniGPT-4`` to *repo_dir* (skipped if exists).
        2. Download Vicuna-13B v0 delta weights
           (``lmsys/vicuna-13b-delta-v0``) to *weights_dir*.
        3. Download the MiniGPT-4 pre-trained checkpoint to *weights_dir*.
        4. Update the MiniGPT-4 eval config YAML with the correct weight
           paths so the model can be loaded without manual edits.

    Args:
        weights_dir: Directory for caching model weights.
            Defaults to ``/content/weights/``.
        repo_dir: Directory to clone the MiniGPT-4 repository into.
            Defaults to ``/content/MiniGPT-4``.

    Raises:
        RuntimeError: If cloning, downloading, or configuration fails.
    """
    logger.info("Setting up MiniGPT-4 …")

    # 1. Clone repo
    _clone_repo(_MINIGPT4_REPO_URL, repo_dir)

    # 2. Download Vicuna-13B v0 weights
    _download_hf_model(_VICUNA_V0_MODEL_ID, weights_dir)

    # 3. Download MiniGPT-4 checkpoint
    checkpoint_path = weights_dir / _MINIGPT4_CHECKPOINT_PATH.name
    if checkpoint_path.exists():
        logger.info(
            "MiniGPT-4 checkpoint already exists at %s — skipping.",
            checkpoint_path,
        )
    else:
        logger.info("Downloading MiniGPT-4 checkpoint …")
        try:
            import gdown  # noqa: WPS433
        except ImportError as exc:
            raise RuntimeError(
                "gdown is required to download the MiniGPT-4 checkpoint. "
                "Install with: pip install gdown"
            ) from exc
        try:
            gdown.download(
                _MINIGPT4_CHECKPOINT_URL,
                str(checkpoint_path),
                quiet=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download MiniGPT-4 checkpoint: {exc}"
            ) from exc
        if not checkpoint_path.exists():
            raise RuntimeError(
                f"MiniGPT-4 checkpoint download did not produce {checkpoint_path}"
            )
        logger.info("MiniGPT-4 checkpoint downloaded to %s.", checkpoint_path)

    # 4. Configure eval YAML with correct paths
    _configure_minigpt4_yaml(
        yaml_path=repo_dir / "eval_configs" / "minigpt4_eval.yaml",
        vicuna_weights_dir=weights_dir,
        checkpoint_path=checkpoint_path,
    )

    logger.info("MiniGPT-4 setup complete.")


def _configure_minigpt4_yaml(
    yaml_path: Path,
    vicuna_weights_dir: Path,
    checkpoint_path: Path,
) -> None:
    """Update the MiniGPT-4 eval config YAML with correct weight paths.

    Reads the YAML file, updates the ``llama_model`` and ``ckpt`` fields,
    and writes it back.  If the YAML file does not exist (repo not yet
    cloned), this is a no-op with a warning.

    Args:
        yaml_path: Path to the MiniGPT-4 eval config YAML.
        vicuna_weights_dir: Directory containing Vicuna-13B v0 weights.
        checkpoint_path: Path to the MiniGPT-4 pre-trained checkpoint.
    """
    if not yaml_path.exists():
        logger.warning(
            "MiniGPT-4 eval config not found at %s — skipping YAML update.",
            yaml_path,
        )
        return

    logger.info("Updating MiniGPT-4 eval config at %s …", yaml_path)
    content = yaml_path.read_text(encoding="utf-8")

    # Build the HF cache path for Vicuna weights.  snapshot_download stores
    # models under cache_dir/models--{org}--{name}/snapshots/{hash}/.
    # We point to the top-level cache dir and let MiniGPT-4's loader resolve.
    vicuna_path = str(vicuna_weights_dir)
    ckpt_path = str(checkpoint_path)

    # Replace known placeholder patterns in the YAML.
    # The default YAML has lines like:
    #   llama_model: "/path/to/vicuna/"
    #   ckpt: "/path/to/pretrained/ckpt/"
    content = re.sub(
        r'(llama_model:\s*")[^"]*(")',
        rf"\g<1>{vicuna_path}\g<2>",
        content,
    )
    content = re.sub(
        r'(ckpt:\s*")[^"]*(")',
        rf"\g<1>{ckpt_path}\g<2>",
        content,
    )

    yaml_path.write_text(content, encoding="utf-8")
    logger.info("MiniGPT-4 eval config updated.")
