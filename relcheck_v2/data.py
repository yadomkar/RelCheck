"""
RelCheck v2 — Data Loading
============================
R-Bench dataset download/parsing and image loading utilities.
Handles Colab Drive paths, caching, and format normalization.
"""

from __future__ import annotations

import json
import os
import pathlib
import random
import subprocess
import zipfile
from typing import Any

from PIL import Image

from ._logging import log
from .config import DRIVE_IMAGES_DIR, RBENCH_PATH


# ── R-Bench parsing ──────────────────────────────────────────────────


def _parse_rbench_entries(raw_list: list[dict[str, Any]]) -> dict[str, list[dict[str, str]]]:
    """Convert flat R-Bench annotation list to {img_id: [{question, answer}]}.

    Normalizes various JSON formats (image/img, text/question, label/answer keys)
    into a unified dict structure.

    Args:
        raw_list: List of annotation dicts from R-Bench JSON (format agnostic).

    Returns:
        Dict mapping image_id (filename stem) → list of {question, answer} dicts.
        Answers are normalized to lowercase strings.
    """
    rbench_data: dict[str, list[dict[str, str]]] = {}
    for entry in raw_list:
        img = entry.get("image", entry.get("img", ""))
        img_id = os.path.splitext(os.path.basename(img))[0]
        q = str(entry.get("text", entry.get("question", "")))
        a = str(entry.get("label", entry.get("answer", ""))).lower().strip()
        if img_id and q:
            rbench_data.setdefault(img_id, []).append({"question": q, "answer": a})
    return rbench_data


def load_rbench(
    rbench_path: str = RBENCH_PATH,
    download_dir: str = "/content/R-Bench",
    drive_file_id: str = "1sqO0MWBg_HXp5cIKb-nstjNEEk5crUWH",
) -> dict[str, list[dict[str, str]]]:
    """Load R-Bench data from cache or download + parse from source.

    Attempts to load from `rbench_path` (cached JSON). If not found, downloads
    the R-Bench repository and annotations via git clone + gdown, then extracts
    and parses the annotations ZIP.

    Args:
        rbench_path: Path to cached R-Bench JSON (default: config.RBENCH_PATH).
        download_dir: Directory to clone R-Bench repo into (default: /content/R-Bench).
        drive_file_id: Google Drive file ID for annotations ZIP (default: public R-Bench).

    Returns:
        Dict mapping image_id → list of {question, answer} dicts.
        Normalizes both dict and list JSON formats into unified structure.

    Raises:
        subprocess.CalledProcessError: If git clone or gdown fails.
        json.JSONDecodeError: If annotation JSON is malformed.
        zipfile.BadZipFile: If annotations ZIP is corrupted.
    """
    # ── Try cached path first ──
    if os.path.exists(rbench_path):
        with open(rbench_path) as f:
            data = json.load(f)
        # Handle both dict and list formats
        if isinstance(data, list):
            rbench_data = _parse_rbench_entries(data)
        else:
            rbench_data = data
        n_qs = sum(len(v) for v in rbench_data.values())
        log.info("Loaded R-Bench: %d images, %d questions", len(rbench_data), n_qs)
        return rbench_data

    # ── Download from source ──
    log.info("R-Bench data not found. Downloading...")
    if not os.path.exists(download_dir):
        log.debug("Cloning R-Bench repository to %s", download_dir)
        subprocess.run(
            ["git", "clone", "https://github.com/mrwu-mac/R-Bench", download_dir],
            check=True,
        )

    # ── Download annotations ZIP via gdown ──
    annotations_zip = f"{download_dir}/rbench_annotations.zip"
    log.debug("Downloading annotations ZIP from Google Drive")
    subprocess.run(
        ["gdown", "--id", drive_file_id, "-O", annotations_zip, "-q"],
        check=True,
    )

    # ── Extract and find annotation JSON ──
    log.debug("Extracting annotations from ZIP")
    with zipfile.ZipFile(annotations_zip, "r") as z:
        z.extractall(download_dir)

    # Find the right JSON: prefer "image-level" or "image_level", fallback to largest list
    all_jsons = sorted(pathlib.Path(download_dir).rglob("*.json"))
    rbench_raw: list[dict[str, Any]] | None = None
    for f in all_jsons:
        if "image-level" in f.name.lower() or "image_level" in f.name.lower():
            with open(f) as fh:
                data = json.load(fh)
            if isinstance(data, list) and len(data) > 50:
                rbench_raw = data
                break
    if rbench_raw is None:
        for f in all_jsons:
            with open(f) as fh:
                data = json.load(fh)
            if isinstance(data, list) and len(data) > 100:
                rbench_raw = data
                break

    if rbench_raw is None:
        raise ValueError("Could not find valid R-Bench annotation JSON after download")

    # ── Parse and cache ──
    rbench_data = _parse_rbench_entries(rbench_raw)
    with open(rbench_path, "w") as f:
        json.dump(rbench_data, f)
    n_qs = sum(len(v) for v in rbench_data.values())
    log.info("Saved %d images, %d questions to %s", len(rbench_data), n_qs, rbench_path)
    return rbench_data


# ── Image loading ────────────────────────────────────────────────────


def load_images(
    rbench_data: dict[str, list[dict[str, str]]],
    n_images: int = 20,
    images_dir: str = DRIVE_IMAGES_DIR,
    seed: int = 42,
) -> tuple[dict[str, Image.Image], dict[str, list[dict[str, str]]]]:
    """Load a random sample of images that have R-Bench questions.

    Samples `n_images` from the available images in `images_dir` that appear
    in the R-Bench dataset, maintaining reproducibility via `seed`.

    Args:
        rbench_data: Dict of image_id → questions (from load_rbench()).
        n_images: Number of images to sample (default: 20).
        images_dir: Directory containing nocaps .jpg images (default: config.DRIVE_IMAGES_DIR).
        seed: Random seed for reproducibility (default: 42).

    Returns:
        Tuple of:
        - pil_images: Dict mapping image_id → PIL Image (RGB format).
        - rbench_questions: Dict mapping image_id → list of {question, answer} dicts.
        Only includes successfully loaded images; errors are logged and skipped.
    """
    if not os.path.exists(images_dir):
        log.warning("Images dir not found: %s", images_dir)
        log.warning("Please download nocaps images and place in that folder.")
        return {}, {}

    # ── Find available images ──
    all_files = list(pathlib.Path(images_dir).glob("*.jpg"))
    available = {f.stem: str(f) for f in all_files}
    pool = [(img_id, qs) for img_id, qs in rbench_data.items() if img_id in available]

    if not pool:
        log.warning(
            "No images found that match R-Bench dataset. "
            "Available: %d images, R-Bench references: %d images",
            len(available),
            len(rbench_data),
        )
        return {}, {}

    # ── Sample and load ──
    random.seed(seed)
    selected = random.sample(pool, min(n_images, len(pool)))

    pil_images: dict[str, Image.Image] = {}
    rbench_questions: dict[str, list[dict[str, str]]] = {}
    for img_id, qs in selected:
        try:
            pil_images[img_id] = Image.open(available[img_id]).convert("RGB")
            rbench_questions[img_id] = qs
        except Exception as e:
            log.warning("Image load error [%s]: %s", img_id, e)

    n_qs = sum(len(v) for v in rbench_questions.values())
    log.info("Loaded %d images with %d R-Bench questions", len(pil_images), n_qs)
    return pil_images, rbench_questions
