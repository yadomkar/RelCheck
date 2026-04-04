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
from typing import Any

from PIL import Image

from .config import DRIVE_IMAGES_DIR, RBENCH_PATH


# ── R-Bench parsing ──────────────────────────────────────────────────

def _parse_rbench_entries(raw_list: list[dict]) -> dict[str, list[dict]]:
    """Convert flat R-Bench annotation list to {img_id: [{question, answer}]}."""
    rbench_data: dict[str, list[dict]] = {}
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
) -> dict[str, list[dict]]:
    """Load R-Bench data from cache or download + parse from source.

    Returns:
        Dict mapping image_id → list of {question, answer} dicts.
    """
    if os.path.exists(rbench_path):
        with open(rbench_path) as f:
            data = json.load(f)
        # Handle both dict and list formats
        if isinstance(data, list):
            rbench_data = _parse_rbench_entries(data)
        else:
            rbench_data = data
        n_qs = sum(len(v) for v in rbench_data.values())
        print(f"Loaded R-Bench: {len(rbench_data)} images, {n_qs} questions")
        return rbench_data

    # Download
    print("R-Bench data not found. Downloading...")
    if not os.path.exists(download_dir):
        os.system(f"git clone https://github.com/mrwu-mac/R-Bench {download_dir}")

    annotations_zip = f"{download_dir}/rbench_annotations.zip"
    os.system(f"gdown --id {drive_file_id} -O {annotations_zip} -q")

    import zipfile
    with zipfile.ZipFile(annotations_zip, "r") as z:
        z.extractall(download_dir)

    # Find the right JSON
    all_jsons = sorted(pathlib.Path(download_dir).rglob("*.json"))
    rbench_raw = None
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

    rbench_data = _parse_rbench_entries(rbench_raw)
    with open(rbench_path, "w") as f:
        json.dump(rbench_data, f)
    n_qs = sum(len(v) for v in rbench_data.values())
    print(f"Saved {len(rbench_data)} images, {n_qs} questions")
    return rbench_data


# ── Image loading ────────────────────────────────────────────────────

def load_images(
    rbench_data: dict[str, list[dict]],
    n_images: int = 20,
    images_dir: str = DRIVE_IMAGES_DIR,
    seed: int = 42,
) -> tuple[dict[str, Image.Image], dict[str, list[dict]]]:
    """Load a random sample of images that have R-Bench questions.

    Returns:
        Tuple of (pil_images dict, rbench_questions dict) for selected images.
    """
    if not os.path.exists(images_dir):
        print(f"Images dir not found: {images_dir}")
        print("Please download nocaps images and place in that folder.")
        return {}, {}

    all_files = list(pathlib.Path(images_dir).glob("*.jpg"))
    available = {f.stem: str(f) for f in all_files}
    pool = [(img_id, qs) for img_id, qs in rbench_data.items() if img_id in available]

    random.seed(seed)
    selected = random.sample(pool, min(n_images, len(pool)))

    pil_images: dict[str, Image.Image] = {}
    rbench_questions: dict[str, list[dict]] = {}
    for img_id, qs in selected:
        try:
            pil_images[img_id] = Image.open(available[img_id]).convert("RGB")
            rbench_questions[img_id] = qs
        except Exception as e:
            print(f"  [{img_id}] load error: {e}")

    n_qs = sum(len(v) for v in rbench_questions.values())
    print(f"Loaded {len(pil_images)} images with {n_qs} R-Bench questions")
    return pil_images, rbench_questions
