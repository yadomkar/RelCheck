"""Disk-backed inference cache for MLLM and judge outputs.

Stores one JSON file per entry, keyed by a deterministic SHA-256 hash of
(model_id, image_id, prompt_hash).  Handles corrupted files gracefully by
logging a warning and returning ``None``.

Requirements: 2.5, 2.6, 12.1, 12.2, 12.3
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class CacheEntry(BaseModel):
    """Single cached inference result persisted as JSON on disk."""

    key: str
    value: str
    model_id: str = ""
    image_id: str = ""
    timestamp: str = ""


class InferenceCache:
    """Disk-backed cache that persists MLLM / judge outputs as JSON files.

    Each entry is stored as ``{cache_dir}/{key}.json`` containing a
    :class:`CacheEntry` serialised to JSON.

    Args:
        cache_dir: Directory where cache files are stored.  Created
            automatically if it does not exist.
    """

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str) -> str | None:
        """Return the cached value for *key*, or ``None`` on miss / corruption.

        Args:
            key: Cache key previously produced by :meth:`make_key`.

        Returns:
            The cached string value, or ``None`` if the key is not present
            or the file is corrupted.
        """
        path = self._path_for(key)
        if not path.exists():
            return None
        try:
            raw = path.read_text(encoding="utf-8")
            entry = CacheEntry.model_validate_json(raw)
            return entry.value
        except Exception:
            logger.warning("Corrupted cache file %s — removing", path)
            try:
                path.unlink()
            except OSError:
                pass
            return None

    def put(self, key: str, value: str, model_id: str = "", image_id: str = "") -> None:
        """Persist *value* to disk under *key*.

        Args:
            key: Cache key previously produced by :meth:`make_key`.
            value: The string payload to cache.
            model_id: Optional model identifier stored as metadata.
            image_id: Optional image identifier stored as metadata.
        """
        entry = CacheEntry(
            key=key,
            value=value,
            model_id=model_id,
            image_id=image_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        path = self._path_for(key)
        path.write_text(entry.model_dump_json(indent=2), encoding="utf-8")

    @staticmethod
    def make_key(model_id: str, image_id: str, prompt_hash: str) -> str:
        """Build a deterministic cache key from the three identity components.

        The key is the first 16 hex characters of
        ``sha256(model_id + "|" + image_id + "|" + prompt_hash)``.

        Args:
            model_id: Model identifier (e.g. ``"llava-hf/llava-1.5-7b-hf"``).
            image_id: Image identifier (e.g. filename or COCO id).
            prompt_hash: Hash or literal of the prompt text.

        Returns:
            A 16-character hex string suitable as a cache key.
        """
        raw = f"{model_id}|{image_id}|{prompt_hash}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _path_for(self, key: str) -> Path:
        """Return the filesystem path for a given cache key."""
        return self._cache_dir / f"{key}.json"
