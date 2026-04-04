"""
RelCheck v2 — Logging Configuration
=====================================
Single logger instance for the entire package.
Replaces scattered print() calls with structured logging.

Usage in any module:
    from ._logging import log
    log.info("Loading model on %s", DEVICE)
    log.debug("VQA votes: %dY / %dN", yes, no)
"""

from __future__ import annotations

import logging

log: logging.Logger = logging.getLogger("relcheck")
"""Package-wide logger. Configure via ``logging.basicConfig()`` or by
attaching handlers to ``logging.getLogger("relcheck")``."""

# Default: INFO to stdout so Colab notebooks see progress messages
# without any setup. Users can override with logging.basicConfig().
if not log.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    log.addHandler(_handler)
    log.setLevel(logging.INFO)
    log.propagate = False  # Prevent duplicate output in Colab (root logger also has a handler)
