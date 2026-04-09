"""RelTR configuration — mirrors the relevant v2 config constants."""

ENABLE_RELTR: bool = True
"""Master toggle for RelTR scene graph generation.
When False (default), no RelTR code paths are executed."""

RELTR_CONF_THRESHOLD: float = 0.3
"""Minimum confidence for accepting a RelTR triple.
Subject, predicate, AND object confidence must all exceed this."""

RELTR_CHECKPOINT_PATH: str = "/content/drive/MyDrive/RelCheck_Data/checkpoint0149.pth"
"""Path to the pretrained RelTR weights file."""
