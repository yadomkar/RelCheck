"""Pluggable protocol interfaces for Caption_Editor and POPE_Responder."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from PIL.Image import Image


@runtime_checkable
class Caption_Editor(Protocol):
    """Protocol for any method that edits a caption given an image.

    Accepts either a file path string or a PIL Image, and a Ref-Cap string.
    Returns an edited caption string.
    """

    def edit_caption(self, image: str | Image, ref_cap: str) -> str:
        """Given an image and a broken caption, return an edited caption."""
        ...


@runtime_checkable
class POPE_Responder(Protocol):
    """Protocol for any method that answers a POPE yes/no question given an image.

    Accepts either a file path string or a PIL Image, and a question string.
    Returns 'yes' or 'no'.
    """

    def answer_pope(self, image: str | Image, question: str) -> str:
        """Given an image and a yes/no question, return 'yes' or 'no'."""
        ...
