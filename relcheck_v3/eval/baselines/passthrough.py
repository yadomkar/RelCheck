"""Passthrough caption editor — returns Ref-Cap unchanged for baseline scoring."""

from __future__ import annotations

from PIL.Image import Image


class PassthroughCaptionEditor:
    """Caption editor that returns the Ref-Cap unchanged.

    Used to compute the Ref-Caps baseline row in Table 2 of Kim et al.
    Implements the Caption_Editor protocol from interfaces.py.
    """

    def edit_caption(self, image: str | Image, ref_cap: str) -> str:
        """Return the Ref-Cap unchanged.

        Args:
            image: Image file path or PIL Image (ignored).
            ref_cap: The reference caption to return as-is.

        Returns:
            The input ref_cap string, unmodified.
        """
        return ref_cap
