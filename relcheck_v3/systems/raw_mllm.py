"""RawMLLM — passthrough correction system (lower bound).

Returns the MLLM output unchanged, serving as the no-correction baseline
in the evaluation harness.
"""

from __future__ import annotations


class RawMLLM:
    """Passthrough correction system that returns MLLM output unchanged.

    This serves as the lower-bound baseline: no correction is applied,
    so any improvement by other systems is measured against this.

    Attributes:
        system_id: Identifier for this system variant (``"raw"``).
    """

    system_id: str = "raw"

    def correct(self, image_path: str, mllm_output: str) -> str:
        """Return *mllm_output* unchanged.

        Args:
            image_path: Absolute path to the source image file (unused).
            mllm_output: Raw text produced by the MLLM.

        Returns:
            The original *mllm_output* string, unmodified.
        """
        return mllm_output
