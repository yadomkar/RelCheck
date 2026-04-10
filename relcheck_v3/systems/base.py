"""Base protocol for correction systems.

All correction system variants implement the ``CorrectionSystem`` protocol,
which guarantees a uniform ``correct(image_path, mllm_output) -> str``
interface so the evaluation runner can treat them interchangeably.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class CorrectionSystem(Protocol):
    """Uniform interface for all correction system variants.

    Every system exposes a ``system_id`` string for identification and a
    ``correct`` method that takes an image path together with raw MLLM
    output and returns a (possibly corrected) string.

    This is a :pep:`544` structural protocol decorated with
    ``@runtime_checkable`` so that ``isinstance`` checks work at runtime.
    """

    system_id: str

    def correct(self, image_path: str, mllm_output: str) -> str:
        """Return a corrected version of *mllm_output* for the given image.

        Args:
            image_path: Absolute path to the source image file.
            mllm_output: Raw text produced by the MLLM for this image.

        Returns:
            The (possibly corrected) output string.
        """
        ...
