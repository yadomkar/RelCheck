"""Property-based and unit tests for GroundingDINODetector.

Property 9: Bounding box coordinates normalized to [0, 1]
Unit tests: CUDA-not-available raises RuntimeError, default thresholds used

Validates: Requirements 3.4, 8.3, 8.5
"""

from unittest.mock import MagicMock, patch, call

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from relcheck_v3.claim_generation.models import Detection


# ---------------------------------------------------------------------------
# Property 9: Bounding box coordinates normalized to [0, 1]
# ---------------------------------------------------------------------------
# Feature: woodpecker-correction, Property 9: Bounding box coordinates normalized to [0, 1]


@given(
    x_min=st.floats(min_value=0.0, max_value=1.0),
    y_min=st.floats(min_value=0.0, max_value=1.0),
    x_max=st.floats(min_value=0.0, max_value=1.0),
    y_max=st.floats(min_value=0.0, max_value=1.0),
    confidence=st.floats(min_value=0.0, max_value=1.0),
)
@settings(max_examples=100)
def test_bbox_coordinates_normalized_to_unit_range(
    x_min: float, y_min: float, x_max: float, y_max: float, confidence: float
) -> None:
    """For any Detection object with bbox values in [0, 1], all four coordinates
    remain in [0.0, 1.0].

    **Validates: Requirements 8.5**
    """
    detection = Detection(
        bbox=[x_min, y_min, x_max, y_max],
        confidence=confidence,
    )
    for coord in detection.bbox:
        assert 0.0 <= coord <= 1.0, (
            f"Bounding box coordinate {coord} is outside [0.0, 1.0]"
        )


# ---------------------------------------------------------------------------
# Unit Tests (Task 5.3)
# ---------------------------------------------------------------------------


class TestCudaNotAvailable:
    """CUDA-not-available raises RuntimeError with GPU message.

    **Validates: Requirements 8.3**
    """

    def test_cuda_not_available_raises_runtime_error(self) -> None:
        with patch("relcheck_v3.claim_generation.grounding_dino.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            from relcheck_v3.claim_generation.grounding_dino import GroundingDINODetector

            with pytest.raises(RuntimeError, match="GPU"):
                GroundingDINODetector()


class TestDefaultThresholds:
    """Default thresholds (box=0.35, text=0.25) are used when not overridden.

    **Validates: Requirements 3.4**
    """

    def test_default_thresholds_used_in_detect(self) -> None:
        with patch("relcheck_v3.claim_generation.grounding_dino.torch") as mock_torch, \
             patch("relcheck_v3.claim_generation.grounding_dino.AutoProcessor") as mock_proc_cls, \
             patch("relcheck_v3.claim_generation.grounding_dino.AutoModelForZeroShotObjectDetection") as mock_model_cls:

            # --- Setup CUDA as available ---
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.memory_allocated.return_value = 500 * 1024 * 1024  # 500 MB
            mock_torch.no_grad.return_value.__enter__ = MagicMock()
            mock_torch.no_grad.return_value.__exit__ = MagicMock()

            # --- Setup mock tensor for target_sizes ---
            mock_tensor = MagicMock()
            mock_torch.tensor.return_value = mock_tensor

            # --- Setup mock processor ---
            mock_processor = MagicMock()
            mock_inputs = MagicMock()
            mock_inputs.__getitem__ = MagicMock(return_value=MagicMock())
            mock_processor.return_value = mock_inputs
            mock_processor.post_process_grounded_object_detection.return_value = []
            mock_proc_cls.from_pretrained.return_value = mock_processor

            # --- Setup mock model ---
            mock_model = MagicMock()
            mock_model.to.return_value = mock_model
            mock_model_cls.from_pretrained.return_value = mock_model

            from relcheck_v3.claim_generation.grounding_dino import GroundingDINODetector
            from PIL import Image

            detector = GroundingDINODetector()

            # Create a small test image
            test_image = Image.new("RGB", (100, 100), color="red")

            # Call detect without specifying thresholds (should use defaults)
            detector.detect(test_image, "dog")

            # Verify post_process was called with default thresholds
            pp_call = mock_processor.post_process_grounded_object_detection
            pp_call.assert_called_once()
            call_kwargs = pp_call.call_args
            # Check keyword arguments for threshold and text_threshold
            assert call_kwargs.kwargs.get("threshold") == 0.35, (
                f"Expected box_threshold=0.35, got {call_kwargs.kwargs.get('threshold')}"
            )
            assert call_kwargs.kwargs.get("text_threshold") == 0.25, (
                f"Expected text_threshold=0.25, got {call_kwargs.kwargs.get('text_threshold')}"
            )
