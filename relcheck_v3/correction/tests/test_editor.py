"""Unit tests for RelCheckCaptionEditor (full Stage 1-5 orchestration).

These tests mock GPU-heavy dependencies (ClaimGenerationPipeline, build_kb)
to run without torch/groundingdino.  The editor uses lazy imports so that
it can be imported in CPU-only test environments.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from relcheck_v3.claim_generation.models import (
    CountClaim,
    ObjectAnswer,
    OverallClaim,
    SampleResult,
    SpecificClaim,
    StageTimings,
    VisualKnowledgeBase,
)
from relcheck_v3.correction.config import CorrectionConfig
from relcheck_v3.correction.corrector import HallucinationResult


# ── Helpers ───────────────────────────────────────────────────────────


def _make_sample_result(
    ref_cap: str = "A cat on a mat.",
    success: bool = True,
) -> SampleResult:
    """Build a minimal SampleResult for testing."""
    vkb = VisualKnowledgeBase(
        count_claims=[
            CountClaim(
                object_name="cat",
                count=1,
                claim_text="There is 1 cat.",
                bboxes=[[0.1, 0.2, 0.5, 0.6]],
            )
        ],
        specific_claims=[
            SpecificClaim(object_name="cat", claim_text="The cat is orange.")
        ],
        overall_claims=[
            OverallClaim(claim_text="A cat is sitting on a mat.")
        ],
    )
    return SampleResult(
        image_id="test",
        ref_cap=ref_cap,
        key_concepts=["cat", "mat"],
        object_questions=["Is there a cat?"],
        attribute_questions=[],
        object_answers={
            "cat": ObjectAnswer(
                object_name="cat", count=1, bboxes=[[0.1, 0.2, 0.5, 0.6]]
            ),
        },
        attribute_answers=[],
        visual_knowledge_base=vkb,
        vkb_text=vkb.format(),
        timings=StageTimings(),
        success=success,
    )


def _make_mock_kb():
    """Build a mock KnowledgeBase."""
    kb = MagicMock()
    kb.format.return_value = "=== CLAIM ===\nThere is 1 cat."
    kb.spatial_facts = ["cat is on mat"]
    kb.scene_graph = []
    return kb


# ── Tests ─────────────────────────────────────────────────────────────


class TestRelCheckCaptionEditor:
    """Test the editor orchestration with mocked pipeline + corrector."""

    def _build_editor(
        self, mock_pipeline_cls, mock_corrector_cls, mock_config_cls=None
    ):
        """Build an editor with mocked heavy deps injected via patches."""
        from relcheck_v3.correction.editor import RelCheckCaptionEditor

        editor = RelCheckCaptionEditor.__new__(RelCheckCaptionEditor)
        editor._claim_gen_config = MagicMock()
        editor._correction_config = CorrectionConfig(openai_api_key="sk-test")
        editor._pipeline = mock_pipeline_cls
        editor._corrector = mock_corrector_cls
        return editor

    def test_successful_correction(self):
        """Editor wires claim gen → KB → corrector and returns corrected caption."""
        mock_pipeline = MagicMock()
        mock_pipeline.process_single.return_value = _make_sample_result()

        mock_corrector = MagicMock()
        mock_corrector.run.return_value = (
            "A cat on a blue mat.",
            HallucinationResult(
                hallucinated_span="mat",
                reason="CLAIM says blue mat",
                correction_hint="blue mat",
                confidence="high",
                raw_json={},
            ),
        )

        editor = self._build_editor(mock_pipeline, mock_corrector)

        mock_kb = _make_mock_kb()
        with patch("relcheck_v3.kb.builder.build_kb", return_value=mock_kb):
            result = editor.edit_caption("/tmp/image.jpg", "A cat on a mat.")

        assert result == "A cat on a blue mat."
        mock_pipeline.process_single.assert_called_once()
        mock_corrector.run.assert_called_once()

    def test_passthrough_on_claim_gen_failure(self):
        """Returns original caption when claim generation fails."""
        mock_pipeline = MagicMock()
        mock_pipeline.process_single.return_value = _make_sample_result(
            success=False
        )
        mock_corrector = MagicMock()

        editor = self._build_editor(mock_pipeline, mock_corrector)
        result = editor.edit_caption("/tmp/image.jpg", "A cat on a mat.")

        assert result == "A cat on a mat."
        mock_corrector.run.assert_not_called()

    def test_passthrough_on_no_hallucination(self):
        """Returns original caption when no hallucination is identified."""
        mock_pipeline = MagicMock()
        mock_pipeline.process_single.return_value = _make_sample_result()

        mock_corrector = MagicMock()
        mock_corrector.run.return_value = ("A cat on a mat.", None)

        editor = self._build_editor(mock_pipeline, mock_corrector)

        mock_kb = _make_mock_kb()
        with patch("relcheck_v3.kb.builder.build_kb", return_value=mock_kb):
            result = editor.edit_caption("/tmp/image.jpg", "A cat on a mat.")

        assert result == "A cat on a mat."

    def test_passthrough_on_kb_build_failure(self):
        """Returns original caption when KB build raises."""
        mock_pipeline = MagicMock()
        mock_pipeline.process_single.return_value = _make_sample_result()

        mock_corrector = MagicMock()

        editor = self._build_editor(mock_pipeline, mock_corrector)

        with patch(
            "relcheck_v3.kb.builder.build_kb",
            side_effect=RuntimeError("GPU error"),
        ):
            result = editor.edit_caption("/tmp/image.jpg", "A cat on a mat.")

        assert result == "A cat on a mat."
        mock_corrector.run.assert_not_called()


class TestCaptionEditorProtocol:
    """Verify RelCheckCaptionEditor satisfies the Caption_Editor protocol."""

    def test_has_edit_caption_method(self):
        from relcheck_v3.correction.editor import RelCheckCaptionEditor

        assert hasattr(RelCheckCaptionEditor, "edit_caption")

    def test_signature_matches_protocol(self):
        from relcheck_v3.correction.editor import RelCheckCaptionEditor

        sig = RelCheckCaptionEditor.edit_caption.__code__.co_varnames
        assert "image" in sig
        assert "ref_cap" in sig
