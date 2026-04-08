"""Tests for TypeAssigner."""

from relcheck_v3.hallucination_generation.models import CaptionRecord, HallucinationType
from relcheck_v3.hallucination_generation.type_assigner import TypeAssigner


def _make_records(n: int) -> list[CaptionRecord]:
    """Create n dummy CaptionRecords for testing."""
    return [
        CaptionRecord(image_id=str(i), caption=f"caption {i}", image_path=f"/img/{i}.jpg")
        for i in range(n)
    ]


class TestTypeAssigner:
    def test_cycle_order(self):
        """CYCLE has the four types in the correct order."""
        assert TypeAssigner.CYCLE == [
            HallucinationType.OBJECT_EXISTENCE,
            HallucinationType.ATTRIBUTE,
            HallucinationType.INTERACTION,
            HallucinationType.COUNT,
        ]

    def test_assign_single_cycling(self):
        ta = TypeAssigner()
        assert ta.assign_single(0) == HallucinationType.OBJECT_EXISTENCE
        assert ta.assign_single(1) == HallucinationType.ATTRIBUTE
        assert ta.assign_single(2) == HallucinationType.INTERACTION
        assert ta.assign_single(3) == HallucinationType.COUNT
        assert ta.assign_single(4) == HallucinationType.OBJECT_EXISTENCE
        assert ta.assign_single(7) == HallucinationType.COUNT

    def test_assign_empty_list(self):
        ta = TypeAssigner()
        result = ta.assign([])
        assert result == []

    def test_assign_returns_annotated_records(self):
        ta = TypeAssigner()
        records = _make_records(5)
        annotated = ta.assign(records)

        assert len(annotated) == 5
        for i, rec in enumerate(annotated):
            assert rec.image_id == str(i)
            assert rec.caption == f"caption {i}"
            assert rec.image_path == f"/img/{i}.jpg"
            assert rec.index == i
            assert rec.hallucination_type == TypeAssigner.CYCLE[i % 4]

    def test_assign_preserves_record_data(self):
        ta = TypeAssigner()
        records = [
            CaptionRecord(image_id="abc", caption="a dog on a bench", image_path="/x.jpg"),
        ]
        annotated = ta.assign(records)
        assert annotated[0].image_id == "abc"
        assert annotated[0].caption == "a dog on a bench"
        assert annotated[0].image_path == "/x.jpg"

    def test_assign_type_distribution_divisible_by_4(self):
        ta = TypeAssigner()
        annotated = ta.assign(_make_records(8))
        counts = {}
        for rec in annotated:
            counts[rec.hallucination_type] = counts.get(rec.hallucination_type, 0) + 1
        # 8 records / 4 types = exactly 2 each
        assert all(c == 2 for c in counts.values())

    def test_assign_type_distribution_not_divisible_by_4(self):
        ta = TypeAssigner()
        annotated = ta.assign(_make_records(5))
        counts = {}
        for rec in annotated:
            counts[rec.hallucination_type] = counts.get(rec.hallucination_type, 0) + 1
        # max - min should be at most 1
        assert max(counts.values()) - min(counts.values()) <= 1
