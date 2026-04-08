"""TypeAssigner: round-robin hallucination type cycling."""

from relcheck_v3.hallucination_generation.models import AnnotatedRecord, CaptionRecord, HallucinationType


class TypeAssigner:
    """Assigns hallucination types to records via i % 4 cycling."""

    CYCLE = [
        HallucinationType.OBJECT_EXISTENCE,
        HallucinationType.ATTRIBUTE,
        HallucinationType.INTERACTION,
        HallucinationType.COUNT,
    ]

    def assign(self, records: list[CaptionRecord]) -> list[AnnotatedRecord]:
        """Attach hallucination type via i % 4 cycling.

        Args:
            records: List of caption records to annotate.

        Returns:
            List of AnnotatedRecord with hallucination_type and index set.
        """
        return [
            AnnotatedRecord(
                image_id=record.image_id,
                caption=record.caption,
                image_path=record.image_path,
                hallucination_type=self.CYCLE[i % 4],
                index=i,
            )
            for i, record in enumerate(records)
        ]

    def assign_single(self, index: int) -> HallucinationType:
        """Return the hallucination type for a given zero-based index.

        Args:
            index: Zero-based position in the dataset.

        Returns:
            The HallucinationType for that index.
        """
        return self.CYCLE[index % 4]
