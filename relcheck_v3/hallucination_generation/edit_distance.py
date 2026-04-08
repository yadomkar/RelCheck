"""EditDistanceValidator: Levenshtein distance check (5 < d < 50)."""

from Levenshtein import distance as levenshtein_distance

from relcheck_v3.hallucination_generation.models import ValidationResult


class EditDistanceValidator:
    """Validates that edit distance between GT-Cap and Ref-Cap is within bounds.

    Default bounds: distance must be strictly greater than 5 and strictly less than 50.
    """

    def __init__(self, min_distance: int = 6, max_distance: int = 49):
        """Initialize with inclusive bounds.

        Args:
            min_distance: Minimum accepted distance (inclusive). Default 6 means distance > 5.
            max_distance: Maximum accepted distance (inclusive). Default 49 means distance < 50.
        """
        self.min_distance = min_distance
        self.max_distance = max_distance

    def validate(self, gt_cap: str, ref_cap: str) -> ValidationResult:
        """Compute Levenshtein distance and return accept/reject with reason.

        Args:
            gt_cap: Ground-truth caption.
            ref_cap: Reference (hallucinated) caption.

        Returns:
            ValidationResult with edit_distance, accepted flag, and rejection_reason if rejected.
        """
        dist = levenshtein_distance(gt_cap, ref_cap)

        if dist < self.min_distance:
            return ValidationResult(
                edit_distance=dist,
                accepted=False,
                rejection_reason="edit_distance_too_small",
            )

        if dist > self.max_distance:
            return ValidationResult(
                edit_distance=dist,
                accepted=False,
                rejection_reason="edit_distance_too_large",
            )

        return ValidationResult(
            edit_distance=dist,
            accepted=True,
            rejection_reason=None,
        )
