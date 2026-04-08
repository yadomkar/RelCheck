"""ResponseParser: extract structured fields from GPT response."""

import json
import re
from typing import Optional

from relcheck_v3.hallucination_generation.models import AnnotatedRecord, ParsedRecord


class ResponseParser:
    """Extract structured fields from GPT-4o-mini responses.

    Tries JSON parsing first, falls back to regex key-value extraction.
    """

    # Mapping from various JSON key forms to our canonical field names
    _JSON_KEY_MAP: dict[str, str] = {
        "image id": "image_id",
        "image_id": "image_id",
        "gt-cap": "gt_cap",
        "gt_cap": "gt_cap",
        "ref-cap": "ref_cap",
        "ref_cap": "ref_cap",
        "type": "hallucination_type",
        "reason": "reason",
    }

    # Regex patterns for key-value extraction (handles quoted and unquoted values)
    # Each pattern captures the value after the key, stopping at the next key or end of string.
    _KV_PATTERNS: dict[str, re.Pattern[str]] = {
        "image_id": re.compile(
            r"['\"]?image[_ ]id['\"]?\s*[:=]\s*['\"]?(.+?)(?:['\"]?\s*(?:,\s*)?['\"]?(?:GT-Cap|gt.cap)\b|['\"]?\s*$)",
            re.IGNORECASE,
        ),
        "gt_cap": re.compile(
            r"['\"]?GT-Cap['\"]?\s*[:=]\s*['\"]?(.+?)(?:['\"]?\s*(?:,\s*)?['\"]?(?:Ref-Cap|ref.cap)\b|['\"]?\s*$)",
            re.IGNORECASE | re.DOTALL,
        ),
        "ref_cap": re.compile(
            r"['\"]?Ref-Cap['\"]?\s*[:=]\s*['\"]?(.+?)(?:['\"]?\s*(?:,\s*)?['\"]?(?:Type|type)\b|['\"]?\s*$)",
            re.IGNORECASE | re.DOTALL,
        ),
        "hallucination_type": re.compile(
            r"['\"]?Type['\"]?\s*[:=]\s*['\"]?(.+?)(?:['\"]?\s*(?:,\s*)?['\"]?(?:Reason|reason)\b|['\"]?\s*$)",
            re.IGNORECASE | re.DOTALL,
        ),
        "reason": re.compile(
            r"['\"]?Reason['\"]?\s*[:=]\s*['\"]?(.+?)(?:['\"]?\s*$)",
            re.IGNORECASE | re.DOTALL,
        ),
    }

    _REQUIRED_FIELDS = {"image_id", "gt_cap", "ref_cap", "hallucination_type", "reason"}

    def parse(self, raw_text: str, original_record: AnnotatedRecord) -> ParsedRecord:
        """Extract image_id, GT-Cap, Ref-Cap, Type, Reason from GPT response.

        Tries JSON parsing first, falls back to key-value regex extraction.
        Returns ParsedRecord with parse_success=True on success, or
        parse_success=False with raw_text stored for debugging on failure.
        """
        # Try JSON parsing first
        fields = self._try_json_parse(raw_text)

        # Fall back to regex key-value extraction
        if fields is None:
            fields = self._try_regex_parse(raw_text)

        # If we got fields, check completeness
        if fields is not None:
            # Strip whitespace from all extracted values
            fields = {k: v.strip() for k, v in fields.items()}
            # Check all required fields are present and non-empty
            if all(fields.get(f) for f in self._REQUIRED_FIELDS):
                return ParsedRecord(
                    image_id=str(fields["image_id"]),
                    gt_cap=fields["gt_cap"],
                    ref_cap=fields["ref_cap"],
                    hallucination_type=fields["hallucination_type"],
                    reason=fields["reason"],
                    parse_success=True,
                )

        # Parse failure — use original_record fields as fallback
        return ParsedRecord(
            image_id=original_record.image_id,
            gt_cap=original_record.caption,
            ref_cap="",
            hallucination_type=original_record.hallucination_type.value,
            reason="",
            parse_success=False,
            raw_text=raw_text,
        )

    def serialize(self, record: ParsedRecord) -> str:
        """Serialize a ParsedRecord back to key-value text format."""
        return (
            f"'image id': {record.image_id}, "
            f"'GT-Cap': {record.gt_cap}, "
            f"'Ref-Cap': {record.ref_cap}, "
            f"'Type': {record.hallucination_type}, "
            f"'Reason': {record.reason}"
        )

    def _try_json_parse(self, raw_text: str) -> Optional[dict[str, str]]:
        """Attempt to parse raw_text as JSON and extract fields."""
        try:
            # Try to find JSON object in the text
            text = raw_text.strip()

            # If the text itself isn't valid JSON, try to find a JSON block
            data = None
            if text.startswith("{"):
                data = json.loads(text)
            else:
                # Look for JSON object embedded in text
                match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
                if match:
                    data = json.loads(match.group())

            if data is None or not isinstance(data, dict):
                return None

            # Normalize keys to canonical field names
            fields: dict[str, str] = {}
            for key, value in data.items():
                canonical = self._JSON_KEY_MAP.get(key.lower().strip())
                if canonical:
                    fields[canonical] = str(value)

            return fields if fields else None

        except (json.JSONDecodeError, ValueError, TypeError):
            return None

    def _try_regex_parse(self, raw_text: str) -> Optional[dict[str, str]]:
        """Attempt to extract fields using regex key-value patterns."""
        fields: dict[str, str] = {}

        for field_name, pattern in self._KV_PATTERNS.items():
            match = pattern.search(raw_text)
            if match:
                value = match.group(1).strip().rstrip("'\"")
                if value:
                    fields[field_name] = value

        return fields if fields else None
