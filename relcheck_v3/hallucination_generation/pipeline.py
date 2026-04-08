"""Pipeline: synchronous end-to-end orchestration."""

import logging
import time

from tqdm import tqdm

from relcheck_v3.hallucination_generation.api_client import APIClient
from relcheck_v3.hallucination_generation.config import PipelineConfig
from relcheck_v3.hallucination_generation.data_loader import DataLoader
from relcheck_v3.hallucination_generation.edit_distance import EditDistanceValidator
from relcheck_v3.hallucination_generation.models import (
    AnnotatedRecord,
    ParsedRecord,
    RecordStatus,
    ResultRecord,
)
from relcheck_v3.hallucination_generation.response_parser import ResponseParser
from relcheck_v3.hallucination_generation.result_store import ResultStore
from relcheck_v3.hallucination_generation.type_assigner import TypeAssigner

logger = logging.getLogger(__name__)


class Pipeline:
    """Synchronous end-to-end hallucination generation pipeline."""

    def __init__(self, config: PipelineConfig) -> None:
        """Initialize all components from config."""
        self.config = config
        self.data_loader = DataLoader()
        self.type_assigner = TypeAssigner()
        self.api_client = APIClient(
            api_key=config.openai_api_key or "",
            model="gpt-4o-mini",
        )
        self.response_parser = ResponseParser()
        self.edit_distance_validator = EditDistanceValidator()
        self.result_store = ResultStore(output_dir=config.output_dir)

    def run(self) -> None:
        """Synchronous end-to-end pipeline execution."""
        # 1. Validate config
        if not self.config.dry_run and not self.config.openai_api_key:
            raise ValueError(
                "OpenAI API key is required when dry_run is not enabled. "
                "Set openai_api_key in PipelineConfig or enable dry_run mode."
            )

        start_time = time.time()

        # 2. Load data
        records = self.data_loader.load(
            dataset_name=self.config.dataset_name,
            annotation_path=self.config.annotation_path,
            image_dir=self.config.image_dir,
        )
        logger.info("Loaded %d records from %s", len(records), self.config.dataset_name)

        # 3. Assign types
        annotated = self.type_assigner.assign(records)
        logger.info("Assigned hallucination types to %d records", len(annotated))

        # 4. Apply max_samples limit
        if self.config.max_samples is not None:
            annotated = annotated[: self.config.max_samples]
            logger.info("Limited to %d samples (max_samples=%d)", len(annotated), self.config.max_samples)

        # 5. Load checkpoint and filter already-processed
        processed_keys = self.result_store.load_checkpoint()
        if processed_keys:
            logger.info("Checkpoint: %d records already processed", len(processed_keys))

        remaining = [
            rec for rec in annotated
            if f"{rec.image_id}::{rec.caption}" not in processed_keys
        ]
        logger.info("Processing %d remaining records", len(remaining))

        # 6. Iterate with tqdm, process each sample
        accepted = 0
        rejected = 0
        parse_failures = 0
        api_errors = 0

        for record in tqdm(remaining, desc="Processing"):
            result = self._process_single(record)
            self.result_store.append(result)

            if result.status == RecordStatus.ACCEPTED:
                accepted += 1
            elif result.status == RecordStatus.REJECTED:
                rejected += 1
            elif result.status == RecordStatus.PARSE_FAILURE:
                parse_failures += 1
            elif result.status == RecordStatus.API_ERROR:
                api_errors += 1

        # Log counts
        logger.info(
            "Processing complete: accepted=%d, rejected=%d, parse_failures=%d, api_errors=%d",
            accepted, rejected, parse_failures, api_errors,
        )

        # 7. Export CSV and write summary
        duration = time.time() - start_time
        self.result_store.export_csv()
        self.result_store.write_summary(duration_seconds=duration)
        logger.info("Results exported. Duration: %.1f seconds", duration)

    def _process_single(self, record: AnnotatedRecord) -> ResultRecord:
        """Process one sample: API call -> parse -> validate -> result."""
        # Dry-run mode: skip API call, use placeholder
        if self.config.dry_run:
            parsed = ParsedRecord(
                image_id=record.image_id,
                gt_cap=record.caption,
                ref_cap="[DRY_RUN_PLACEHOLDER]",
                hallucination_type=record.hallucination_type.value,
                reason="dry_run",
                parse_success=True,
            )
        else:
            # Call API
            api_response = self.api_client.generate_hallucination(record)

            if not api_response.success:
                logger.warning(
                    "API error for image_id=%s: %s",
                    record.image_id,
                    api_response.error_message,
                )
                return ResultRecord(
                    image_id=record.image_id,
                    gt_cap=record.caption,
                    ref_cap="",
                    hallucination_type=record.hallucination_type.value,
                    reason=api_response.error_message or "Unknown API error",
                    edit_distance=0,
                    status=RecordStatus.API_ERROR,
                )

            # Parse response
            parsed = self.response_parser.parse(api_response.raw_text, record)

        # Handle parse failure
        if not parsed.parse_success:
            return ResultRecord(
                image_id=parsed.image_id,
                gt_cap=parsed.gt_cap,
                ref_cap=parsed.ref_cap,
                hallucination_type=parsed.hallucination_type,
                reason=parsed.reason,
                edit_distance=0,
                status=RecordStatus.PARSE_FAILURE,
            )

        # Validate edit distance
        validation = self.edit_distance_validator.validate(parsed.gt_cap, parsed.ref_cap)

        if validation.accepted:
            status = RecordStatus.ACCEPTED
        else:
            status = RecordStatus.REJECTED

        return ResultRecord(
            image_id=parsed.image_id,
            gt_cap=parsed.gt_cap,
            ref_cap=parsed.ref_cap,
            hallucination_type=parsed.hallucination_type,
            reason=parsed.reason,
            edit_distance=validation.edit_distance,
            status=status,
        )
