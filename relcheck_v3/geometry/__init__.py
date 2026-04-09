"""GEOM layer — deterministic spatial relation verification from bounding boxes."""

from relcheck_v3.geometry.geometry import (
    compute_spatial_facts,
    spatial_verdict,
    extract_spatial_triples,
    parse_spatial_facts,
    check_spatial_contradictions,
)

__all__ = [
    "compute_spatial_facts",
    "spatial_verdict",
    "extract_spatial_triples",
    "parse_spatial_facts",
    "check_spatial_contradictions",
]
