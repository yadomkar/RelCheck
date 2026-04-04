"""
RelCheck v2 — Training-free relational hallucination detection and correction.

Public API:
    api.init_client          — initialize Together.ai client
    kb.build_visual_kb       — 3-layer Visual Knowledge Base construction
    verification.verify_triple — type-aware relation verification
    correction.enrich_caption_v3 — full correction/enrichment pipeline
    captioning.caption_image — unified captioning router
    evaluation.rpope_judge   — R-POPE LLM-judge evaluation
    injection.question_to_statement — synthetic hallucination injection
"""

__version__ = "2.0.0"
