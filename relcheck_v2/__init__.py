"""
RelCheck v2 — Training-free relational hallucination detection and correction.

A modular, type-aware pipeline for detecting and correcting relational
hallucinations in multimodal LLM outputs.  Works as a black-box post-
processor on *any* captioning model (BLIP-2, LLaVA, Qwen, InstructBLIP).

Pipeline stages
---------------
1. **Captioning** — generate or load a caption for an image.
2. **KB construction** — build a Visual Knowledge Base (GroundingDINO
   detections + VLM description) for grounded verification.
3. **Verification** — type-aware routing: spatial relations use
   deterministic bbox geometry; action/attribute relations use
   crop-based VQA with contrastive forced-choice.
4. **Correction** — enrichment for short captions (< 30 words),
   surgical span editing for long captions (>= 30 words).
5. **Evaluation** — R-POPE LLM-judge, CLIPScore, R-CHAIR.

Public API
----------
api.init_client            — initialize Together.ai client
kb.build_visual_kb         — 3-layer Visual Knowledge Base construction
verification.verify_triple — type-aware relation verification
correction.enrich_caption_v3 — full correction / enrichment pipeline
captioning.caption_image   — unified captioning router
evaluation.rpope_judge     — R-POPE LLM-judge evaluation
injection.question_to_statement — synthetic hallucination injection
"""

__version__ = "2.0.0"

__all__ = [
    # Subpackages / modules consumers are most likely to import
    "api",
    "captioning",
    "config",
    "correction",
    "data",
    "detection",
    "entity",
    "evaluation",
    "injection",
    "kb",
    "models",
    "prompts",
    "spatial",
    "types",
    "verification",
]
