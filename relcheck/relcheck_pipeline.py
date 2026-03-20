"""
RelCheck — Full Pipeline
=========================
Wires together Stage 1 (Triple Extractor), Stage 2 (Relation Verifier),
and Stage 3 (Minimal Corrector) into one end-to-end system.

Usage (in Colab):
    from relcheck_pipeline import RelCheckPipeline
    pipeline = RelCheckPipeline()
    result = pipeline.run(image_path="my_image.jpg")
    print(result.corrected_caption)

Author: Siddhi Patil | CS298 Spring 2026
"""

import json
import os
import csv
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from PIL import Image

from triple_extractor import Triple, TripleExtractor, LLMTripleExtractor
from relation_verifier import RelationVerifier
from corrector import MinimalCorrector, RuleBasedCorrector


# ---------------------------------------------------------------------------
# Result data model
# ---------------------------------------------------------------------------

@dataclass
class RelCheckResult:
    """Everything RelCheck produces for one image."""
    image_path:          str
    original_caption:    str
    corrected_caption:   str
    triples:             list[Triple]
    n_triples:           int           = 0
    n_hallucinated:      int           = 0
    n_corrected:         int           = 0
    edit_rate:           float         = 0.0   # Levenshtein / len(original)
    any_hallucination:   bool          = False  # for R-CHAIR_s computation

    def __post_init__(self):
        self.n_triples      = len(self.triples)
        self.n_hallucinated = sum(1 for t in self.triples if t.hallucinated is True)
        self.n_corrected    = sum(1 for t in self.triples if t.correction is not None)
        self.any_hallucination = self.n_hallucinated > 0
        self.edit_rate      = _levenshtein_rate(self.original_caption, self.corrected_caption)

    def print_summary(self):
        print(f"\n{'='*60}")
        print(f"Image: {self.image_path}")
        print(f"{'='*60}")
        print(f"ORIGINAL:  {self.original_caption}")
        print(f"CORRECTED: {self.corrected_caption}")
        print(f"\nTriples found: {self.n_triples}")
        for t in self.triples:
            print(f"  {t}")
        print(f"\nHallucinated: {self.n_hallucinated}/{self.n_triples}")
        print(f"Corrected:    {self.n_corrected}")
        print(f"Edit rate:    {self.edit_rate:.3f}")
        print(f"{'='*60}")

    def as_dict(self) -> dict:
        return {
            "image_path":        self.image_path,
            "original_caption":  self.original_caption,
            "corrected_caption": self.corrected_caption,
            "n_triples":         self.n_triples,
            "n_hallucinated":    self.n_hallucinated,
            "n_corrected":       self.n_corrected,
            "edit_rate":         round(self.edit_rate, 4),
            "any_hallucination": self.any_hallucination,
            "triples":           [t.as_dict() for t in self.triples],
        }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class RelCheckPipeline:
    """
    End-to-end RelCheck pipeline.

    Args:
        together_api_key: Together.ai key for Mistral correction.
                          If None, falls back to RuleBasedCorrector.
        use_rule_corrector: Force rule-based corrector even if API key is set.
                            Useful for ablation (detect-only) experiments.
        detection_only:   If True, run Stages 1+2 only (no correction).
                          Used for the detect-only ablation.
        skip_spatial:     Disable SpatialVerifier (ablation).
        skip_vqa:         Disable VQAVerifier (ablation).
    """

    def __init__(
        self,
        together_api_key:    Optional[str]  = None,
        use_rule_corrector:  bool           = False,
        detection_only:      bool           = False,
        skip_spatial:        bool           = False,
        skip_vqa:            bool           = False,
        # Cross-model verification: inject LLaVA model + processor
        llava_model=None,
        llava_processor=None,
        # LLM-based triple extraction (Mistral via Together.ai)
        use_llm_extractor:   bool           = False,
        # Multi-question VQA voting (VisMin-inspired)
        num_paraphrases:     int            = 3,
    ):
        print("[RelCheckPipeline] Initializing...")
        api_key = together_api_key or os.environ.get("TOGETHER_API_KEY")

        # Stage 1: Triple Extractor
        spacy_extractor = TripleExtractor()
        if use_llm_extractor and api_key:
            print("[RelCheckPipeline] Using LLM-based triple extractor (Mistral).")
            self.extractor = LLMTripleExtractor(
                api_key=api_key,
                fallback_extractor=spacy_extractor,
            )
        else:
            self.extractor = spacy_extractor

        # Stage 2: Relation Verifier (with multi-question VQA voting)
        self.verifier = RelationVerifier(
            llava_model=llava_model,
            llava_processor=llava_processor,
            num_paraphrases=num_paraphrases,
        )
        self.verifier.skip_spatial = skip_spatial
        self.verifier.skip_vqa     = skip_vqa

        # Stage 3: Corrector
        self.detection_only = detection_only
        if detection_only:
            print("[RelCheckPipeline] Detection-only mode — correction disabled.")
            self.corrector = None
        elif use_rule_corrector:
            self.corrector = RuleBasedCorrector()
        else:
            if api_key:
                self.corrector = MinimalCorrector(api_key=api_key, extractor=self.extractor)
            else:
                print("[RelCheckPipeline] No API key found — using rule-based corrector.")
                self.corrector = RuleBasedCorrector()

        print("[RelCheckPipeline] Ready.\n")

    def run(
        self,
        image_path: str,
        caption: Optional[str] = None,
        blip2_processor=None,
        blip2_model=None,
    ) -> RelCheckResult:
        """
        Run the full RelCheck pipeline on one image.

        Args:
            image_path:      Path to the image file.
            caption:         Pre-computed caption. If None, BLIP-2 generates one.
            blip2_processor: Loaded BLIP-2 processor (needed if caption=None).
            blip2_model:     Loaded BLIP-2 model     (needed if caption=None).

        Returns:
            RelCheckResult with all triples, hallucination flags, and corrected caption.
        """
        import torch

        image = Image.open(image_path).convert("RGB")

        # ── Generate caption if not provided ──────────────────────────────────
        if caption is None:
            if blip2_processor is None or blip2_model is None:
                raise ValueError(
                    "Provide either caption= or blip2_processor + blip2_model."
                )
            device = next(blip2_model.parameters()).device
            prompt = "Describe this image in detail."
            inputs = blip2_processor(
                images=image, text=prompt, return_tensors="pt"
            ).to(device, torch.float16)
            with torch.no_grad():
                ids = blip2_model.generate(**inputs, max_new_tokens=80)
            caption = blip2_processor.decode(ids[0], skip_special_tokens=True).strip()
            print(f"[Pipeline] BLIP-2 caption: {caption}")

        # ── Stage 1: Extract triples ──────────────────────────────────────────
        triples = self.extractor.extract(caption)
        print(f"[Pipeline] Extracted {len(triples)} triple(s).")

        # ── Stage 2: Verify triples ───────────────────────────────────────────
        triples = self.verifier.verify_all(image, triples)

        hallucinated = [t for t in triples if t.hallucinated is True]
        print(f"[Pipeline] Hallucinated: {len(hallucinated)}/{len(triples)}")

        # ── Stage 3: Correct hallucinations ──────────────────────────────────
        if self.detection_only or not hallucinated or self.corrector is None:
            corrected_caption = caption
        else:
            corrected_caption = self.corrector.correct_all(caption, triples)

        return RelCheckResult(
            image_path=image_path,
            original_caption=caption,
            corrected_caption=corrected_caption,
            triples=triples,
        )

    def run_batch(
        self,
        image_paths: list[str],
        captions: Optional[list[str]] = None,
        blip2_processor=None,
        blip2_model=None,
        output_csv: Optional[str] = None,
    ) -> list[RelCheckResult]:
        """
        Run RelCheck on a list of images and optionally save results to CSV.

        Args:
            image_paths:     List of image file paths.
            captions:        Optional pre-computed captions (same length as image_paths).
            blip2_processor: For caption generation if captions=None.
            blip2_model:     For caption generation if captions=None.
            output_csv:      If set, write results to this CSV file path.

        Returns:
            List of RelCheckResult objects.
        """
        results = []

        for i, img_path in enumerate(image_paths):
            print(f"\n[Pipeline] Processing {i+1}/{len(image_paths)}: {img_path}")
            cap = captions[i] if captions else None
            try:
                result = self.run(
                    image_path=img_path,
                    caption=cap,
                    blip2_processor=blip2_processor,
                    blip2_model=blip2_model,
                )
                result.print_summary()
                results.append(result)
            except Exception as e:
                print(f"[Pipeline] ERROR on {img_path}: {e}")

        if output_csv:
            self._save_csv(results, output_csv)

        return results

    @staticmethod
    def _save_csv(results: list[RelCheckResult], path: str):
        """Save results to a flat CSV (one row per image)."""
        rows = [r.as_dict() for r in results]
        # Flatten triples list to a JSON string for CSV
        for row in rows:
            row["triples"] = json.dumps(row["triples"])

        if not rows:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[Pipeline] Results saved to: {path}")


# ---------------------------------------------------------------------------
# Levenshtein edit rate
# ---------------------------------------------------------------------------

def _levenshtein_rate(s1: str, s2: str) -> float:
    """Character-level Levenshtein distance normalized by len(s1)."""
    if not s1:
        return 0.0
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[j] = prev[j-1]
            else:
                dp[j] = 1 + min(prev[j], dp[j-1], prev[j-1])
    return dp[n] / m
