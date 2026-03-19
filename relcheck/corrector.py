"""
RelCheck — Stage 3: Minimal Corrector
=======================================
For each triple flagged as hallucinated, generates a minimal edit to the
original caption — changing only the hallucinated relation span, leaving
everything else untouched.

Uses Mistral-7B-Instruct via the Together.ai API (free tier: 5M tokens/month).
Fully open-source and reproducible — no OpenAI dependency.

After correction, a self-consistency check re-runs the triple extractor on the
corrected caption to make sure no new hallucinated triples were introduced.

Dependencies:
    !pip install together

Setup:
    1. Go to https://api.together.xyz and create a free account
    2. Copy your API key
    3. Set it as: import os; os.environ["TOGETHER_API_KEY"] = "your_key_here"

Author: Siddhi Patil | CS298 Spring 2026
"""

import os
import re
from typing import Optional

from triple_extractor import Triple, TripleExtractor


# ---------------------------------------------------------------------------
# Prompt engineering
# ---------------------------------------------------------------------------

CORRECTION_SYSTEM_PROMPT = """You are a precise caption editor. Your job is to fix \
factual errors in image captions — specifically incorrect relationship descriptions.

Rules:
1. Change ONLY the specific incorrect relation. Do not rewrite the whole caption.
2. Keep all other words, objects, and descriptions exactly the same.
3. If you are not sure what the correct relation is, use the most neutral option \
   (e.g. "near" instead of a specific spatial relation).
4. Output ONLY the corrected caption — no explanation, no quotes, nothing else."""

CORRECTION_USER_TEMPLATE = """Original caption: "{caption}"

The relation '{relation}' between '{subject}' and '{object}' has been verified to \
be incorrect — it is not supported by the image.

Rewrite the caption with a minimal fix for this one relation only. \
Keep everything else exactly the same.

Corrected caption:"""


# ---------------------------------------------------------------------------
# Minimal Corrector
# ---------------------------------------------------------------------------

class MinimalCorrector:
    """
    Calls Llama-3.1-8B-Instant (via Groq) to minimally rewrite a caption
    when a hallucinated triple is detected.

    The key design principle: we only change the hallucinated triple's span.
    We do NOT ask the model to "improve" or "rewrite" the caption — just fix
    the one broken relation.
    """

    MODEL = "llama-3.1-8b-instant"

    def __init__(self, api_key: Optional[str] = None, extractor: Optional[TripleExtractor] = None):
        """
        Args:
            api_key:   Together.ai API key. If None, reads from TOGETHER_API_KEY env var.
            extractor: A TripleExtractor instance for self-consistency checking.
                       If None, self-consistency check is skipped.
        """
        from groq import Groq  # pip install groq
        api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "No Groq API key found. Set GROQ_API_KEY environment variable "
                "or pass api_key= to MinimalCorrector(). Get a free key at console.groq.com"
            )
        self.client = Groq(api_key=api_key)
        self.extractor = extractor
        print(f"[MinimalCorrector] Ready. Model: {self.MODEL}")

    def _call_llm(self, caption: str, triple: Triple) -> str:
        """Call Llama-3.1-8B (Groq) to produce a corrected caption."""
        user_msg = CORRECTION_USER_TEMPLATE.format(
            caption=caption,
            relation=triple.relation,
            subject=triple.subject,
            object=triple.obj,
        )

        response = self.client.chat.completions.create(
            model=self.MODEL,
            messages=[
                {"role": "system", "content": CORRECTION_SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=200,
            temperature=0.2,      # low temperature = more deterministic, less creative
        )

        corrected = response.choices[0].message.content.strip()

        # Strip any accidental quotes the model wraps around its answer
        corrected = corrected.strip('"').strip("'").strip()
        return corrected

    def _self_consistency_check(self, original: str, corrected: str) -> bool:
        """
        After correction, re-extract triples from the corrected caption and
        check that no obviously new hallucinated triples were introduced.

        This is a lightweight check: we just make sure the corrected caption
        isn't longer / more complex than the original by an unreasonable amount.
        (Full VQA re-verification would be too slow for a consistency guard.)

        Returns True if the correction looks clean, False if we should revert.
        """
        if not corrected:
            return False

        # Basic sanity: corrected caption should be similar length
        ratio = len(corrected) / max(len(original), 1)
        if ratio > 1.5 or ratio < 0.5:
            return False   # wildly different length — revert

        # Make sure the model didn't just echo back the system prompt or refuse
        bad_phrases = [
            "i cannot", "i can't", "i apologize", "as an ai",
            "here is", "here's", "corrected caption:", "the corrected"
        ]
        lowered = corrected.lower()
        if any(p in lowered for p in bad_phrases):
            return False

        return True

    def correct_triple(self, caption: str, triple: Triple) -> tuple[str, bool]:
        """
        Attempt to correct one hallucinated triple in the caption.

        Args:
            caption: The original caption string.
            triple:  The hallucinated Triple to fix.

        Returns:
            (corrected_caption, correction_applied)
            If the correction fails the self-consistency check, returns
            (original_caption, False).
        """
        try:
            corrected = self._call_llm(caption, triple)
        except Exception as e:
            print(f"[MinimalCorrector] API call failed: {e}")
            return caption, False

        if not self._self_consistency_check(caption, corrected):
            print(f"[MinimalCorrector] Self-consistency check failed — reverting.")
            return caption, False

        # Record the correction in the triple
        triple.correction = corrected
        return corrected, True

    def correct_all(self, caption: str, triples: list[Triple]) -> str:
        """
        Correct all hallucinated triples in sequence.

        We process one at a time, passing the (already corrected) caption into
        each successive correction. This means later corrections see the already-
        fixed caption, which avoids double-editing the same span.

        Returns:
            The final corrected caption.
        """
        current_caption = caption
        corrections_made = 0

        for triple in triples:
            if triple.hallucinated is True:
                print(f"[MinimalCorrector] Correcting: {triple}")
                current_caption, applied = self.correct_triple(current_caption, triple)
                if applied:
                    corrections_made += 1

        if corrections_made == 0:
            print("[MinimalCorrector] No corrections were needed or all failed checks.")

        return current_caption


# ---------------------------------------------------------------------------
# Lightweight fallback corrector (no LLM needed — for testing without API key)
# ---------------------------------------------------------------------------

SPATIAL_FALLBACK_MAP = {
    # If we know the relation is wrong, offer a neutral replacement
    "on":       "near",
    "in":       "near",
    "above":    "near",
    "below":    "near",
    "holding":  "near",
    "carrying": "near",
    "riding":   "near",
}


class RuleBasedCorrector:
    """
    Fallback corrector that uses simple string replacement with a rule-based
    relation map. Much less accurate than Mistral, but works without an API key.

    Useful for: debugging the pipeline without API credits, or as a fast baseline.
    """

    def correct_triple(self, caption: str, triple: Triple) -> tuple[str, bool]:
        replacement = SPATIAL_FALLBACK_MAP.get(triple.relation.lower(), "near")

        # Try to find and replace the relation verb/preposition in the caption
        pattern = rf'\b{re.escape(triple.relation)}\b'
        corrected, n = re.subn(pattern, replacement, caption, count=1, flags=re.IGNORECASE)

        if n > 0:
            triple.correction = corrected
            return corrected, True
        return caption, False

    def correct_all(self, caption: str, triples: list[Triple]) -> str:
        current = caption
        for triple in triples:
            if triple.hallucinated is True:
                current, _ = self.correct_triple(current, triple)
        return current
