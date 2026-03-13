"""
RelCheck — Stage 1: Triple Extractor
=====================================
Parses a natural-language caption into structured (subject, relation, object)
triples, and classifies each relation as SPATIAL, ACTION, or ATTRIBUTE.

Dependencies (install in Colab):
    !pip install spacy
    !python -m spacy download en_core_web_sm

Author: Siddhi Patil | CS298 Spring 2026
"""

import re
from dataclasses import dataclass, field
from typing import Optional

import spacy


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

RELATION_TYPES = ("SPATIAL", "ACTION", "ATTRIBUTE", "OTHER")

# Relations we explicitly care about, grouped by type.
SPATIAL_KEYWORDS = {
    "on", "in", "above", "below", "under", "over", "beside",
    "next to", "near", "behind", "in front of", "left of",
    "right of", "between", "inside", "outside", "on top of",
    "at", "around", "across", "along", "within",
}

ACTION_KEYWORDS = {
    "hold", "carry", "eat", "drink", "ride", "wear", "walk",
    "run", "sit", "stand", "lie", "throw", "catch", "hit",
    "kick", "push", "pull", "hug", "touch", "grab", "feed",
    "look", "watch", "talk", "play", "drive", "climb", "jump",
    "lean", "rest", "hang", "point", "reach", "smile", "laugh",
}

ATTRIBUTE_KEYWORDS = {
    "be",  # copula: "the car is red" → (car, is, red)
}


@dataclass
class Triple:
    """A single (subject, relation, object) triple extracted from a caption."""
    subject: str
    relation: str
    obj: str                          # 'object' is a Python builtin — using 'obj'
    relation_type: str = "OTHER"      # SPATIAL | ACTION | ATTRIBUTE | OTHER
    confidence: float = 1.0          # reserved for future scoring
    hallucinated: Optional[bool] = None   # filled in by Stage 2 (Verifier)
    correction: Optional[str] = None      # filled in by Stage 3 (Corrector)

    def __str__(self):
        tag = f"[{self.relation_type}]"
        hallu = ""
        if self.hallucinated is True:
            hallu = " ⚠ HALLUCINATED"
        elif self.hallucinated is False:
            hallu = " ✓ verified"
        return f"{tag} ({self.subject}, {self.relation}, {self.obj}){hallu}"

    def as_dict(self):
        return {
            "subject": self.subject,
            "relation": self.relation,
            "object": self.obj,
            "relation_type": self.relation_type,
            "hallucinated": self.hallucinated,
            "correction": self.correction,
        }


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def classify_relation(relation: str) -> str:
    """Classify a relation string as SPATIAL, ACTION, ATTRIBUTE, or OTHER."""
    rel = relation.lower().strip()

    if rel in SPATIAL_KEYWORDS:
        return "SPATIAL"

    # Check action: use lemma matching
    if rel in ACTION_KEYWORDS:
        return "ACTION"

    # Copula → attribute
    if rel in ("is", "are", "was", "were", "be"):
        return "ATTRIBUTE"

    return "OTHER"


def _get_span_text(token) -> str:
    """
    For a noun token, return its full noun phrase (e.g. 'park bench' not just 'bench').
    We expand left to include compound modifiers and adjectives.
    """
    # Collect the subtree tokens that are part of the noun phrase,
    # but stop before any prepositional phrases.
    parts = []
    for t in token.subtree:
        if t.dep_ in ("prep", "relcl", "acl", "punct"):
            break
        if t.pos_ in ("NOUN", "PROPN", "ADJ", "NUM", "DET", "PART"):
            parts.append(t.text)
        elif t.dep_ == "compound":
            parts.append(t.text)
    if parts:
        return " ".join(parts).strip()
    return token.lemma_


def _find_child(token, deps: tuple):
    """Return the first child of token whose dep_ is in deps, or None."""
    for child in token.children:
        if child.dep_ in deps:
            return child
    return None


# ---------------------------------------------------------------------------
# Core extractor
# ---------------------------------------------------------------------------

class TripleExtractor:
    """
    Extracts (subject, relation, object) triples from a caption string
    using spaCy dependency parsing.

    Handles three structural patterns:
        Pattern A — Verb with subject + direct object
            "A woman is holding a dog" → (woman, hold, dog) [ACTION]

        Pattern B — Noun with prepositional phrase
            "a cup on the table" → (cup, on, table) [SPATIAL]

        Pattern C — Copula + predicate adjective
            "the car is red" → (car, is, red) [ATTRIBUTE]
    """

    def __init__(self, model_name: str = "en_core_web_sm"):
        print(f"[TripleExtractor] Loading spaCy model: {model_name}")
        self.nlp = spacy.load(model_name)
        print("[TripleExtractor] Ready.")

    def extract(self, caption: str) -> list[Triple]:
        """
        Main entry point. Takes a caption string, returns a list of Triples.

        Args:
            caption: Raw caption from an MLLM (e.g. from BLIP-2).

        Returns:
            List of Triple objects. May be empty if caption is too simple.
        """
        doc = self.nlp(caption)
        triples = []
        seen = set()  # deduplicate

        for token in doc:

            # ------------------------------------------------------------------
            # Pattern A: Verb → subject + direct object
            # e.g. "holding" → nsubj="woman", dobj="dog"
            # ------------------------------------------------------------------
            if token.pos_ in ("VERB",) and token.dep_ in (
                "ROOT", "relcl", "xcomp", "advcl", "conj"
            ):
                subj_tok = _find_child(token, ("nsubj", "nsubjpass"))
                obj_tok  = _find_child(token, ("dobj", "attr", "oprd"))

                if subj_tok and obj_tok:
                    s = _get_span_text(subj_tok)
                    r = token.lemma_
                    o = _get_span_text(obj_tok)
                    key = (s, r, o)
                    if key not in seen and s and o:
                        seen.add(key)
                        triples.append(Triple(
                            subject=s,
                            relation=r,
                            obj=o,
                            relation_type=classify_relation(r),
                        ))

            # ------------------------------------------------------------------
            # Pattern B: Noun → prep → pobj
            # e.g. "cup" → prep="on" → pobj="table"
            # ------------------------------------------------------------------
            if token.pos_ in ("NOUN", "PROPN"):
                for child in token.children:
                    if child.dep_ == "prep":
                        pobj_tok = _find_child(child, ("pobj",))
                        if pobj_tok:
                            s = _get_span_text(token)
                            r = child.lower_
                            o = _get_span_text(pobj_tok)
                            key = (s, r, o)
                            if key not in seen and s and o:
                                seen.add(key)
                                triples.append(Triple(
                                    subject=s,
                                    relation=r,
                                    obj=o,
                                    relation_type=classify_relation(r),
                                ))

            # ------------------------------------------------------------------
            # Pattern C: Copula → subject + attribute
            # e.g. "The car is red" → subj="car", cop="is", attr="red"
            # ------------------------------------------------------------------
            if token.dep_ == "attr" and token.pos_ in ("ADJ", "NOUN"):
                head = token.head  # the verb (usually "is")
                if head.dep_ == "ROOT":
                    subj_tok = _find_child(head, ("nsubj",))
                    if subj_tok:
                        s = _get_span_text(subj_tok)
                        r = head.lemma_  # "be"
                        o = token.lemma_
                        key = (s, r, o)
                        if key not in seen and s and o:
                            seen.add(key)
                            triples.append(Triple(
                                subject=s,
                                relation=r,
                                obj=o,
                                relation_type="ATTRIBUTE",
                            ))

        return triples

    def extract_batch(self, captions: list[str]) -> list[list[Triple]]:
        """Extract triples from a list of captions. Returns a list of lists."""
        return [self.extract(c) for c in captions]


# ---------------------------------------------------------------------------
# Quick test — run this cell directly in Colab to verify Day 1 works
# ---------------------------------------------------------------------------

TEST_CAPTIONS = [
    # Spatial
    "A cup is on the table next to a white mug.",
    # Action
    "A woman is holding a dog on a park bench near some trees.",
    # Multi-relation
    "A man riding a bicycle near a red car on the street.",
    # Attribute
    "The large dog is sitting beside a small child.",
    # Complex
    "Two birds are perched on a branch above a lake surrounded by mountains.",
]

EXPECTED = [
    ["(cup, on, table)", "(cup, next to, mug)"],
    ["(woman, hold, dog)", "(dog, on, park bench)", "(park bench, near, trees)"],
    ["(man, ride, bicycle)", "(bicycle, near, red car)", "(red car, on, street)"],
    ["(dog, sit, ...)"],   # attribute + action mix
    ["(birds, perch, branch)", "(branch, above, lake)"],
]


def run_tests(extractor: TripleExtractor):
    print("\n" + "="*70)
    print("TRIPLE EXTRACTOR — TEST RESULTS")
    print("="*70)
    for i, caption in enumerate(TEST_CAPTIONS):
        print(f"\nCaption {i+1}: \"{caption}\"")
        triples = extractor.extract(caption)
        if triples:
            for t in triples:
                print(f"  → {t}")
        else:
            print("  → (no triples extracted)")
    print("\n" + "="*70)
    print(f"Done. Processed {len(TEST_CAPTIONS)} captions.")
    print("="*70)


if __name__ == "__main__":
    extractor = TripleExtractor()
    run_tests(extractor)
