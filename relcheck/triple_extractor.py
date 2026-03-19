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

SPATIAL_KEYWORDS = {
    "on", "in", "above", "below", "under", "over", "beside",
    "next to", "near", "behind", "in front of", "left of",
    "right of", "between", "inside", "outside", "on top of",
    "at", "around", "across", "along", "within", "with",
}

ACTION_KEYWORDS = {
    "hold", "carry", "eat", "drink", "ride", "wear", "walk",
    "run", "sit", "stand", "lie", "throw", "catch", "hit",
    "kick", "push", "pull", "hug", "touch", "grab", "feed",
    "look", "watch", "talk", "play", "drive", "climb", "jump",
    "lean", "rest", "hang", "point", "reach", "smile", "laugh",
    "hide", "swing", "perch",
}

ATTRIBUTE_KEYWORDS = {
    "be",
}


@dataclass
class Triple:
    """A single (subject, relation, object) triple extracted from a caption."""
    subject: str
    relation: str
    obj: str
    relation_type: str = "OTHER"
    confidence: float = 1.0
    hallucinated: Optional[bool] = None
    correction: Optional[str] = None

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
    if rel in ACTION_KEYWORDS:
        return "ACTION"
    if rel in ("is", "are", "was", "were", "be"):
        return "ATTRIBUTE"
    return "OTHER"


def _get_span_text(token) -> str:
    """Return the full noun phrase for a token."""
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

    Handles four structural patterns:
        Pattern A — Verb with subject + direct object
            "A woman is holding a dog" → (woman, hold, dog) [ACTION]
            "A kitten wearing a chef hat" → (kitten, wear, chef hat) [ACTION]

        Pattern B — Noun with prepositional phrase
            "a cup on the table" → (cup, on, table) [SPATIAL]

        Pattern C — Copula + predicate adjective or noun
            "the car is red" → (car, be, red) [ATTRIBUTE]
            "the hair dryer is black" → (hair dryer, be, black) [ATTRIBUTE]

        Pattern D — Verb with subject + prepositional phrase
            "a cat hiding under a couch" → (cat, under, couch) [SPATIAL]
            "a couple sitting on an escalator" → (couple, on, escalator) [SPATIAL]
            "a baby playing with a toy box" → (baby, with, toy box) [SPATIAL]
            "two men smiling at a table" → (men, at, table) [SPATIAL]
    """

    def __init__(self, model_name: str = "en_core_web_sm"):
        print(f"[TripleExtractor] Loading spaCy model: {model_name}")
        self.nlp = spacy.load(model_name)
        print("[TripleExtractor] Ready.")

    def extract(self, caption: str) -> list[Triple]:
        """
        Main entry point. Takes a caption string, returns a list of Triples.
        """
        doc = self.nlp(caption)
        triples = []
        seen = set()

        VERB_DEPS = ("ROOT", "relcl", "xcomp", "advcl", "conj", "acl", "ccomp")

        for token in doc:

            # ------------------------------------------------------------------
            # Pattern A: Verb → subject + direct object
            # Also handles participial modifiers (acl) where head noun = subject
            # ------------------------------------------------------------------
            if token.pos_ in ("VERB",) and token.dep_ in VERB_DEPS:
                subj_tok = _find_child(token, ("nsubj", "nsubjpass"))

                if subj_tok is None and token.dep_ == "acl":
                    head = token.head
                    if head.pos_ in ("NOUN", "PROPN"):
                        subj_tok = head

                obj_tok = _find_child(token, ("dobj", "attr", "oprd"))

                if subj_tok and obj_tok:
                    s = _get_span_text(subj_tok)
                    r = token.lemma_
                    o = _get_span_text(obj_tok)
                    key = (s, r, o)
                    if key not in seen and s and o:
                        seen.add(key)
                        triples.append(Triple(
                            subject=s, relation=r, obj=o,
                            relation_type=classify_relation(r),
                        ))

            # ------------------------------------------------------------------
            # Pattern B: Noun → prep → pobj
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
                                    subject=s, relation=r, obj=o,
                                    relation_type=classify_relation(r),
                                ))

            # ------------------------------------------------------------------
            # Pattern C: Copula → subject + predicate (attr or acomp)
            # attr  = noun predicates:  "the car is a truck"
            # acomp = adjective predicates: "the car is red"
            # ------------------------------------------------------------------
            if token.dep_ in ("attr", "acomp") and token.pos_ in ("ADJ", "NOUN", "ADV"):
                head = token.head
                if head.dep_ in ("ROOT", "ccomp"):
                    subj_tok = _find_child(head, ("nsubj",))
                    if subj_tok:
                        s = _get_span_text(subj_tok)
                        r = head.lemma_
                        o = token.lemma_
                        key = (s, r, o)
                        if key not in seen and s and o:
                            seen.add(key)
                            triples.append(Triple(
                                subject=s, relation=r, obj=o,
                                relation_type="ATTRIBUTE",
                            ))

            # ------------------------------------------------------------------
            # Pattern D: Verb → subject + prep → pobj
            # Captures spatial/other relations expressed as verb+preposition:
            # "hiding under", "sitting on", "playing with", "smiling at"
            # ------------------------------------------------------------------
            if token.pos_ in ("VERB", "AUX") and token.dep_ in VERB_DEPS:
                subj_tok = _find_child(token, ("nsubj", "nsubjpass"))

                if subj_tok is None and token.dep_ == "acl":
                    head = token.head
                    if head.pos_ in ("NOUN", "PROPN"):
                        subj_tok = head

                if subj_tok:
                    for child in token.children:
                        if child.dep_ == "prep":
                            pobj_tok = _find_child(child, ("pobj",))
                            if pobj_tok:
                                s = _get_span_text(subj_tok)
                                r = child.lower_
                                o = _get_span_text(pobj_tok)
                                key = (s, r, o)
                                if key not in seen and s and o:
                                    seen.add(key)
                                    triples.append(Triple(
                                        subject=s, relation=r, obj=o,
                                        relation_type=classify_relation(r),
                                    ))

        return triples

    def extract_batch(self, captions: list[str]) -> list[list[Triple]]:
        """Extract triples from a list of captions. Returns a list of lists."""
        return [self.extract(c) for c in captions]


# ---------------------------------------------------------------------------
# LLM-based Triple Extractor  (Llama-3.1-8B via Groq)
# ---------------------------------------------------------------------------

class LLMTripleExtractor:
    """
    Uses Llama-3.1-8B-Instant via Groq to extract triples from captions.

    Advantages over spaCy dependency parsing:
      - Handles complex sentence structures, passive voice, relative clauses
      - Extracts implicit relations spaCy misses
      - More natural relation lemmatization
      - Falls back to spaCy if API call fails

    The prompt is designed to produce clean pipe-separated output that parses
    deterministically, with relation type classification included.
    """

    MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

    SYSTEM_PROMPT = (
        "You are a precise information extraction engine. "
        "Extract ALL (subject, relation, object) triples from image captions. "
        "Be exhaustive — every spatial, action, and attribute relation counts."
    )

    USER_TEMPLATE = """Extract every (subject, relation, object) triple from this image caption.

Caption: "{caption}"

Rules:
- Subject and object must be concrete noun phrases (no pronouns)
- Relation must be a single verb, preposition, or short phrase
- Classify each relation as: SPATIAL, ACTION, or ATTRIBUTE
  * SPATIAL: positional/spatial prepositions (on, under, near, beside, above, etc.)
  * ACTION: verbs describing what someone/something is doing (hold, ride, eat, wear, etc.)
  * ATTRIBUTE: copula + adjective/noun (is red, is large, is a dog, etc.)

Output format — one triple per line, nothing else:
subject | relation | object | TYPE

Example output:
woman | hold | dog | ACTION
dog | on | leash | SPATIAL
woman | near | bench | SPATIAL
bench | be | wooden | ATTRIBUTE"""

    def __init__(self, api_key: str, fallback_extractor: "TripleExtractor | None" = None):
        import together
        self.client   = together.Together(api_key=api_key)
        self.fallback = fallback_extractor   # spaCy fallback if API fails

    def extract(self, caption: str) -> list[Triple]:
        """Extract triples using Llama-3.1-8B (Groq), falling back to spaCy on failure."""
        try:
            return self._extract_llm(caption)
        except Exception as e:
            print(f"[LLMTripleExtractor] API error: {e} — falling back to spaCy")
            if self.fallback is not None:
                return self.fallback.extract(caption)
            return []

    def _extract_llm(self, caption: str) -> list[Triple]:
        response = self.client.chat.completions.create(
            model=self.MODEL,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user",   "content": self.USER_TEMPLATE.format(caption=caption)},
            ],
            max_tokens=300,
            temperature=0.0,   # deterministic extraction
        )
        raw = response.choices[0].message.content.strip()
        return self._parse_output(raw)

    def _parse_output(self, raw: str) -> list[Triple]:
        triples = []
        seen    = set()
        for line in raw.splitlines():
            line = line.strip()
            if not line or "|" not in line:
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 3:
                continue
            s, r, o = parts[0], parts[1], parts[2]
            if not s or not o or not r:
                continue
            # Relation type — use LLM output if valid, else classify from relation string
            rel_type = parts[3].upper().strip() if len(parts) >= 4 else ""
            if rel_type not in RELATION_TYPES:
                rel_type = classify_relation(r)
            key = (s.lower(), r.lower(), o.lower())
            if key not in seen:
                seen.add(key)
                triples.append(Triple(
                    subject=s, relation=r, obj=o, relation_type=rel_type
                ))
        return triples

    def extract_batch(self, captions: list[str]) -> list[list[Triple]]:
        return [self.extract(c) for c in captions]


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

TEST_CAPTIONS = [
    "A cup is on the table next to a white mug.",
    "A woman is holding a dog on a park bench near some trees.",
    "A man riding a bicycle near a red car on the street.",
    "The large dog is sitting beside a small child.",
    "Two birds are perched on a branch above a lake surrounded by mountains.",
    "A cat is hiding under a couch.",
    "A couple sitting on an escalator.",
    "A kitten wearing a chef hat.",
    "The hair dryer is black and gold.",
    "A baby is playing with a toy box.",
    "Two men smiling at a table with food.",
    "A man is swinging a baseball bat in a field.",
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
