"""
RelCheck v2 — Prompt Templates
================================
All LLM/VLM prompt templates as named constants.
Used by correction.py, kb.py, and verification.py.
"""

# ── Enrichment mode (short captions, e.g. BLIP-2) ───────────────────────

ANALYSIS_PROMPT = """You are a caption quality improver. You have a short image caption and a detailed Visual Knowledge Base (KB) built from the actual image.

CAPTION: "{caption}"

=== VISUAL KNOWLEDGE BASE ===
DETECTED OBJECTS (highly reliable, from object detector):
{hard_facts}

SPATIAL RELATIONSHIPS (from bounding box geometry -- deterministic):
{spatial_facts}

VISUAL DESCRIPTION (from a separate vision model observing the image):
{visual_description}

=== TASK ===
Step 1: Identify ALL problems:
  - ERRORS: Any caption claim that the KB contradicts
  - MISSING: ALL important facts from the KB that the caption omits (objects, spatial positions, actions, attributes)

Step 2: Write a COMPREHENSIVE improved caption (5-8 sentences):
  - Fix any errors using KB evidence
  - Add ALL missing relationships, spatial positions, actions, and attributes from the KB
  - MUST cover spatial relationships: describe WHERE every detected object is relative to others
    (left, right, on top of, below, in front of, behind, near, etc.)
  - MUST cover actions & interactions: describe WHAT objects/people are doing together
  - MUST cover attributes: colors, sizes, materials, states for key objects
  - MUST cover all GroundingDINO detected objects -- do not omit any detected entity
  - The caption should be detailed enough that someone could answer any spatial, action,
    or attribute question about the image from the caption alone
  - Write naturally -- not a list, but flowing descriptive sentences
  - Only include facts supported by the KB

Output valid JSON:
{{"errors": [{{"claim": "...", "correction": "..."}}],
  "missing": [{{"fact": "...", "source": "..."}}],
  "improved_caption": "The comprehensive rewritten caption."}}"""


VERIFY_PROMPT = """Check if this caption is accurate based on KB evidence.

Caption: "{rewritten}"
KB objects: {objects}
KB relationships: {relationships}

FAIL only if caption:
  - Directly contradicts a KB fact
  - Has bad grammar or is incoherent
  - Contains nonsensical repetition
Do NOT fail just because the caption includes details not explicitly in KB.

Answer ONLY "PASS" or "FAIL: [reason]"."""


# ── Triple extraction ────────────────────────────────────────────────────

TRIPLE_EXTRACT_PROMPT = """Extract all relational claims from this image caption as structured triples.

Caption: "{caption}"

For each claim about how two entities relate, output a JSON object with:
  "subject"  : the entity performing or described first
  "relation" : the relationship word or short phrase (keep it brief, 1-4 words)
  "object"   : the entity being related to
  "type"     : one of SPATIAL | ACTION | ATTRIBUTE
    SPATIAL   = positional/directional (left of, right of, above, below, on, under,
                in front of, behind, next to, inside, near)
    ACTION    = dynamic interaction (riding, holding, eating, carrying, walking beside,
                playing with, pushing, throwing, sitting on, standing on)
    ATTRIBUTE = descriptive state (wearing, covered in, holding [as possession],
                decorated with, attached to, painted on)

Rules:
- Only include claims that describe a relationship BETWEEN two distinct entities.
- Skip generic single-entity descriptions ("a dog is brown").
- Use the shortest natural phrasing for relation (e.g. "on" not "is sitting on top of").
- subject and object should be short noun phrases (1-3 words).

Output ONLY a valid JSON array. No explanation. No markdown.
Example: [{{"subject": "dog", "relation": "left of", "object": "cat", "type": "SPATIAL"}},
          {{"subject": "man", "relation": "riding", "object": "horse", "type": "ACTION"}}]

If no relational claims exist, output: []"""


# ── Single-triple correction ─────────────────────────────────────────────

TRIPLE_CORRECT_PROMPT = """Edit exactly one relationship word or phrase in this caption.

Caption: "{caption}"

Task: The caption incorrectly describes how {subj} and {obj} relate to each other.
Find the word or phrase that expresses this relationship between {subj} and {obj}.
It may not be exactly "{wrong_phrase}" — the caption might use a synonym or different phrasing.
Replace it with: "{correct_phrase}"

Rules:
- Change ONLY the relationship word/phrase between {subj} and {obj}.
- Do NOT change any other words, word order, punctuation, or sentence structure.
- Do NOT add, remove, or reorder any sentences.
- The corrected caption must be the same length (±10%) as the original.
- Output the FULL corrected caption only. No explanation, no prefix, no quotes."""


# ── Batch correction ─────────────────────────────────────────────────────

BATCH_CORRECT_PROMPT = """You are correcting an image caption that contains confirmed factual errors.

Original caption:
"{caption}"

CONFIRMED ERRORS (fix ALL of these — do not skip any):
{error_list}

Rules (in priority order — higher rules override lower ones):
1. Fix EVERY error listed above — address each one explicitly.
2. WORD REPLACEMENT: If guidance says "Replace ONLY X with Y", find the exact word or phrase X in
   the sentence and replace it with Y. Do NOT change any other words in that sentence. Do NOT change
   the subject or object. Do NOT add, remove, or rewrite anything else.
3. SENTENCE DELETION: If guidance says "COMPLETELY DELETE" or "NOT in this image", remove the ENTIRE
   sentence containing the named claim. The sentence disappears completely — do NOT rephrase, soften,
   or keep any fragment of it. This overrides rule 4-7.
4. If a relationship is wrong and no specific replacement is given, correct only that relationship —
   keep all surrounding text intact.
5. Do NOT add new information not present in the original caption.
6. Do NOT shorten the caption by removing correct information (unless Rule 3 applies).
7. The output must be fluent, grammatical English — no dangling phrases or partial sentences.
8. CONSISTENCY: If you replace a word, replace ALL occurrences of that word used in the same sense.
9. Output the FULL corrected caption ONLY. No explanation, no prefix, no quotes."""


# ── Missing facts addendum ───────────────────────────────────────────────

MISSING_FACTS_PROMPT = """You are improving an image caption by inserting missing facts at semantically appropriate positions.

CAPTION:
"{caption}"

VISUAL DESCRIPTION (from a separate vision model — treat as ground truth):
"{visual_description}"

TASK:
1. Identify up to 2 important facts visible in the Visual Description but COMPLETELY ABSENT from the Caption.
   Missing facts can be anything: objects, actions, attributes, colors, counts, materials, spatial relationships, context.

   Priority order (highest value first):
     A. MISSING OBJECTS — an entire object or entity not mentioned at all in the caption.
        When adding a missing object, include WHERE it is relative to the main scene naturally
        (e.g. "a wooden box resting on the windowsill above the sink").
        This is the most valuable type of insertion because it answers relational questions.
     B. MISSING ACTIONS — what a person or animal is doing, if not described.
     C. MISSING ATTRIBUTES — important attributes (count, color, material) of objects already in caption.

   Skip: vague background details, minor clutter, anything already implied or easily inferred.

2. For each missing fact, find the sentence in the caption it belongs with and insert it there naturally.
   The right sentence is the one that already discusses the same subject or scene context.
   - Missing object → insert in the most scene-descriptive sentence, or append a short sentence
   - Missing action → insert in the sentence describing what that entity is doing
   - Missing attribute → insert in the sentence describing that entity
   - If truly no sentence fits, append one short sentence at the end

RULES:
- Only INSERT — never delete, rephrase, or shorten any existing text.
- Insert BETWEEN complete clauses or at sentence boundaries — NEVER inside an existing noun phrase or adjective phrase.
  Wrong: "lettering on the knee the right knee"  ← inserted inside "on the right knee"
  Right: "lettering on the right knee, and he is also wearing X"  ← appended after
- Use natural conjunctions ("and", "also", "along with", "with", ", including") to weave in the new fact.
- When introducing a missing object, include its location relative to other objects naturally.
- If nothing important is missing: output exactly NONE.

Output the full modified caption ONLY. No explanation, no prefix."""


# ── Visual KB construction ───────────────────────────────────────────────

KB_DESCRIPTION_PROMPT = """Describe the RELATIONSHIPS between objects in this image.

An object detector found these objects:
{detection_list}

For each pair of detected objects that interact, describe:
- SPATIAL: Where are they relative to each other?
- ACTIONS: What is each person/animal doing?
- ATTRIBUTES: Relevant attributes tied to relationships

Rules: only describe what you can clearly see. Be brief and factual.
Format as a numbered list."""
