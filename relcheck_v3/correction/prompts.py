"""Correction stage prompts.

Stage 5a (Thinking): GPT-5.4 with reasoning_effort=high identifies the
    specific hallucinated claim by cross-referencing caption against 3-layer KB.
Stage 5b (Correction): Verbatim Woodpecker correction prompt (Table 6,
    Appendix A.3) applies minimal surgical edits guided by KB evidence.
"""

# =============================================================================
# Stage 5a: Hallucination Identification (Thinking)
#
# GPT-5.4 with reasoning_effort=high analyzes the caption against the
# three-layer KB (CLAIM + GEOM + SCENE) to find the ONE inconsistency.
# =============================================================================

THINKING_SYSTEM_MESSAGE = (
    "You are a visual fact-checking assistant. You are given a caption "
    "describing an image and a Visual Knowledge Base (VKB) containing "
    "verified facts about that image from three sources:\n"
    "\n"
    "1. CLAIM — Object counts, specific attributes, and overall observations "
    "verified by an object detector and VQA model.\n"
    "2. GEOM — Deterministic spatial relationships computed from bounding box "
    "geometry (these are mathematically certain).\n"
    "3. SCENE — Scene graph triples from a relation detection model "
    "(subject-predicate-object with confidence scores).\n"
    "\n"
    "Your task is to find the ONE claim in the caption that contradicts "
    "the VKB. The caption has exactly one subtle hallucination — a small "
    "factual error (wrong object, wrong count, wrong attribute, or wrong "
    "spatial/action relationship). Everything else in the caption is correct.\n"
    "\n"
    "Think step by step:\n"
    "1. Break the caption into individual claims.\n"
    "2. For each claim, check whether the VKB supports, contradicts, or has "
    "no evidence for it.\n"
    "3. Prioritize GEOM facts (mathematically certain) over SCENE facts "
    "(model-based) over CLAIM facts (VQA-based).\n"
    "4. Identify the single claim that is most clearly contradicted.\n"
    "5. State what the caption says vs. what the VKB says.\n"
    "\n"
    "Output a JSON object with exactly these fields:\n"
    '  "hallucinated_span": the exact substring from the caption that is wrong\n'
    '  "reason": what the VKB says instead (cite the specific KB layer)\n'
    '  "correction_hint": what the span should be replaced with\n'
    '  "confidence": "high", "medium", or "low"\n'
)

# Placeholders: {caption}, {vkb}
THINKING_USER_TEMPLATE = (
    "Caption:\n"
    "{caption}\n"
    "\n"
    "Visual Knowledge Base:\n"
    "{vkb}\n"
    "\n"
    "Find the ONE hallucinated claim and output the JSON."
)


# =============================================================================
# Stage 5b: Hallucination Correction
#
# Verbatim Woodpecker correction prompt from Table 6 (Appendix A.3).
# Modified to include our KB evidence and the thinking stage output.
# =============================================================================

CORRECTION_SYSTEM_MESSAGE = (
    "You are a language assistant that helps to refine a passage "
    "according to instructions."
)

# Woodpecker's original correction prompt from Table 6 (Appendix A.3),
# extended with our hallucination identification output.
#
# Placeholders: {examples}, {kb_info}, {passage}, {hallucination_info}
CORRECTION_USER_TEMPLATE = (
    "Given a passage and some supplementary information, you are "
    "required to correct and output the refined passage in a fluent "
    "and natural style, following these rules:\n"
    "1. The supplementary information may include some of the "
    "following parts:\n"
    '   "Count" information that specifies how many instances of a '
    "certain kind of entity exist, and their associated bounding boxes;\n"
    '   "Specific" information that describes attribute information '
    "specific to each entity instance, including bounding boxes, colors, "
    "etc. The information is arranged in the form of "
    '"entity 1: [bbox] info of this entity". '
    "Note that the entity in "
    '"Specific" information corresponds to that in the "Count" '
    "information.\n"
    '   "Overall" information that may involve information about '
    "multiple entity objects.\n"
    "2. Try to retain the original sentence with minimal changes.\n"
    "3. The number of entity instances should match the number in "
    'the "Count" information. Also correct the number counts if the '
    "number stated in the original sentence does not match the "
    "counting information.\n"
    "4. If the original sentence is already correct, then just keep "
    "it. If you need to rewrite the original sentence, when "
    "rewriting, try to modify the original sentence as little as "
    "possible based on the original sentence, and use the "
    "supplementary information as guidance to correct or enrich the "
    "original sentence.\n"
    "5. IMPORTANT: Do NOT add bounding box coordinates to the output. "
    "Return only natural language text.\n"
    "\n"
    "Identified hallucination:\n"
    "{hallucination_info}\n"
    "\n"
    "---\n"
    "Supplementary information:\n"
    "{kb_info}\n"
    "\n"
    "Passage:\n"
    "{passage}\n"
    "\n"
    "Refined passage:"
)
