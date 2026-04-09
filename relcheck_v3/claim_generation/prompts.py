"""Verbatim prompt constants from the Woodpecker paper (Yin et al., 2023).

Source: https://github.com/VITA-MLLM/Woodpecker
Paper: "Woodpecker: Hallucination Correction for Multimodal Large Language Models"

All prompt templates use Python .format() compatible placeholders.
"""

# =============================================================================
# Stage 1: Key Concept Extraction
# Source: models/entity_extractor.py in the Woodpecker repo
# =============================================================================

STAGE1_SYSTEM_MESSAGE = (
    "You are a language assistant that helps to extract "
    "information from given sentences."
)

# In-context examples sourced from the Woodpecker GitHub repo
STAGE1_EXAMPLES = (
    "Sentence:\n"
    "The image depicts a man laying on the ground next to a motorcycle, "
    "which appears to have been involved in a crash.\n"
    "\n"
    "Output:\n"
    "man.motorcycle\n"
    "\n"
    "Sentence:\n"
    "There are a few people around, including one person standing close "
    "to the motorcyclist and another person further away.\n"
    "\n"
    "Output:\n"
    "person.motorcyclist\n"
    "\n"
    "Sentence:\n"
    "No, there is no car in the image.\n"
    "\n"
    "Output:\n"
    "car\n"
    "\n"
    "Sentence:\n"
    "The image depicts a group of animals, with a black dog, a white "
    "kitten, and a gray cat, sitting on a bed.\n"
    "\n"
    "Output:\n"
    "dog.cat.bed"
)

# User prompt template from Woodpecker paper Table 4 (Appendix A.1)
# Placeholders: {examples}, {sentence}
STAGE1_USER_TEMPLATE = (
    "Given a sentence, extract the entities within the sentence for me. \n"
    "Extract the common objects and summarize them as general categories "
    "without repetition, merge essentially similar objects.\n"
    "Avoid extracting abstract or non-specific entities. \n"
    "Extract entity in the singular form. Output all the extracted types "
    "of items in one line and separate each object type with a period. "
    'If there is nothing to output, then output a single "None".\n'
    "\n"
    "{examples}\n"
    "\n"
    "Sentence:\n"
    "{sentence}\n"
    "\n"
    "Output:"
)


# =============================================================================
# Stage 2: Question Formulation
# Source: models/questioner.py in the Woodpecker repo
# =============================================================================

STAGE2_SYSTEM_MESSAGE = (
    "You are a language assistant that helps to ask questions "
    "about a sentence."
)

# In-context examples sourced from the Woodpecker GitHub repo
STAGE2_EXAMPLES = (
    "Sentence:\n"
    "There are one black dog and two white cats in the image.\n"
    "\n"
    "Entities:\n"
    "dog.cat\n"
    "\n"
    "Questions:\n"
    "What color is the cat?&cat\n"
    "What color is the dog?&dog\n"
    "\n"
    "Sentence:\n"
    "The man is wearing a baseball cap and appears to be smoking.\n"
    "\n"
    "Entities:\n"
    "man\n"
    "\n"
    "Questions:\n"
    "What is the man wearing?&man\n"
    "What is the man doing?&man\n"
    "\n"
    "Sentence:\n"
    "The image depicts a busy kitchen, with a man in a white apron. "
    "The man is standing in the middle of the kitchen.\n"
    "\n"
    "Entities:\n"
    "kitchen.man\n"
    "\n"
    "Questions:\n"
    "What does the man wear?&man\n"
    "Is the man standing in the middle of the kitchen?&man.kitchen"
)

# User prompt template from Woodpecker paper Table 5 (Appendix A.2)
# Placeholders: {examples}, {sentence}, {entities}
STAGE2_USER_TEMPLATE = (
    "Given a sentence and some entities connnected by periods, "
    "you are required to ask some relevant questions about the specified "
    "entities involved in the sentence, so that the questions can help "
    "to verify the factuality of the sentence.\n"
    "Questions may involve basic attributes such as colors, actions "
    "mentioned in the sentence. Do not ask questions involving object "
    "counts or the existence of object.\n"
    "When asking questions about attributes, try to ask simple questions "
    "that only involve one entity. \n"
    "Ask questions that can be easily decided visually. Do not ask "
    "questions that require complex reasoning.\n"
    "Do not ask semantically similar questions. Do not ask questions "
    "only about scenes or places.\n"
    "Do not ask questions about uncertain or conjecture parts of the "
    "sentence, for example, the parts described with "
    '"maybe" or "likely", etc.\n'
    "It is no need to cover all the specified entities. If there is no "
    "question to ask, simply output a 'None'.\n"
    "When asking questions, do not assume the claims in the description "
    "as true in advance. Only ask questions relevant to the information "
    "in the sentence.\n"
    "Only ask questions about common, specific and concrete entities. "
    "The entities involved in the questions are limited to the range "
    "within the given entities.\n"
    "Output only one question in each line. For each line, first output "
    "the question, then a single '&', and finally entities involved in "
    "the question, still connected by periods if multiple entities are "
    "involved. \n"
    "\n"
    "Examples:\n"
    "{examples}\n"
    "\n"
    "Sentence:\n"
    "{sentence}\n"
    "\n"
    "Entities:\n"
    "{entities}\n"
    "\n"
    "Questions:"
)

# =============================================================================
# Stage 4: Visual Claim Generation — QA-to-Claim conversion
# The original Woodpecker uses a dedicated QA2C model (Huang et al., 2023):
# khhuang/zerofec-qa2claim-t5-base — we use the same model.
# These prompt constants are kept for reference / alternative usage.
# =============================================================================

STAGE4_QA_TO_CLAIM_SYSTEM_MESSAGE = (
    "You are a language assistant that converts question-answer pairs "
    "into declarative claim sentences."
)

# Placeholders: {question}, {answer}
STAGE4_QA_TO_CLAIM_TEMPLATE = (
    "Convert the following question and answer into a single declarative "
    "claim sentence.\n"
    "\n"
    "Question: {question}\n"
    "Answer: {answer}\n"
    "\n"
    "Claim:"
)

# =============================================================================
# Object-level question template (hardcoded, Stage 2)
# =============================================================================

# Placeholder: {object}
OBJECT_QUESTION_TEMPLATE = "Is there any {object} in the image? How many are there?"
