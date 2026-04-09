# How Woodpecker Generates Visual Claims

This document focuses on ONE thing: Woodpecker's 5-stage pipeline for diagnosing and correcting hallucinations. Specifically, the first 4 stages that build a "visual knowledge base" of claims about what's actually in the image.

Paper: "Woodpecker: Hallucination Correction for Multimodal Large Language Models" — Yin et al., 2023
Code: https://github.com/BradyFU/Woodpecker

---

## The Big Picture

Woodpecker is a training-free, post-hoc correction framework. An MLLM generates a response, and Woodpecker fixes it after the fact. It does NOT retrain the model.

The pipeline has 5 stages:

```
MLLM Response (possibly hallucinated)
    ↓
Stage 1: Key Concept Extraction     → what objects are mentioned?
    ↓
Stage 2: Question Formulation       → what should we verify about them?
    ↓
Stage 3: Visual Knowledge Validation → what does the image actually show?
    ↓
Stage 4: Visual Claim Generation    → structured facts about the image
    ↓
Stage 5: Hallucination Correction   → fix the response using the facts
```

---

## The Three Models Used

Woodpecker uses three off-the-shelf models (no training):

| Model | Role | What it does |
|---|---|---|
| GPT-3.5-turbo | LLM | Key concept extraction, question formulation, hallucination correction (Stages 1, 2, 5) |
| Grounding DINO | Open-set object detector | Determines if objects exist and how many, with bounding boxes (Stage 3, object-level) |
| BLIP-2-FlanT5-XXL | VQA model | Answers attribute questions about the image (Stage 3, attribute-level) |

---

## Stage 1: Key Concept Extraction

### What It Does
Takes the MLLM's generated text and extracts the main objects mentioned in it.

### Who Does It
GPT-3.5-turbo (text-only LLM, no image needed here)

### Input
The MLLM's response text. Example: "The man is wearing a black hat."

### Output
A list of objects: "man. hat"

### The Exact Prompt

System message:
```
You are a language assistant that helps to extract information 
from given sentences.
```

User prompt:
```
Given a sentence, extract the existent entities within the 
sentence for me.

Extract the common objects and summarize them as general 
categories without repetition, merge essentially similar objects.

Avoid extracting abstract or non-specific entities. Only extract 
concrete, certainly existent objects that fall in general 
categories and are described in a certain tone in the sentence.

Extract entity in the singular form. Output all the extracted 
types of items in one line and separate each object type with a 
period. If there is nothing to output, then output a single "None".

Examples:
{In-context examples}

Sentence:
{Input sentence}

Output:
```

### Key Rules
- Extract concrete objects only (not abstract concepts)
- Singular form ("hat" not "hats")
- Merge similar objects (don't list "dog" and "puppy" separately)
- Separated by periods: "man. hat. bicycle"
- Includes in-context examples to guide the LLM

---

## Stage 2: Question Formulation

### What It Does
Takes the extracted objects and the original sentence, and generates questions to verify whether the claims in the sentence are true.

### Who Does It
GPT-3.5-turbo (text-only LLM)

### Two Types of Questions

#### Object-Level Questions (hardcoded, not LLM-generated)
These are always the same format:
```
"Is there any {object} in the image? How many are there?"
```
One question per extracted object. These check existence and count.

#### Attribute-Level Questions (LLM-generated, context-dependent)
These are diverse and depend on what the sentence says:
```
"What is {object} doing?"
"Is {object1} on the right side of {object2}?"
"What color is the {object}?"
```
The LLM formulates these based on the sentence content.

### The Exact Prompt for Attribute Questions

System message:
```
You are a language assistant that helps to ask questions about 
a sentence.
```

User prompt:
```
Given a sentence and some entities connected by periods, you are 
required to ask some relevant questions about the specified 
entities involved in the sentence, so that the questions can help 
to verify the factuality of the sentence.

Questions may involve basic attributes such as colors and actions 
mentioned in the sentence. Do not ask questions involving object 
counts or the existence of objects.

When asking questions about attributes, try to ask simple 
questions that only involve one entity.

Ask questions that can be easily decided visually. Do not ask 
questions that require complex reasoning.

Do not ask semantically similar questions. Do not ask questions 
only about scenes or places.

Use "where" type questions to query the position information of 
the involved entities.

Do not ask questions about uncertain or conjecture parts of the 
sentence, for example, the parts described with "maybe" or 
"likely", etc.

It is no need to cover all the specified entities. If there is 
no question to ask, simply output a "None".

When asking questions, do not assume the claims in the 
description as true in advance. Only ask questions relevant to 
the information in the sentence.

Only ask questions about common, specific, and concrete entities. 
The entities involved in the questions are limited to the range 
within the given entities.

Output only one question in each line. For each line, first 
output the question, then a single "&", and finally entities 
involved in the question, still connected by periods if multiple 
entities are involved.

Examples:
{In-context examples}

Sentence:
{Input sentence}

Entities:
{Input entities}

Questions:
```

### Key Rules for Attribute Questions
- Don't ask about counts or existence (that's handled by object-level questions)
- Simple questions, one entity at a time
- Visually decidable (no complex reasoning)
- No duplicate/similar questions
- Use "where" for position queries
- Don't assume the sentence is true
- Output format: `question & entity1.entity2`

---

## Stage 3: Visual Knowledge Validation

### What It Does
Answers the questions from Stage 2 by actually looking at the image.

### Two Different Solvers for Two Types of Questions

#### Object-Level → Grounding DINO (open-set detector)
For "Is there any {object}? How many?":
- Grounding DINO detects all instances of the object in the image
- Returns: count + bounding boxes for each instance
- Uses default detection thresholds
- If count = 0 → object doesn't exist

#### Attribute-Level → BLIP-2-FlanT5-XXL (VQA model)
For attribute questions like "What color is the hat?":
- BLIP-2 answers the question conditioned on the image
- Returns: a short text answer
- Why BLIP-2 and not the MLLM? Because BLIP-2 generates shorter, less hallucinated answers than MLLMs

### Why Two Models?
- Grounding DINO is better at object detection (existence + counting + localization)
- BLIP-2 is better at answering attribute questions (color, action, position)
- Neither alone covers everything

---

## Stage 4: Visual Claim Generation

### What It Does
Converts the Q&A pairs from Stage 3 into a structured "visual knowledge base" — a set of factual claims about the image.

### Three Types of Claims

#### 1. Object-Level Claims (from Grounding DINO)
For objects that exist:
```
"There are {count} {name}."
```
Example: "There are 2 dogs."

For objects that don't exist:
```
"There is no {name}."
```
Example: "There is no cat."

#### 2. Attribute-Level Claims (from BLIP-2 via QA-to-Claim model)
The Q&A pairs are converted into declarative claims using a QA-to-Claim model (from Huang et al., 2023).

Example:
- Q: "What color is the hat?" A: "black"
- Claim: "The hat is black."

#### 3. Global/Interaction Claims
Claims about relationships between objects or objects and background:
```
"The cat is lying next to the dog."
```

### The Visual Knowledge Base Structure

The final knowledge base has three sections:

```
Count:
  - There are 2 dogs. [bbox1] [bbox2]
  - There is no cat.
  - There is 1 man. [bbox3]

Specific:
  - dog 1: [bbox1] The dog is brown.
  - dog 2: [bbox2] The dog is white.
  - man: [bbox3] The man is wearing a black hat.

Overall:
  - The man is standing next to the dogs.
```

---

## Stage 5: Hallucination Correction

### What It Does
Takes the original MLLM response + the visual knowledge base and produces a corrected response.

### Who Does It
GPT-3.5-turbo

### The Exact Prompt

System message:
```
You are a language assistant that helps to refine a passage 
according to instructions.
```

User prompt:
```
Given a passage and some supplementary information, you are 
required to correct and output the refined passage in a fluent 
and natural style, following these rules:

1. The supplementary information may include some of the 
following parts:
"Count" information that specifies how many instances of a 
certain kind of entity exist, and their associated bounding boxes;
"Specific" information that describes attribute information 
specific to each entity instance, including bounding boxes, 
colors, etc. The information is arranged in the form of 
"entity 1: [bbox]" info of this entity. Note that the entity in 
"Specific" information corresponds to that in the "Count" 
information.
"Overall" information that may involve information about multiple 
entity objects.

2. Try to retain the original sentence with minimal changes.

3. The number of entity instances should match the number in the 
"Count" information. Also correct the number counts if the number 
stated in the original sentence does not match the counting 
information.

4. If the original sentence is already correct, then just keep 
it. If you need to rewrite the original sentence, when rewriting, 
try to modify the original sentence as little as possible based 
on the original sentence, and use the supplementary information 
as guidance to correct or enrich the original sentence.

5. In the refined passage, when describing entities mentioned in 
the "Specific" supplementary information, add their associated 
bounding boxes in parentheses right after them, in the form of 
"entity([bbox])". If multiple entities of the same kind are 
mentioned, then separate the box with ";", in the form of 
"entity([bbox1];[bbox2])"

Examples:
{In-context examples}

———————-
Supplementary information:
{Input information}

Passage:
{Input passage}

Refined passage:
```

### Key Correction Rules
- Minimal changes to the original text
- Fix object counts to match detector output
- Fix attributes to match VQA answers
- Add bounding boxes as evidence (for interpretability)
- If the original is correct, keep it unchanged

---

## The Complete Flow, Concrete Example

Image: A man riding a bicycle in a park

MLLM says: "Two men are riding a red bicycle near a lake."

```
Stage 1 — Key Concept Extraction (GPT-3.5):
  Objects: "man. bicycle. lake"

Stage 2 — Question Formulation:
  Object-level (hardcoded):
    "Is there any man in the image? How many are there?"
    "Is there any bicycle in the image? How many are there?"
    "Is there any lake in the image? How many are there?"
  Attribute-level (GPT-3.5):
    "What color is the bicycle?" & bicycle
    "Where is the man?" & man

Stage 3 — Visual Knowledge Validation:
  Grounding DINO:
    man → 1 detected [0.2, 0.3, 0.5, 0.9]
    bicycle → 1 detected [0.3, 0.5, 0.7, 0.9]
    lake → 0 detected
  BLIP-2:
    "What color is the bicycle?" → "blue"
    "Where is the man?" → "on a bicycle in a park"

Stage 4 — Visual Claim Generation:
  Count:
    "There is 1 man." [0.2, 0.3, 0.5, 0.9]
    "There is 1 bicycle." [0.3, 0.5, 0.7, 0.9]
    "There is no lake."
  Specific:
    man: [0.2, 0.3, 0.5, 0.9] The man is on a bicycle in a park.
    bicycle: [0.3, 0.5, 0.7, 0.9] The bicycle is blue.
  Overall:
    The man is riding the bicycle in a park.

Stage 5 — Hallucination Correction (GPT-3.5):
  Input: "Two men are riding a red bicycle near a lake."
  + Knowledge base above
  Output: "A man([0.2,0.3,0.5,0.9]) is riding a blue 
           bicycle([0.3,0.5,0.7,0.9]) in a park."
```

Corrections made:
- "Two men" → "A man" (count fixed by detector)
- "red bicycle" → "blue bicycle" (color fixed by VQA)
- "near a lake" → "in a park" (lake doesn't exist per detector, location fixed by VQA)

---

## Two Types of Hallucination Woodpecker Handles

| Type | What it is | How detected | How fixed |
|---|---|---|---|
| Object-level | Wrong objects or wrong counts | Grounding DINO (detection) | Fix counts, remove non-existent objects |
| Attribute-level | Wrong color, position, action | BLIP-2 (VQA) | Fix attributes based on VQA answers |

---

## Comparison to RelCheck

| Dimension | Woodpecker | RelCheck |
|---|---|---|
| Hallucination types | Object-level + Attribute-level | Relation-level (spatial, action, attribute) |
| Detection method | Grounding DINO + BLIP-2 VQA | VQA probes + OWLv2 spatial verification |
| Correction method | GPT-3.5 with knowledge base | Llama-3.3-70B minimal correction |
| Structured representation | Visual knowledge base (Count/Specific/Overall) | (Subject, Relation, Object) triples |
| LLM used | GPT-3.5-turbo | Llama-3.3-70B (via Together.ai) |
| Object detector | Grounding DINO | OWLv2 |
| VQA model | BLIP-2-FlanT5-XXL | BLIP-2-FlanT5-XL |
| Training required | None | None |
| Bounding box evidence | Yes (added to corrected text) | No |
| Focus | Object existence + attributes | Relations between objects |
