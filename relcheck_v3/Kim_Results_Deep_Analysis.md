# How Kim et al. Evaluate Their Results

This document focuses on ONE thing: what metrics they use, how they set up their evaluation, and what the numbers mean.

---

## They Run Two Completely Separate Evaluations

The paper has two evaluation tracks that measure different things:

1. **Caption Editing Quality** — "Did the model fix the caption correctly?" (Tables 1 and 2)
2. **Hallucination Robustness** — "Does the model resist hallucinating objects?" (Table 3)

These are independent evaluations with different metrics, different test sets, and different baselines.

---

## Evaluation 1: Caption Editing Quality

### What They're Measuring

Given an image and a broken caption (Ref-Cap), the model produces an edited caption. They measure how close that edited caption is to the ground-truth caption (GT-Cap).

This is a standard **text generation evaluation** — compare generated text to reference text.

### The Five Metrics

All five metrics compare the model's output caption against the GT-Cap:

#### BLEU-1 (B-1)
- Measures **unigram precision** — what fraction of individual words in the output also appear in the GT-Cap
- Range: 0–100 (they report as percentages)
- Higher = better
- Tells you: "Did the model use the right words?"

#### BLEU-4 (B-4)
- Measures **4-gram precision** — what fraction of 4-word sequences in the output also appear in the GT-Cap
- Range: 0–100
- Higher = better
- Tells you: "Did the model produce the right phrases?" (much stricter than BLEU-1)
- Reference: Papineni et al., ACL 2002

#### ROUGE-L (R)
- Measures the **longest common subsequence** between output and GT-Cap
- Range: 0–100
- Higher = better
- Tells you: "How much of the sentence structure is preserved?"
- Reference: Lin, 2004

#### CIDEr (C)
- Measures **consensus-based similarity** using TF-IDF weighted n-grams
- Range: 0–1000+ (not bounded at 100)
- Higher = better
- Tells you: "How well does the output match what a human would write?" — it weights rare, informative words higher than common ones
- This is the most important metric for captioning tasks
- Reference: Vedantam et al., CVPR 2015

#### SPICE (S)
- Measures **semantic propositional similarity** — parses both captions into scene graphs (objects, attributes, relations) and compares them
- Range: 0–100
- Higher = better
- Tells you: "Does the output describe the same scene semantically?" — it doesn't care about exact wording, just meaning
- This is the most relevant metric for your RelCheck work because it explicitly evaluates relational structure
- Reference: Anderson et al., ECCV 2016

### How They Compute These Metrics

They don't implement these themselves. These are standard captioning evaluation metrics available in the `pycocoevalcap` package (the official COCO evaluation toolkit). You give it a list of (hypothesis, reference) pairs and it computes all five.

```python
# Standard usage
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
```

### The Test Sets

Two separate test sets, one per domain:

| Test Set | Source | Samples | What's in it |
|---|---|---|---|
| COCO-CE Test | COCO images + GPT-4 generated Ref-Caps | 5,366 | Only negative samples (all captions need editing) |
| Flickr30K-CE Test | Flickr30K images + GPT-4 generated Ref-Caps | 4,910 | Only negative samples (all captions need editing) |

Important: the test sets contain **only broken captions** (every sample needs editing). The 50/50 positive/negative split is only in the training data.

### The Test Prompt

At test time, the model receives this prompt:

```
"Please edit the following sentence to be consistent with the 
given image, making only the minimal necessary changes: [Ref-Cap]"
```

Note: this is a **simpler prompt** than the training prompt. The training prompt asks "Is this consistent? If not, edit it." The test prompt just says "edit it" — because all test samples are known to be incorrect.

### The Baseline: "Ref-Caps"

In every table, there's a row called "Ref-Caps". This is the score you get if you **do nothing** — just return the broken caption as-is. It's the floor. Any model that scores below Ref-Caps is making things worse.

Why Ref-Caps scores aren't zero: because the broken captions are only slightly different from the GT-Caps (edit distance < 50 characters). Most of the caption is still correct, so even the broken version gets decent BLEU/ROUGE/CIDEr scores.

### Two Evaluation Scenarios

#### In-Domain
- Train on COCO data → test on COCO test set
- Train on Flickr30K data → test on Flickr30K test set
- This measures: "How well does the model learn the specific editing task?"

#### Cross-Domain
- Train on Flickr30K data → test on COCO test set
- Train on COCO data → test on Flickr30K test set
- This measures: "Does the editing ability generalize to unseen image domains?"

---

## Table 1: Prior Explicit Editing Benchmark (COCO-EE / Flickr30K-EE)

This table uses TIger's original test sets (not Kim et al.'s new CE test sets). It compares against prior caption editing methods.

### What's Being Compared

| Model | What it is |
|---|---|
| Ref-Caps | Do nothing (return the broken caption) |
| ETN | Prior implicit editing method (Sammani & Melas-Kyriazi, CVPR 2020) |
| TIger | Prior explicit editing method (Wang et al., ECCV 2022) — the SOTA before this paper |
| DECap | Prior explicit editing via diffusion (Wang et al., ECCV 2024) |
| LLaVA-1.5 Baseline | Vanilla LLaVA-1.5 with no editing training |
| LLaVA-1.5 + COCO-EE | LLaVA-1.5 fine-tuned on TIger's COCO editing data |
| mPLUG-Owl2 Baseline | Vanilla mPLUG-Owl2 with no editing training |
| mPLUG-Owl2 + COCO-EE | mPLUG-Owl2 fine-tuned on TIger's COCO editing data |

### Key Finding from Table 1

Simply fine-tuning a VLM on TIger's editing data already beats all prior specialized methods:

```
TIger (prior SOTA):        BLEU-4 = 24.7,  CIDEr = 194.8
LLaVA-1.5 + COCO-EE:      BLEU-4 = 29.0,  CIDEr = 237.6  ← +22% CIDEr
```

This motivates the whole paper: VLMs are better at caption editing than specialized models, but they still hallucinate.

---

## Table 2: Context-Aware Caption Editing (Their New Benchmark)

This is the main results table. Uses their new CE test sets with GPT-4-generated broken captions.

### What's Being Compared

For each baseline VLM (LLaVA-1.5 and mPLUG-Owl2), they test these training configurations:

| Config | What was added to the 665K LLaVA base instructions |
|---|---|
| Baseline | Nothing — vanilla VLM |
| + EE | TIger's explicit editing data (prior work) |
| + MOS | Their multi-object selection task only |
| + MOS-CE | Both their tasks combined (the full proposed method) |

### The Progression (COCO-CE, LLaVA-1.5, In-Domain)

```
Ref-Caps (do nothing):     CIDEr = 507.6
Baseline:                  CIDEr = 511.8   ← barely better than doing nothing
+ COCO-EE:                 CIDEr = 327.2   ← WORSE than doing nothing!
+ COCO-MOS:                CIDEr = 524.4   ← small improvement
+ COCO-MOS-CE:             CIDEr = 769.8   ← massive jump (+50% over baseline)
```

The story:
1. Vanilla VLMs can barely edit captions (roughly equal to doing nothing)
2. TIger's EE data actually hurts on this benchmark (makes it worse)
3. MOS alone helps a little (better object perception)
4. MOS + CE together is dramatically better

### Why EE Hurts

TIger's COCO-EE dataset constructs reference captions by picking negative samples based on edit distance — the broken captions are often only superficially different from the GT-Caps. Training on this teaches the model to make tiny surface-level edits, which doesn't transfer well to the CE benchmark where the errors are more semantically meaningful (object swaps, interaction changes, count errors).

---

## Evaluation 2: Hallucination Robustness (POPE)

### What POPE Is

POPE (Polling-based Object Probing Evaluation) is a standard benchmark for measuring object hallucination in VLMs. It was introduced by Li et al. (EMNLP 2023).

### How POPE Works

It's dead simple. The model is asked yes/no questions about objects:

```
"Is there a [object] in the image?"
```

Half the questions are about objects that ARE in the image (answer: yes).
Half are about objects that are NOT in the image (answer: no).

The model's accuracy on these binary questions measures how well it can tell what's real vs. hallucinated.

### The Three POPE Settings

The key difference is how the **negative objects** (the ones NOT in the image) are chosen:

#### Random
- Negative objects are picked randomly from the 80 COCO categories
- Easiest setting — random objects are often obviously wrong
- Example: image has a cat → asked about "airplane" (easy to say no)

#### Popular
- Negative objects are the most frequently occurring objects in the dataset
- Medium difficulty — these objects appear a lot in training, so the model is biased toward saying "yes"
- Example: image has a cat → asked about "person" (person appears in tons of images)

#### Adversarial
- Negative objects are those that most frequently co-occur with the objects actually in the image
- Hardest setting — these are exactly the objects the model is most likely to hallucinate
- Example: image has a laptop → asked about "mouse" (laptops and mice often appear together)

### The Three POPE Domains

They evaluate on three different image sources:

| Domain | Source | Why |
|---|---|---|
| COCO | MS-COCO images | Same domain as training data |
| AOKVQA | A-OKVQA images | Tests generalization to VQA-style images |
| GQA | GQA images | Tests generalization to scene graph images |

### POPE Metrics

Two metrics per setting:

- **Accuracy (Acc)**: Simple binary classification accuracy — what % of yes/no answers are correct
- **F1 Score (F1)**: Harmonic mean of precision and recall — balances false positives (hallucinations) and false negatives (missed objects)

### Table 3 Results (LLaVA-1.5 only)

```
                        COCO Adversarial    AOKVQA Adversarial    GQA Adversarial
                        Acc     F1          Acc     F1            Acc     F1
Baseline:               85.3    84.3        88.4    88.2          91.2    91.2
+ COCO-EE:              85.5    84.4        87.6    87.3          90.9    91.0
+ COCO-MOS:             86.5    85.4        89.0    88.7          92.5    92.4
+ COCO-MOS-CE:          86.8    85.8        89.6    89.3          92.6    92.6
```

The story:
1. EE training doesn't help hallucination (and slightly hurts on AOKVQA/GQA)
2. MOS training improves hallucination resistance (the object selection task works)
3. Adding CE on top of MOS improves it further — caption editing training itself reduces hallucination

This is the paper's key finding: **caption editing and hallucination resistance are complementary**. Training the model to fix captions also makes it better at not hallucinating.

---

## The Qualitative Evaluation (Figure 3)

They show 5 examples from Flickr30K-CE with outputs from 5 models. This isn't scored — it's visual comparison. (The paper states "five caption editing examples" — four are shown with text in Figure 3, the fifth is image-only in the figure.)

### What They Show Per Example

| Column | What it is |
|---|---|
| Reference Caption (Input) | The broken caption fed to the model |
| Ground-Truth Caption | What the correct caption should be |
| TIger | Output from the prior SOTA explicit editing method (max 5 editing rounds, using released model trained on Flickr30K-EE) |
| LLaVA-1.5 | Output from vanilla LLaVA-1.5 (no editing training) |
| mPLUG-Owl2 | Output from vanilla mPLUG-Owl2 (no editing training) |
| LLaVA-1.5 + Flickr30K-EE | Output from LLaVA-1.5 trained on TIger's data |
| LLaVA-1.5 + Flickr30K-MOS-CE | Output from their full method |

### What They Highlight

They color-code the text:
- Red = incorrect descriptions
- Blue = correct descriptions
- Green = insertions that deviate from GT-Cap but are still consistent with the image

### The Failure Patterns They Identify

| Model | Failure Pattern |
|---|---|
| TIger | Over-deletes — removes too many words, loses context ("A man is riding a horse in the street" → "A man is riding a horse") |
| Vanilla VLMs | Blind adherence — copies the broken caption without changing anything |
| VLM + EE | Fixes the error but loses context ("Two people are standing near a cart" — drops "dressed differently" and "looking at vegetables") |
| VLM + MOS-CE (theirs) | Minimal edit — changes only the wrong part, preserves everything else |

---

## Summary: The Complete Evaluation Setup

```
Evaluation 1: Caption Editing Quality
├── Metrics: BLEU-1, BLEU-4, ROUGE-L, CIDEr, SPICE
├── Tool: pycocoevalcap (standard COCO eval toolkit)
├── Test sets: COCO-CE (5,366), Flickr30K-CE (4,910)
├── Scenarios: In-domain + Cross-domain
├── Baseline: Ref-Caps (do nothing)
└── Compared against: ETN, TIger, DECap, vanilla VLMs, VLM+EE, VLM+MOS, VLM+MOS-CE

Evaluation 2: Hallucination Robustness
├── Metric: POPE (Accuracy + F1)
├── Settings: Random, Popular, Adversarial
├── Domains: COCO, AOKVQA, GQA
├── All use 80 COCO object categories
└── Compared: Baseline, +EE, +MOS, +MOS-CE
```

---

## What This Means for RelCheck

The metrics most relevant to your work:

1. **SPICE** — directly evaluates relational/semantic structure (objects, attributes, relations as scene graphs). This is the closest existing metric to what RelCheck cares about.

2. **CIDEr** — the standard "did you get the caption right" metric for captioning. Good for overall quality.

3. **POPE Adversarial** — measures exactly the kind of hallucination your system detects (objects that co-occur but aren't present). Your R-POPE metric is inspired by this.

4. **Edit distance** — they don't use this as an evaluation metric, only as a constraint during data generation. But for RelCheck's correction evaluation, edit rate (how much the caption changed) is important for measuring minimality.
