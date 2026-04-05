# RelCheck Architecture Analysis — What Works, What Doesn't

## Current Pipeline Flow

```mermaid
flowchart TD
    A[Caption Input] --> B[Extract Triples via LLM]
    B --> C{For each triple}
    
    C --> D{SPATIAL triple?}
    C --> E{ACTION/ATTRIBUTE triple?}
    
    D --> D1[KB Synonym Match]
    D1 -->|"3.7% hit rate<br/>Only 3/81 triples"| D1Y[✅ CORRECT - HIGH]
    D1 -->|"96.3% miss"| D2[KB Opposite Match]
    D2 -->|"rare"| D2Y[❌ INCORRECT - HIGH]
    D2 -->|"miss"| D3[Geometry Contradiction Set]
    D3 -->|"rare"| D3Y[❌ INCORRECT - HIGH]
    D3 -->|"miss"| D4[VQA Fallback]
    D4 -->|"Does all the work<br/>43/68 INCORRECT"| D4R[Verdict from VQA]
    
    E --> E1[Geometry Pre-screen]
    E1 -->|"Only 15% have both bboxes<br/>5/33 triples"| E1R[Geo result]
    E1 -->|"85% skip"| E2
    E1R --> E2[Consensus Pre-filter]
    E2 -->|"Never fires<br/>cross_captions=None"| E3[VQA Verification]
    E3 -->|"Does all the work"| E3R[Verdict from VQA]
    
    D4R --> F[Collect all errors]
    E3R --> F
    D1Y --> G[No error]
    
    F --> H[Build Guidance per error]
    H --> I[Batch LLM Correction]
    I -->|"100% acceptance"| J[Post-verification]
    J -->|"0% revert rate"| K[Addendum]
    
    K --> K1[Spatial Addendum]
    K1 -->|"Adds ~30 words<br/>No accuracy impact"| K1R[Bloated caption]
    K --> K2[Missing Fact Addendum]
    K2 -->|"0% acceptance<br/>Always rejected"| K2R[Nothing added]
    
    K1R --> L[Final Caption]
    K2R --> L

    style D1 fill:#ff9999
    style D2 fill:#ff9999
    style D3 fill:#ff9999
    style E2 fill:#ff9999
    style K1 fill:#ff9999
    style K2 fill:#ff9999
    style D4 fill:#90EE90
    style E3 fill:#90EE90
    style I fill:#90EE90
    style H fill:#90EE90
```

**Legend:** 🟢 Green = pulling its weight, 🔴 Red = low/no impact

---

## Component Scorecard

### 🟢 HIGH IMPACT — Keep and improve

| Component | What it does | Evidence | Impact |
|---|---|---|---|
| **VQA Verification** | Asks VLM yes/no about each triple | 43/68 INCORRECT verdicts come from VQA | Core of the system |
| **Entity Existence Check** | GDino can't find entity → VQA confirms absence | 8/68 INCORRECT verdicts | Catches hallucinated objects |
| **KB Correct Relation Lookup** | Finds the RIGHT relation from KB spatial facts | 24/68 corrections use spatial_kb | Free, no API cost |
| **Batch LLM Correction** | Rewrites caption to fix errors | 100% acceptance rate | Actually fixes the text |
| **Triple Extraction** | Parses caption into (S, R, O) triples | 7.4 triples/image average | Foundation of everything |

### 🟡 MEDIUM IMPACT — Keep but needs improvement

| Component | What it does | Evidence | Issue |
|---|---|---|---|
| **Grasping Geometry** | Checks wrist keypoints near object | 3 correct decisions out of 4 | Only fires when GDino finds both bboxes (15%) |
| **Geometry-Grounded VQA Prompts** | Tells VQA what geometry found | +3.1% supplemental improvement | May cause false positives |
| **ViTPose Keypoints** | Gets body pose for action verification | 4 loads, 3/4 agreed with VQA | Limited by bbox coverage |

### 🔴 LOW/NO IMPACT — Candidates for removal

| Component | What it does | Evidence | Why it's dead weight |
|---|---|---|---|
| **Missing Fact Addendum** | LLM adds facts from visual description | 0% acceptance across ALL runs | Every attempt rejected by quality guards |
| **Spatial Addendum** | Appends KB spatial facts to caption | +30 words, 0% accuracy impact | Bloats captions, no benefit |
| **KB Synonym Match** | Word-level match of spatial relations | 3.7% hit rate (3/81 triples) | Too rigid, almost never matches |
| **Mounting Geometry** | Checks if subject is above object | 2 hits, 1 wrong | Too strict for real "sitting on" |
| **Consensus Pre-filter** | Cross-captioner agreement check | 0 fires (never used) | cross_captions always None |
| **Post-verification Revert** | Checks if correction introduced errors | 0% revert rate | Never triggers |

---

## What to prune

```mermaid
flowchart TD
    subgraph REMOVE["🔴 REMOVE — Zero impact"]
        R1[Missing Fact Addendum<br/>0% acceptance]
        R2[Consensus Pre-filter<br/>Never fires]
    end
    
    subgraph DISABLE["🟡 DISABLE BY DEFAULT — Negative/no impact"]
        D1[Spatial Addendum<br/>Bloats captions]
        D2[Mounting Geometry<br/>Too strict, 50% wrong]
    end
    
    subgraph REPLACE["🔄 REPLACE — With NLI"]
        P1[KB Synonym Match<br/>3.7% hit rate]
        P2["→ NLI semantic matching<br/>Should be much higher"]
    end
    
    subgraph KEEP["🟢 KEEP — Core value"]
        K1[VQA Verification]
        K2[Entity Existence]
        K3[KB Correct Rel Lookup]
        K4[Batch LLM Correction]
        K5[Grasping Geometry + ViTPose]
    end
    
    P1 --> P2
```

---

## Simplified Pipeline (after pruning)

```mermaid
flowchart TD
    A[Caption Input] --> B[Extract Triples via LLM]
    B --> C{For each triple}
    
    C --> D{SPATIAL?}
    C --> E{ACTION/ATTRIBUTE?}
    
    D --> N1[NLI: Check KB evidence]
    N1 -->|SUPPORT| N1Y[✅ CORRECT — skip VQA]
    N1 -->|CONTRADICT| N1C[Flag + confirm with VQA]
    N1 -->|NEUTRAL| D4[VQA Fallback]
    N1C --> D4
    D4 --> F
    
    E --> E1[Grasping Geometry + ViTPose]
    E1 -->|"geo result available"| E3[VQA with geometry hint]
    E1 -->|"no bboxes"| E3plain[VQA plain]
    E3 --> F
    E3plain --> F
    
    F[Collect errors] --> H[Build Guidance]
    H --> I[Batch LLM Correction]
    I --> L[Final Caption]
    
    style N1 fill:#90EE90
    style D4 fill:#90EE90
    style E3 fill:#90EE90
    style I fill:#90EE90
```

**What changed:**
- NLI replaces the rigid KB synonym/opposite matching
- Addendum removed entirely (both spatial and missing fact)
- Consensus pre-filter removed (unused)
- Mounting geometry removed (unreliable)
- Post-verification kept but simplified (it's cheap, just in case)

---

## Expected Impact of Pruning

| Metric | Before | After (expected) |
|---|---|---|
| KB hit rate | 3.7% | 20-40% (NLI semantic matching) |
| VQA calls per image | ~10 | ~6-7 (NLI skips some) |
| Caption word bloat | +30 words | +0 words (no addendum) |
| Wasted LLM calls | ~2/image (addendum) | 0 |
| Code complexity | 10 modules | 8 modules |
| Pipeline latency | ~30s/image | ~25s/image |

---

## Plain English Summary

Think of the pipeline like a factory assembly line:

1. **Triple Extraction** = the inspector who reads the caption and lists every claim ("man holding cup", "dog on couch")

2. **KB Synonym Match** = a dictionary lookup. "Does our dictionary say 'on' means the same as 'on top of'?" Almost never works because language is too varied. **Replace with NLI.**

3. **Geometry Check** = a ruler measurement. "Are the man's hands near the cup?" Only works when we can find both objects with GDino (half the time). **Keep grasping, remove mounting.**

4. **VQA** = showing the image to another AI and asking "Is this true?" This is the workhorse — does 63% of all error detection. **Keep, it's essential.**

5. **Entity Existence** = "Does this object even exist in the image?" Catches hallucinated objects like "snake" when there's no snake. **Keep.**

6. **Batch Correction** = the editor who rewrites the caption to fix errors. Works perfectly (100% acceptance). **Keep.**

7. **Addendum** = a fact-appender that adds extra info to captions. Spatial addendum bloats without helping. Missing fact addendum is rejected every single time. **Remove both.**

8. **Consensus** = checking if other captioners agree. Never used because we only run one captioner. **Remove.**
