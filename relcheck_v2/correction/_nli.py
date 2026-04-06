"""
NLI-based KB verification module.
==================================
Types and functions for Natural Language Inference pre-filtering
of triple verification against Visual Knowledge Base evidence.

Uses text-only LLM (Llama-3.3-70B) to classify whether KB evidence
SUPPORTS, CONTRADICTS, or is NEUTRAL toward each caption claim,
enabling tiered confidence decisions before expensive VQA calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from ..config import ABSTRACT_ENTITIES, ENTITY_SYNONYMS, NLI_USE_VISUAL_DESCRIPTION
from .._logging import log
from ..api import llm_call
from ..types import Triple
from ._utils import core_noun, entity_matches
from ._vqa import _parse_spatial_facts

# ── Enums ────────────────────────────────────────────────────────────────


class NLIVerdict(str, Enum):
    """Result of NLI entailment check against KB evidence."""

    SUPPORT = "SUPPORT"
    CONTRADICT = "CONTRADICT"
    NEUTRAL = "NEUTRAL"


class EvidenceSource(str, Enum):
    """Origin of the KB evidence that drove the NLI verdict."""

    SPATIAL_FACT = "spatial_fact"
    VISUAL_DESCRIPTION = "visual_description"
    ENTITY_EXISTENCE = "entity_existence"
    SCENE_GRAPH = "scene_graph"
    MIXED = "mixed"


# ── Dataclasses ──────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class NLIResult:
    """Outcome of a single NLI KB verification.

    Attributes:
        verdict: SUPPORT, CONTRADICT, or NEUTRAL.
        evidence_used: KB facts that were fed to the LLM.
        evidence_source: What type of evidence drove the verdict.
        raw_response: Raw LLM output for debugging (None if no call made).
    """

    verdict: NLIVerdict
    evidence_used: list[str]
    evidence_source: EvidenceSource
    raw_response: str | None


# ── Visual Description Parsing ───────────────────────────────────────────


def _parse_visual_description_sentences(
    visual_description: str,
) -> list[str]:
    """Split visual description into sentences, handling numbered formats.

    Handles both plain period-delimited text and numbered formats
    like '1. A cat is sitting. 2. A dog is nearby.'
    """
    if not visual_description or not visual_description.strip():
        return []

    text = visual_description.strip()

    # Detect numbered format: "1. ... 2. ... 3. ..."
    # Match digit(s) followed by ". " anywhere in text
    num_prefix_re = re.compile(r'\d+\.\s+')
    numbered_matches = list(num_prefix_re.finditer(text))
    if len(numbered_matches) >= 2 and numbered_matches[0].start() == 0:
        # Split by numbered prefixes
        sentences = []
        for i, match in enumerate(numbered_matches):
            start = match.end()
            end = numbered_matches[i + 1].start() if i + 1 < len(numbered_matches) else len(text)
            chunk = text[start:end].strip().rstrip(".")
            if chunk:
                sentences.append(chunk)
        return sentences

    # Fallback: plain period-delimited splitting
    return [s.strip() for s in text.split(".") if s.strip()]

# ── Scene Graph Entity Matching ──────────────────────────────────────────

def _sg_entity_match(caption_entity: str, sg_entity: str) -> bool:
    """Lenient entity match for scene graph Phase 4 evidence.

    Extends ``entity_matches`` with ENTITY_SYNONYMS lookup so that
    RelTR's limited vocabulary (e.g. "person", "phone", "bag") can
    match richer caption entities (e.g. "skateboarders", "cell phone",
    "handbag").

    Cascade:
        1. ``entity_matches`` (substring + rapidfuzz)
        2. Synonym set intersection via ENTITY_SYNONYMS
        3. Single-word overlap between core nouns
    """
    # Level 1: standard matching (substring + fuzzy)
    if entity_matches(caption_entity, sg_entity):
        return True

    cap_core = core_noun(caption_entity)
    sg_core = core_noun(sg_entity)
    if not cap_core or not sg_core:
        return False

    # Level 2: synonym intersection
    cap_syns = {cap_core}
    sg_syns = {sg_core}
    for canonical, syns in ENTITY_SYNONYMS.items():
        syns_lower = [s.lower() for s in syns]
        if cap_core in syns_lower or cap_core == canonical:
            cap_syns.update(s.lower() for s in syns)
            cap_syns.add(canonical)
        if sg_core in syns_lower or sg_core == canonical:
            sg_syns.update(s.lower() for s in syns)
            sg_syns.add(canonical)
    if cap_syns & sg_syns:
        return True

    # Level 3: single-word overlap (catches "baseball glove" ↔ "glove",
    # "tennis racket" ↔ "racket", "dining table" ↔ "table")
    cap_words = set(cap_core.split())
    sg_words = set(sg_core.split())
    if cap_words & sg_words:
        return True

    return False


# ── Evidence Collection ──────────────────────────────────────────────────


def collect_nli_evidence(
    subj: str,
    obj: str,
    spatial_facts: list[str],
    visual_description: str,
    hard_facts: list[str],
    scene_graph: list[dict] | None = None,
) -> list[str]:
    """Gather KB evidence relevant to a subject-object pair.

    Phase 1: spatial facts where both entities match (both orderings).
    Phase 2: entity existence — negative signal for non-abstract entities
             with zero KB detections.
    Phase 3: visual description sentences mentioning both entities
             (synonym-aware via ENTITY_SYNONYMS).

    Returns evidence ordered: spatial → entity existence → visual desc,
    capped at 10 strings.
    """
    evidence: list[str] = []

    # Phase 1: Spatial facts where both entities match
    kb_triples = _parse_spatial_facts(spatial_facts)
    for kb_s, kb_r, kb_o in kb_triples:
        if entity_matches(subj, kb_s) and entity_matches(obj, kb_o):
            evidence.append(f"[spatial_fact] {kb_s} is {kb_r} {kb_o}")
        elif entity_matches(subj, kb_o) and entity_matches(obj, kb_s):
            evidence.append(f"[spatial_fact] {kb_o} is {kb_r} {kb_s}")

    # Phase 2: Entity existence check against hard_facts
    subj_core = core_noun(subj)
    obj_core = core_noun(obj)

    def _entity_detected(entity_core: str) -> bool:
        for fact in hard_facts:
            fact_lower = fact.lower()
            if entity_core in fact_lower:
                return True
            for syn in ENTITY_SYNONYMS.get(entity_core, []):
                if syn in fact_lower:
                    return True
        return False

    if subj_core and subj_core not in ABSTRACT_ENTITIES:
        if not _entity_detected(subj_core):
            evidence.append(
                f"[entity_existence] No '{subj_core}' was detected in the "
                f"image by the object detector."
            )

    if obj_core and obj_core not in ABSTRACT_ENTITIES:
        if not _entity_detected(obj_core):
            evidence.append(
                f"[entity_existence] No '{obj_core}' was detected in the "
                f"image by the object detector."
            )

    # Phase 3: Visual description sentences mentioning both entities
    if NLI_USE_VISUAL_DESCRIPTION and visual_description:
        sentences = _parse_visual_description_sentences(visual_description)
        for sentence in sentences:
            sent_lower = sentence.lower()
            subj_found = subj_core in sent_lower or any(
                syn in sent_lower for syn in ENTITY_SYNONYMS.get(subj_core, [])
            )
            obj_found = obj_core in sent_lower or any(
                syn in sent_lower for syn in ENTITY_SYNONYMS.get(obj_core, [])
            )
            if subj_found and obj_found:
                evidence.append(f"[visual_description] {sentence.strip()}")

    # Phase 4: Scene graph evidence
    # Include all scene graph triples — the LLM is better at entity
    # matching ("man" = "person", "racket" = "tennis racket") than
    # our string-based heuristics. With 4–21 triples per image this
    # is negligible in the 128K context window.
    if scene_graph:
        for triple in scene_graph:
            t_subj = triple.get("subject", "")
            t_pred = triple.get("predicate", "")
            t_obj = triple.get("object", "")
            t_conf = triple.get("predicate_conf", 0.0)
            evidence.append(
                f"[scene_graph] {t_subj} {t_pred} {t_obj} (conf={t_conf:.2f})"
            )

    return evidence


# ── Evidence Source Classification ───────────────────────────────────────


def classify_evidence_source(evidence: list[str]) -> EvidenceSource:
    """Determine the dominant evidence source type from evidence strings.

    Returns SPATIAL_FACT / VISUAL_DESCRIPTION / ENTITY_EXISTENCE if all
    evidence shares one tag, MIXED if multiple tag types present.
    """
    sources: set[EvidenceSource] = set()
    for e in evidence:
        if e.startswith("[spatial_fact]"):
            sources.add(EvidenceSource.SPATIAL_FACT)
        elif e.startswith("[entity_existence]"):
            sources.add(EvidenceSource.ENTITY_EXISTENCE)
        elif e.startswith("[visual_description]"):
            sources.add(EvidenceSource.VISUAL_DESCRIPTION)
        elif e.startswith("[scene_graph]"):
            sources.add(EvidenceSource.SCENE_GRAPH)

    if len(sources) == 0:
        return EvidenceSource.VISUAL_DESCRIPTION  # fallback
    if len(sources) == 1:
        return sources.pop()
    return EvidenceSource.MIXED


# ── Response Parsing ─────────────────────────────────────────────────────


def parse_nli_response(raw: str) -> NLIVerdict:
    """Parse the LLM's raw text response into an NLIVerdict.

    SUPPORT if "SUPPORT" present and "CONTRADICT" absent (case-insensitive).
    CONTRADICT if "CONTRADICT" present and "SUPPORT" absent.
    NEUTRAL otherwise.
    """
    upper = raw.upper()
    has_support = "SUPPORT" in upper
    has_contradict = "CONTRADICT" in upper
    if has_support and not has_contradict:
        return NLIVerdict.SUPPORT
    if has_contradict and not has_support:
        return NLIVerdict.CONTRADICT
    return NLIVerdict.NEUTRAL


_BATCH_LINE_RE = re.compile(
    r'(\d+)\s*[:\.\-]\s*(SUPPORT|CONTRADICT|NEUTRAL)', re.IGNORECASE
)


def parse_batch_nli_response(
    raw: str,
    n_claims: int,
) -> list[NLIVerdict]:
    """Parse the LLM's multi-line batch response into per-claim verdicts.

    Expects lines like '1: SUPPORT', '2: CONTRADICT', etc.
    Missing or unparseable lines default to NEUTRAL.
    Returns exactly n_claims verdicts.
    """
    verdicts = [NLIVerdict.NEUTRAL] * n_claims

    for match in _BATCH_LINE_RE.finditer(raw):
        idx = int(match.group(1)) - 1  # 1-indexed → 0-indexed
        verdict_str = match.group(2).upper()
        if 0 <= idx < n_claims:
            try:
                verdicts[idx] = NLIVerdict(verdict_str)
            except ValueError:
                pass  # keep NEUTRAL

    return verdicts


# ── Prompt Template ──────────────────────────────────────────────────────

NLI_BATCH_PROMPT_TEMPLATE = """You are a factual consistency checker. Given a set of caption claims and knowledge base evidence, determine if the evidence SUPPORTS, CONTRADICTS, or is NEUTRAL toward each claim.

IMPORTANT — Semantic matching guidance:
- Spatial synonyms: "to the left of" and "beside" can both mean "near". "on" and "on top of" are equivalent. "above" and "over" are equivalent. Focus on whether the core spatial relationship is consistent, not exact wording.
- Action synonyms: "playing with" and "interacting with" are semantically similar. "holding" and "carrying" overlap. "sitting on" and "resting on" are equivalent. Focus on whether the core action/interaction is consistent.
- Entity synonyms: "man" and "person" refer to the same entity. "car" and "vehicle" are equivalent. Match entities by meaning, not exact words.
- Negation from absence: If the object detector found NO instances of an entity, that is strong evidence the entity is not present in the image.
- Scene graph evidence: Relations tagged [scene_graph] come from a visual classifier (RelTR) with a limited vocabulary of ~50 predicates. Treat these as strong visual evidence when the predicate matches or is semantically close to the claim.

Verdict definitions:
- SUPPORT: The evidence clearly confirms the claim is true (semantic match, not just word overlap)
- CONTRADICT: The evidence clearly shows the claim is false (e.g., opposite spatial relation, entity not detected)
- NEUTRAL: The evidence does not address this specific claim, or is ambiguous

{claims_and_evidence_block}

For each claim, respond with exactly one line in this format:
<number>: SUPPORT, CONTRADICT, or NEUTRAL

Example response format:
1: SUPPORT
2: CONTRADICT
3: NEUTRAL"""

_TAG_RE = re.compile(r'^\[(spatial_fact|visual_description|entity_existence|scene_graph)\]\s*')


def format_batch_nli_prompt(
    claims: list[str],
    evidence_map: dict[int, list[str]],
) -> str:
    """Build a single batched NLI prompt for all triples in one image.

    Strips internal evidence source tags before including in prompt.
    Claims with no evidence get "No relevant evidence available".
    """
    blocks: list[str] = []
    for i, claim in enumerate(claims):
        evidence = evidence_map.get(i, [])
        clean_evidence = [_TAG_RE.sub('', e) for e in evidence]
        if clean_evidence:
            ev_block = "\n".join(f"  - {e}" for e in clean_evidence)
        else:
            ev_block = "  - No relevant evidence available"
        blocks.append(f'Claim {i + 1}: "{claim}"\nEvidence:\n{ev_block}')

    claims_and_evidence_block = "\n\n".join(blocks)
    return NLI_BATCH_PROMPT_TEMPLATE.replace(
        "{claims_and_evidence_block}", claims_and_evidence_block
    )


# ── Batch NLI Entailment Check ───────────────────────────────────────────


def nli_check_triples_batch(
    triples: list[Triple],
    spatial_facts: list[str],
    visual_description: str,
    hard_facts: list[str],
    scene_graph: list[dict] | None = None,
) -> list[NLIResult]:
    """Run batched NLI entailment check for all triples against KB evidence.

    Collects evidence per triple, formats a single batch prompt, makes
    at most one LLM call. Returns all NEUTRAL if no evidence or LLM fails.
    """
    # Step 1: Collect evidence per triple
    evidence_map: dict[int, list[str]] = {}
    for i, triple in enumerate(triples):
        ev = collect_nli_evidence(
            triple.subject, triple.object,
            spatial_facts, visual_description, hard_facts,
            scene_graph=scene_graph,
        )
        if ev:
            evidence_map[i] = ev

    # Step 2: No evidence → all NEUTRAL, no LLM call
    if not evidence_map:
        return [
            NLIResult(
                verdict=NLIVerdict.NEUTRAL,
                evidence_used=[],
                evidence_source=EvidenceSource.VISUAL_DESCRIPTION,
                raw_response=None,
            )
            for _ in triples
        ]

    # Step 3: Format batch prompt and make ONE LLM call
    claims = [f"{t.subject} {t.relation} {t.object}" for t in triples]
    prompt = format_batch_nli_prompt(claims, evidence_map)
    raw = llm_call(
        [{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.0,
    )

    # Step 4: LLM failure → graceful degradation
    if raw is None:
        log.warning("NLI batch LLM call failed — returning all NEUTRAL")
        return [
            NLIResult(
                verdict=NLIVerdict.NEUTRAL,
                evidence_used=evidence_map.get(i, []),
                evidence_source=(classify_evidence_source(evidence_map[i])
                                 if i in evidence_map
                                 else EvidenceSource.VISUAL_DESCRIPTION),
                raw_response=None,
            )
            for i in range(len(triples))
        ]

    # Step 5: Parse and build results
    verdicts = parse_batch_nli_response(raw, len(triples))
    results: list[NLIResult] = []
    for i, verdict in enumerate(verdicts):
        ev = evidence_map.get(i, [])
        source = classify_evidence_source(ev) if ev else EvidenceSource.VISUAL_DESCRIPTION
        results.append(NLIResult(
            verdict=verdict,
            evidence_used=ev,
            evidence_source=source,
            raw_response=raw,
        ))
    return results


# ── Single-Triple Fallback ───────────────────────────────────────────────


def nli_check_triple(
    subj: str,
    rel: str,
    obj: str,
    spatial_facts: list[str],
    visual_description: str,
    hard_facts: list[str],
    scene_graph: list[dict] | None = None,
) -> NLIResult:
    """Run NLI entailment check for a single triple against KB evidence.

    Retained as fallback. Returns NEUTRAL if no evidence or LLM fails.
    """
    evidence = collect_nli_evidence(
        subj, obj, spatial_facts, visual_description, hard_facts,
        scene_graph=scene_graph,
    )

    if not evidence:
        return NLIResult(
            verdict=NLIVerdict.NEUTRAL,
            evidence_used=[],
            evidence_source=EvidenceSource.VISUAL_DESCRIPTION,
            raw_response=None,
        )

    claim = f"{subj} {rel} {obj}"
    prompt = format_batch_nli_prompt([claim], {0: evidence})
    raw = llm_call(
        [{"role": "user", "content": prompt}],
        max_tokens=50,
        temperature=0.0,
    )

    if raw is None:
        return NLIResult(
            verdict=NLIVerdict.NEUTRAL,
            evidence_used=evidence,
            evidence_source=classify_evidence_source(evidence),
            raw_response=None,
        )

    verdict = parse_nli_response(raw)
    return NLIResult(
        verdict=verdict,
        evidence_used=evidence,
        evidence_source=classify_evidence_source(evidence),
        raw_response=raw,
    )
