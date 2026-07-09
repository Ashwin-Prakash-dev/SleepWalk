"""Ingestion pipeline for the Enceladus knowledge graph.

ingest_text() turns a chunk of raw text into a graph fragment:
extract -> embed -> resolve entity -> insert node -> structural edges ->
semantic edges -> inference nodes/edges.

ingest_from_newsapi() pulls articles from NewsAPI and feeds each through
ingest_text().

Embeddings use OpenAI text-embedding-3-small (1536 dims) via embeddings.py.
Credentials come from the environment (see .env): OPENAI_API_KEY, NEWSAPI_KEY,
plus whatever llm_service.py and db.py need.
"""
from __future__ import annotations

import math
import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import requests
from dotenv import load_dotenv

import db
from embeddings import embed
from llm_service import (
    check_soundness,
    classify_evidence,
    classify_topic_parent,
    debate_adjudicate,
    debate_object,
    debate_rebut,
    enumerate_alternatives,
    extract_node,
    reason_pair,
    resolve_entity_coreference,
)

load_dotenv()

NEWSAPI_URL = "https://newsapi.org/v2/everything"
# Cosine threshold for semantic edges + inference context. Calibrated for the
# local all-MiniLM-L6-v2 model, whose related-text scores (~0.45-0.65) run lower
# than OpenAI's (~0.8+). Raise toward 0.75 if you switch back to OpenAI embeddings.
SIMILARITY_THRESHOLD = 0.45
STRUCTURAL_EDGE_LIMIT = 10
INFERENCE_CONTEXT_SIZE = 15

# ENTITY_MATCH_THRESHOLD kept for backwards compatibility (imported by test_entities.py).
# The main resolution path no longer uses it directly — see ENTITY_CANDIDATE_THRESHOLD.
ENTITY_MATCH_THRESHOLD = 0.85

# Two-stage entity resolution knobs:
#   ENTITY_CANDIDATE_THRESHOLD — wide-net cosine cutoff for candidate generation.
#     Lower than ENTITY_MATCH_THRESHOLD so surface forms like "Russian forces"
#     are surfaced as candidates for "Russia" (they sit ~0.55-0.70 with MiniLM).
#   ENTITY_CANDIDATE_K — max candidates handed to the LLM coreference step.
ENTITY_CANDIDATE_THRESHOLD = 0.50
ENTITY_CANDIDATE_K = 5

# Topic resolution: slightly lower than entity threshold because topic surface
# forms cluster more broadly ("nuclear talks" / "nuclear negotiations" should
# merge; "energy" / "military conflict" should not).
TOPIC_MATCH_THRESHOLD = 0.82

# Max levels to climb when placing a brand-new topic into the hierarchy. Bounds
# the LLM calls spent per never-before-seen topic and backstops runaway chains.
TOPIC_HIERARCHY_MAX_DEPTH = 4

# --- batched, adversarially-verified inference knobs -------------------------
# All tunable; defaults are conservative because every stage is multi-LLM-call
# and the free Groq tier caps daily tokens (Gemini failover absorbs overflow).
INFERENCE_BATCH_SIZE        = 10    # unprocessed raw_input nodes that auto-trigger a pass
PAIR_CANDIDATE_K            = 3     # partners kept per new node
MAX_PAIRS_PER_BATCH         = 15    # hard cap on Pass-1 reasoning calls
DUPLICATE_THRESHOLD         = 0.95  # cosine >= this => restatement, not an inference; skip
MIN_PAIR_SIMILARITY         = 0.45  # cosine floor to pair / retrieve at all
MAX_ALTERNATIVES            = 3     # competing explanations enumerated per inference
EVIDENCE_RETRIEVAL_K        = 8     # semantic hits pulled per evidence signature
EVIDENCE_CLASSIFY_CAP       = 20    # max nodes sent to the classifier prompt
COVERAGE_SATURATION         = 12    # corpus node count at which coverage -> 1.0
COVERAGE_CORROBORATION_MIN  = 0.5   # coverage below this can never reach 'corroborated'
TIME_WINDOW_DAYS            = 30    # structural pull + coverage window around source dates
HIGH_REPORTABILITY          = 0.6   # reportability >= this => silence about it is informative
UNVERIFIED_CONFIDENCE_CAP   = 0.55  # confidence ceiling when status = unverified
DEFEATER_PENALTY            = 0.5   # confidence multiplier when status = contested
CORROBORATION_BONUS         = 0.25  # max additive bonus, scaled by coverage * mean reportability
CONVERGENCE_THRESHOLD       = 0.82  # cosine for "the same inference" (calibrated for MiniLM scale)
CONVERGENCE_BONUS           = 0.15  # additive per independent re-derivation (still coverage-gated)
INDEPENDENCE_MAX_OVERLAP    = 0.34  # source-set Jaccard above which a rederivation is NOT independent
# Convergence is a form of corroboration: by default it only strengthens an already
# 'corroborated' verdict, and only on the strength of OTHER 'corroborated' inferences.
# ENCELADUS_CONVERGENCE_LEGACY=1 restores the old (buggy) unconditional bonus.
CONVERGENCE_LEGACY = os.environ.get("ENCELADUS_CONVERGENCE_LEGACY", "0") == "1"

# --- forward-chaining (derived nodes as premises) ----------------------------
# A corroborated inference may itself become a PREMISE for new reasoning, enabling
# multi-hop derivation. Guarded hard — see [[inference-forward-chaining-design]].
PROMOTE_THRESHOLD       = 0.7   # min confidence + 'corroborated' status to be reusable as a premise
MAX_DERIVATION_DEPTH    = 3     # cap on derived-node depth (raw=0)

# Phase 2: iterative deepening. After a pass creates inferences, the trustworthy
# (corroborated, >=PROMOTE_THRESHOLD, still below the depth cap) ones are fed back
# in as premises for another pass, until none qualify (fixpoint) or the budget runs
# out. Termination is triple-guarded: fixpoint, the depth cap, and the pair budget.
MAX_PAIRS_PER_LEVEL     = 15    # pairs reasoned per deepening pass
DEEPEN_GLOBAL_PAIR_CAP  = 45    # hard cap on pairs across all passes of one batch
MAX_DEEPEN_PASSES       = 4     # belt-and-suspenders cap on pass count

# Redundancy control (flag-gated). When on, a freshly-reasoned conclusion that is a
# near-duplicate (cosine >= MERGE_THRESHOLD) of an existing inference AND rests on
# overlapping raw grounding (Jaccard > INDEPENDENCE_MAX_OVERLAP — i.e. NOT an
# independent re-derivation) is dropped before verification, so deepening spends its
# budget on novel conclusions instead of rephrasings. Independent re-derivations
# still pass through and earn convergence. See [[inference-forward-chaining-design]].
DEDUP_INFERENCES = os.environ.get("ENCELADUS_DEDUP_INFERENCES", "0") == "1"
MERGE_THRESHOLD  = 0.93         # "the same conclusion restated" (> CONVERGENCE_THRESHOLD)

# --- verification experiments (flag-gated; baseline = all defaults) ----------
# Every flag here defaults to the original behavior so the baseline stays exactly
# reproducible; eval/run_eval.py flips them to measure each change against labels.
# Local models only — see rerank.py / nli.py (lazy-loaded, cached, no API cost).
#
# Phase 1: cross-encoder reranking of the evidence pool before the classify cap.
RERANK_EVIDENCE      = os.environ.get("ENCELADUS_RERANK_EVIDENCE", "0") == "1"
EVIDENCE_RERANK_POOL = 40      # wider pre-rerank pool; post-rerank still EVIDENCE_CLASSIFY_CAP
#
# Phase 2: evidence classifier path — "llm" (current), "nli", or "both" (agree-or-abstain).
USE_NLI_CLASSIFIER   = os.environ.get("ENCELADUS_NLI_CLASSIFIER", "llm")
NLI_ENTAIL_THRESHOLD = 0.6     # P(entailment) >= this => supports_inference
NLI_CONTRA_THRESHOLD = 0.6     # P(contradiction) >= this => supports_alternative (defeater)
#
# Phase 1: weight coverage/corroboration by source reliability (see sources.py).
USE_SOURCE_WEIGHTS = os.environ.get("ENCELADUS_SOURCE_WEIGHTS", "0") == "1"
#
# Phase 2: when chaining off a derived premise, pair it with DISTINCT events
# (not its own sources / near-restatements) so deepening synthesises rather than
# restates. See [[inference-forward-chaining-design]].
SYNTH_DISTINCT_PAIRING = os.environ.get("ENCELADUS_SYNTH_DISTINCT", "0") == "1"
SYNTH_MAX_PARTNER_SIM  = 0.90  # when distinct-pairing a derived premise, skip partners above this
#
# Defeater policy: "strict" (default, baseline) contests on ANY defeater; "weighted"
# contests only when defeaters aren't dominated by supports (one stray misclassified
# node no longer flips a well-supported claim).
DEFEATER_POLICY        = os.environ.get("ENCELADUS_DEFEATER_POLICY", "strict")
DEFEATER_SUPPORT_RATIO = 0.5   # weighted: contest iff len(defeaters) >= max(1, ceil(RATIO*len(supports)))

# --- Phase 3: belief revision (living model) ----------------------------------
# The graph revises instead of only accumulating: a newer report can SUPERSEDE an
# older one (edge newer->older; overtaken nodes stop corroborating and count less
# toward coverage), and new evidence RE-OPENS existing verdicts (budgeted re-run of
# Pass 2, recorded in inference_meta.revised_at). See [[project-north-star]].
BELIEF_REVISION  = os.environ.get("ENCELADUS_BELIEF_REVISION", "1") == "1"
SUPERSESSION     = os.environ.get("ENCELADUS_SUPERSESSION", "1") == "1"
SUPERSEDE_SIMILARITY       = 0.75  # cosine floor for "same story, updated" (+ same actor + later date)
SUPERSEDE_MAX_PER_NODE     = 3
SUPERSEDED_COVERAGE_WEIGHT = 0.3   # overtaken nodes count this much toward coverage
REVERIFY_SIMILARITY        = 0.50  # a new node at least this close to an inference re-opens it
REVERIFY_MAX_PER_BATCH     = 6     # LLM budget: re-verifications per batch
#
# Scope the indiscriminate time-window evidence pull to nodes sharing an actor or
# entity with the premises — the unscoped pull floods verification with same-era
# but unrelated stories (the cross-scenario defeater bleed seen in the diagnostic).
SCOPED_RETRIEVAL = os.environ.get("ENCELADUS_SCOPED_RETRIEVAL", "1") == "1"
#
# --- Graded, calibrated verdicts (measured 2026-07-06, n=81 benchmark) --------
# The binary coverage>=0.5 cliff was measured as THE recall wall (23 of 27 true-
# unverified rows failed only that condition), and the fixed caps are measurably
# under-confident (rows capped at 0.55 were empirically ~78-83% true). Graded mode
# (a) admits corroboration on strong direct support (>= GRADED_MIN_SUPPORTS) even
# under thin coverage — confidence stays coverage-ceilinged, so thin corpus still
# means modest confidence; (b) re-anchors the caps to the measured curve.
# Default ON since 2026-07-06: measured on the n=81 benchmark as precision 1.000
# (unchanged), recall 0.278 -> 0.611, ECE 0.151 -> 0.115. Set =0 for the legacy
# binary-gate baseline.
GRADED_VERDICT  = os.environ.get("ENCELADUS_GRADED_VERDICT", "1") == "1"
GRADED_MIN_SUPPORTS   = 2     # supports needed to corroborate below the coverage floor
GRADED_UNVERIFIED_CAP = 0.70  # measured ~0.78-0.83 true; discounted for label noise
GRADED_CEILING_FLOOR  = 0.55  # graded coverage ceiling: 0.55 + 0.40*coverage
GRADED_CEILING_SLOPE  = 0.40
#
# Debate escalation tier: when a verdict is BORDERLINE (support and defeaters both
# present — the measured true-but-contested failure shape), run a structured
# evidential debate over the defeaters: opponent states each objection from the
# cited evidence, the proposer rebuts or concedes, a mediator sustains/overrules.
# Only SUSTAINED objections remain defeaters; the deterministic arithmetic still
# sets status/confidence. ~3 extra LLM calls per debated inference. Fail-safe: any
# LLM failure keeps all defeaters (baseline behavior).
# Default ON since 2026-07-07: measured on the n=70 benchmark vs same-instance
# baseline as P 0.950->0.955, R 0.452->0.500, ECE 0.191->0.116; freed 8/13 true
# rows from spurious 'contested', added zero false corroborations. Set =0 to
# disable the escalation (baseline single-pass classification).
DEBATE_VERDICTS = os.environ.get("ENCELADUS_DEBATE", "1") == "1"
#
# Pass 1.5 soundness gate: a premises-only logic check on each freshly-reasoned
# conclusion, run before the expensive Pass 2. Demotes conclusions that don't
# follow from their premises (reversed causation, an unpinned date/quantity, an
# effect-date treated as a start, overclaimed certainty) to 'unverified'. Needs no
# world knowledge, so it works on post-cutoff events. Targets the specific-value
# overreach false-positive class (benchmark scenario H). Default off pending measurement.
SOUNDNESS_GATE = os.environ.get("ENCELADUS_SOUNDNESS_GATE", "0") == "1"
#
# Count coverage by shared ENTITIES instead of exact actor/subject strings —
# "Sandar's government" vs "Sandar's grid operator" don't string-match, so dense
# scenarios read as thin and the coverage gate starves.
# MEASURED REGRESSION as-is (2026-07-06): any-shared-entity matching inflates
# coverage on topically-entangled corpora — precision 1.000 -> 0.943, ECE 0.115 ->
# 0.155. Stays default-OFF until matching is made discriminative (e.g. require the
# PRIMARY actor entity, not any mention).
ENTITY_COVERAGE = os.environ.get("ENCELADUS_ENTITY_COVERAGE", "0") == "1"

# Module-level embedding cache, keyed by node id (content embeddings are stable).
_emb_cache: dict[str, list[float]] = {}


# --- small helpers -----------------------------------------------------------
def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _pg_quote(value: str) -> str:
    """Escape a value for embedding inside a PostgREST filter string."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _resolve_entity(actor: str, context: str = "") -> str:
    """Return the entity id for `actor`, creating the entity if needed.

    Two-stage pipeline:
      1. Exact (case-insensitive) name or alias match — no embedding, no LLM.
      2a. Embedding KNN (ENTITY_CANDIDATE_THRESHOLD, top ENTITY_CANDIDATE_K) narrows
          the entity table to a small candidate set.
      2b. LLM coreference decision over that set: same real-world referent → merge
          and record alias; distinct entity → fall through.
      3. Create a new entity with its name embedding for future candidate generation.

    `context` is the source sentence the actor was extracted from; it's passed to
    the LLM in step 2b to distinguish coreference from mere relatedness.
    """
    safe = _pg_quote(actor)
    res = (
        db.client()
        .table("entities")
        .select("id")
        .or_(f'name.ilike."{safe}",aliases.cs.{{"{safe}"}}')
        .limit(1)
        .execute()
    )
    if res.data:
        return res.data[0]["id"]

    actor_embedding = embed(actor)
    candidates = db.match_entities(
        actor_embedding,
        match_threshold=ENTITY_CANDIDATE_THRESHOLD,
        match_count=ENTITY_CANDIDATE_K,
    )
    if candidates:
        idx = resolve_entity_coreference(actor, context, candidates)
        if idx is not None and 0 <= idx < len(candidates):
            entity_id = candidates[idx]["id"]
            db.add_entity_alias(entity_id, actor)
            return entity_id

    return db.insert_entity(name=actor, aliases=[actor], embedding=actor_embedding)["id"]


def _structural_edges(new_id: str, column: str, value: Optional[str], edge_type: str) -> None:
    """Link the new node to the most recent nodes sharing `column == value`."""
    if not value:
        return
    res = (
        db.client()
        .table("nodes")
        .select("id")
        .eq(column, value)
        .neq("id", new_id)
        .order("created_at", desc=True)
        .limit(STRUCTURAL_EDGE_LIMIT)
        .execute()
    )
    for row in res.data:
        db.insert_edge(new_id, row["id"], edge_type)


def _edge_exists(a: str, b: str, edge_type: str) -> bool:
    """True if an edge of `edge_type` already links a and b (either direction)."""
    res = (
        db.client()
        .table("edges")
        .select("id")
        .eq("edge_type", edge_type)
        .or_(f"and(source_id.eq.{a},target_id.eq.{b}),and(source_id.eq.{b},target_id.eq.{a})")
        .limit(1)
        .execute()
    )
    return bool(res.data)


def _resolve_topic(name: str, depth: int = 0) -> str:
    """Return the topic id for `name`, creating + placing it in the DAG if needed.

    Mirrors _resolve_entity: exact/alias match → embedding similarity → create.
    A *newly created* topic is then attached to the hierarchy (climbed toward a
    root); matched/existing topics are assumed already placed, so no LLM cost is
    paid on the hot ingest path for topics we've seen before. `depth` bounds the
    upward climb when this call is itself resolving a parent.
    """
    safe = _pg_quote(name)
    res = (
        db.client()
        .table("topics")
        .select("id")
        .or_(f'name.ilike."{safe}",aliases.cs.{{"{safe}"}}')
        .limit(1)
        .execute()
    )
    if res.data:
        return res.data[0]["id"]

    topic_embedding = embed(name)
    matches = db.match_topics(topic_embedding, match_threshold=TOPIC_MATCH_THRESHOLD, match_count=1)
    if matches:
        topic_id = matches[0]["id"]
        db.add_topic_alias(topic_id, name)
        return topic_id

    topic_id = db.insert_topic(name=name, aliases=[name], embedding=topic_embedding)["id"]
    _attach_to_hierarchy(topic_id, name, depth)
    return topic_id


def _attach_to_hierarchy(topic_id: str, name: str, depth: int = 0) -> None:
    """Place a freshly-created topic under a broader parent, climbing toward a root.

    Asks the LLM for the broader domain `name` is a kind of (biased to existing
    roots for vocabulary convergence), resolves that parent through _resolve_topic
    — which may itself create and further climb it — and records a child→parent edge.

    Guards: a depth cap bounds the climb; a topic the LLM judges top-level is
    marked is_root and stops; and the edge is skipped if the proposed parent is
    already a descendant of this topic (which would close a cycle).
    """
    if depth >= TOPIC_HIERARCHY_MAX_DEPTH:
        return
    roots = [r["name"] for r in db.list_root_topics()]
    parent_name = classify_topic_parent(name, roots)
    if not parent_name:
        db.set_topic_root(topic_id)            # already a top-level domain
        return

    parent_id = _resolve_topic(parent_name, depth + 1)
    if parent_id == topic_id:
        db.set_topic_root(topic_id)
        return
    if parent_id in db.topic_descendant_ids(topic_id):
        return                                 # would close a cycle; leave unlinked
    db.insert_topic_relation(child_id=topic_id, parent_id=parent_id)


def backfill_topic_hierarchy() -> dict:
    """Place existing topics that predate the hierarchy (no parents, not a root).

    One-shot maintenance for corpora seeded before topic_relations existed; safe
    to re-run (already-placed topics are skipped).
    """
    topics = db.client().table("topics").select("id,name,is_root").execute().data or []
    placed = 0
    for t in topics:
        if t.get("is_root") or db.topic_parents(t["id"]):
            continue
        _attach_to_hierarchy(t["id"], t["name"], 0)
        placed += 1
    return {"examined": len(topics), "placed": placed}


def _link_topics(node_id: str, domains: Optional[list]) -> None:
    """Resolve each domain string to a topic row and link it to the node."""
    for name in domains or []:
        if not name:
            continue
        topic_id = _resolve_topic(name)
        db.insert_node_topic(node_id, topic_id)


def _link_entities(
    node_id: str,
    primary_entity_id: Optional[str],
    extracted_entities: Optional[list[dict]],
    context: str = "",
) -> None:
    """Populate node_entities for a node (additive; nodes.entity_id is untouched).

    Links the primary actor (role 'actor', via the already-resolved
    `primary_entity_id`) plus every entity from the extraction's `entities` list,
    each resolved through the shared `_resolve_entity` coreference logic. Deduped
    per entity, with the primary actor winning so it always keeps role 'actor'.
    `context` is the source sentence forwarded to the coreference LLM.
    """
    roles: dict[str, str] = {}
    if primary_entity_id:
        roles[primary_entity_id] = "actor"
    for item in extracted_entities or []:
        name = (item or {}).get("name")
        if not name:
            continue
        role = (item or {}).get("role") or "mentioned"
        # Compound surface forms ("US-Iran", "Japan/Korea") link their PARTS when
        # every part is already a known entity — a "US-Iran" pseudo-entity poisons
        # streams and coverage. Unknown-part compounds (e.g. "Coca-Cola") stay intact.
        part_ids = _compound_entity_ids(name)
        if part_ids:
            for pid in part_ids:
                roles.setdefault(pid, role)
            continue
        entity_id = _resolve_entity(name, context)
        roles.setdefault(entity_id, role)
    for entity_id, role in roles.items():
        db.insert_node_entity(node_id, entity_id, role)


def _compound_entity_ids(name: str) -> Optional[list[str]]:
    """Entity ids for the parts of a compound name, or None if it isn't one.

    A name splits only on -, –, or / into 2+ parts, and counts as a compound only
    when EVERY part already resolves to an existing entity by exact name/alias.
    """
    parts = [p.strip() for p in re.split(r"[-–/]", name) if p.strip()]
    if len(parts) < 2:
        return None
    ids: list[str] = []
    for part in parts:
        row = db.find_entity(part)
        if row is None:
            return None
        ids.append(row["id"])
    return ids


# Runtime guards: flip false (with one warning) if the live DB schema predates the
# 'supersedes' edge type / inference_meta.revised_at — apply schema.sql to enable.
_supersession_ok = True
_revision_ok = True


def _apply_supersession(new_id: str, actor: Optional[str], event_date: str, similar: list[dict]) -> None:
    """Record supersedes edges from this node to older reports it overtakes.

    Heuristic (no LLM): a candidate from the ingest similarity search is overtaken
    if it is a raw_input node with the SAME actor, cosine >= SUPERSEDE_SIMILARITY,
    and a strictly EARLIER event_date. Downstream, superseded nodes stop counting
    as corroboration and count less toward coverage.
    """
    global _supersession_ok
    if not _supersession_ok:
        return
    new_dt = _parse_ts(event_date)
    if not new_dt or not actor:
        return
    cand = [m for m in similar if _to_float(m.get("similarity"), 0.0) >= SUPERSEDE_SIMILARITY]
    if not cand:
        return
    rows = {r["id"]: r for r in db.nodes_by_ids([m["id"] for m in cand])}
    made = 0
    for m in cand:
        row = rows.get(m["id"])
        if not row or row.get("node_category") != "raw_input":
            continue
        if (row.get("actor") or "").strip().lower() != actor.strip().lower():
            continue
        old_dt = _parse_ts(row.get("event_date"))
        if not old_dt or old_dt >= new_dt:
            continue
        try:
            if not _edge_exists(new_id, row["id"], "supersedes"):
                db.insert_edge(new_id, row["id"], "supersedes")
        except Exception as exc:  # DB CHECK predates 'supersedes' — degrade loudly, once
            _supersession_ok = False
            print(f"[warn] supersedes edge rejected ({exc}); supersession disabled "
                  f"for this run — apply schema.sql to enable belief revision")
            return
        made += 1
        if made >= SUPERSEDE_MAX_PER_NODE:
            break


# --- pipeline ----------------------------------------------------------------
def ingest_text(
    text: str, source_url: str = None, event_date: str = None, source_weight: float = None
) -> str:
    """Run the full ingestion pipeline on one piece of text.

    `event_date` (ISO string, optional) records when the event actually occurred —
    e.g. a news article's publishedAt — so coverage / time-window / convergence
    reason over real dates instead of ingestion time. Defaults to None (unchanged).

    `source_weight` (optional [0,1]) records source reliability; only persisted when
    provided (source weighting is opt-in, see USE_SOURCE_WEIGHTS), so the baseline
    path neither writes nor requires the column.

    Returns the id of the inserted raw-input node.
    """
    # Step 1 — structured extraction.
    node = extract_node(text, source_url)
    actor = node.get("actor") or None
    subject = node.get("subject") or None

    # Step 2 — embed the core content.
    embedding = embed(node["content"])

    # Step 3 — entity resolution (two-stage: KNN candidates → LLM coreference).
    entity_id = _resolve_entity(actor, text) if actor else None

    # Step 4 — insert the raw-input node.
    # Normalise node_kind: the LLM occasionally invents values outside the CHECK
    # constraint (e.g. "warning", "report"). Fall back to "claim" rather than crash.
    node_kind = node.get("node_kind")
    if node_kind not in db.NODE_KINDS:
        node_kind = "claim"

    new_row = db.insert_node(
        node_category="raw_input",
        node_kind=node_kind,
        content=node["content"],
        actor=actor,
        entity_id=entity_id,
        subject=subject,
        confidence=_to_float(node.get("confidence"), 0.8),
        source_url=source_url,
        event_date=event_date,
        expires_at=node.get("expires_at"),
        source_weight=source_weight,
        embedding=embedding,
    )
    new_id = new_row["id"]

    # Step 4b — multi-entity links (additive). The primary actor (entity_id) and
    # every extracted entity land in node_entities; nodes.entity_id is unchanged.
    _link_entities(new_id, entity_id, node.get("entities"), text)

    # Step 4c — domain links. Fall back to [subject] if the LLM didn't return domains.
    domains = node.get("domains") or ([subject] if subject else [])
    _link_topics(new_id, domains)

    # Step 5 — structural edges (same actor / same subject).
    _structural_edges(new_id, "actor", actor, "same_actor")
    _structural_edges(new_id, "subject", subject, "same_subject")

    # Step 6 + 7 share the same similarity search.
    matches = db.match_nodes(embedding, match_threshold=SIMILARITY_THRESHOLD,
                             match_count=INFERENCE_CONTEXT_SIZE + 1)
    similar = [m for m in matches if m["id"] != new_id]

    # Step 6 — semantic edges.
    for m in similar:
        sim = _to_float(m.get("similarity"), 0.0)
        if sim > SIMILARITY_THRESHOLD and not _edge_exists(new_id, m["id"], "semantically_similar"):
            db.insert_edge(new_id, m["id"], "semantically_similar", weight=sim)

    # Step 6b — supersession (belief revision): a dated report can overtake earlier
    # reports of the same story (same actor, highly similar, strictly earlier date).
    if SUPERSESSION and event_date:
        _apply_supersession(new_id, actor, event_date, similar)

    # Step 7 — inference is no longer inline. The node is left
    # inference_processed=false; once INFERENCE_BATCH_SIZE such nodes accumulate,
    # the batched engine pairs and verifies them.
    maybe_run_inference_batch()

    return new_id


# --- batched, adversarially-verified inference engine ------------------------
def maybe_run_inference_batch() -> Optional[dict]:
    """Auto-trigger a batch once enough unprocessed nodes have accumulated."""
    if db.nodes_unprocessed_count() >= INFERENCE_BATCH_SIZE:
        return run_inference_batch()
    return None


def run_inference_batch(force: bool = False) -> dict:
    """Pair accumulated raw_input nodes, reason, verify, and persist inferences,
    then iteratively deepen: trustworthy new conclusions become premises for
    further passes until a fixpoint / depth cap / pair budget is reached.

    Returns a small summary dict. With force=True a partial batch (below
    INFERENCE_BATCH_SIZE) is processed anyway — used by the CLI / endpoint flush.
    """
    count = db.nodes_unprocessed_count()
    if count == 0 or (count < INFERENCE_BATCH_SIZE and not force):
        return {"pairs": 0, "inferences": 0, "processed": 0, "skipped": count}

    # TODO(out-of-scope): temporal re-opening — when new raw_input arrives in the
    # time window of an existing 'corroborated'/'unverified' verdict, re-run its
    # verification so stale conclusions can be contested by later evidence. Would
    # hook in here, selecting affected prior inferences before processing new nodes.
    new_nodes = db.fetch_unprocessed_nodes(limit=200)
    raw_ids = [n["id"] for n in new_nodes]

    # Iterative deepening: each pass pairs its `frontier`, persists inferences, and
    # promotes the trustworthy new ones into the next pass's frontier. Stops at a
    # fixpoint (nothing promotable), the depth cap, or the global pair budget.
    frontier = new_nodes
    seen_ids = set(raw_ids)
    all_created: set[str] = set()
    total_pairs = total_created = deduped = passes = 0
    budget = DEEPEN_GLOBAL_PAIR_CAP

    while frontier and budget > 0 and passes < MAX_DEEPEN_PASSES:
        passes += 1
        pairs = _candidate_pairs(frontier)[: min(MAX_PAIRS_PER_LEVEL, budget)]
        budget -= len(pairs)
        total_pairs += len(pairs)

        created_nodes: list[dict] = []
        for node_a, node_b, _sim in pairs:
            # Force a non-restatement conclusion when chaining off a derived premise.
            chaining = node_a.get("node_category") == "inference" or node_b.get("node_category") == "inference"
            inf = reason_pair(node_a, node_b, require_novel=SYNTH_DISTINCT_PAIRING and chaining)
            if not inf or not inf.get("content"):
                continue
            # Drop redundant restatements BEFORE the costly verification step.
            if DEDUP_INFERENCES and _find_duplicate(
                embed(inf["content"]), _derivation_roots(node_a, node_b)
            ):
                deduped += 1
                continue
            base_conf = _propagate_confidence(_to_float(inf.get("confidence"), 0.6), node_a, node_b)
            verdict = _verify_inference(inf, node_a, node_b, base_conf)
            row = _persist_inference(inf, node_a, node_b, base_conf, verdict)
            created_nodes.append(row)
            all_created.add(row["id"])
            total_created += 1

        # Promote only conclusions trustworthy enough to be premises and still able
        # to go deeper; dedupe against everything already used as a frontier node.
        frontier = [
            c for c in created_nodes
            if c.get("status") == "corroborated"
            and _to_float(c.get("confidence"), 0.0) >= PROMOTE_THRESHOLD
            and _node_depth(c) < MAX_DERIVATION_DEPTH
            and c["id"] not in seen_ids
        ]
        seen_ids.update(c["id"] for c in frontier)

    # Phase 3 — belief revision: the new evidence may bear on EXISTING verdicts;
    # re-open and re-verify the most affected ones (budgeted).
    revised = _reverify_affected(new_nodes, exclude_ids=all_created) if BELIEF_REVISION else 0

    db.mark_nodes_processed(raw_ids)
    return {
        "pairs": total_pairs,
        "inferences": total_created,
        "deduped": deduped,
        "revised": revised,
        "processed": len(raw_ids),
        "passes": passes,
    }


# --- small math / time helpers ----------------------------------------------
def _node_embedding(node: dict) -> list[float]:
    """Embedding for a node's content, cached by node id."""
    nid = node.get("id")
    if nid and nid in _emb_cache:
        return _emb_cache[nid]
    vec = embed(node.get("content", ""))
    if nid:
        _emb_cache[nid] = vec
    return vec


def _cosine(a: list[float], b: list[float]) -> float:
    """Dot product of two unit-normalised embeddings == cosine similarity."""
    return sum(x * y for x, y in zip(a, b))


def _parse_ts(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        s = value.replace("Z", "+00:00") if isinstance(value, str) else value
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _time_window(node_a: dict, node_b: dict) -> tuple[str, str]:
    """ISO [start, end] spanning the two source dates +/- TIME_WINDOW_DAYS."""
    dates = [
        _parse_ts(n.get("event_date") or n.get("created_at")) for n in (node_a, node_b)
    ]
    dates = [d for d in dates if d]
    span = timedelta(days=TIME_WINDOW_DAYS)
    if not dates:
        now = datetime.now(timezone.utc)
        return (now - span).isoformat(), now.isoformat()
    return (min(dates) - span).isoformat(), (max(dates) + span).isoformat()


def _entity_ids(node_id: str) -> list[str]:
    rows = (
        db.client().table("node_entities").select("entity_id")
        .eq("node_id", node_id).execute().data or []
    )
    return [r["entity_id"] for r in rows]


def _topic_ids(node_id: str) -> list[str]:
    rows = (
        db.client().table("node_topics").select("topic_id")
        .eq("node_id", node_id).execute().data or []
    )
    return [r["topic_id"] for r in rows]


def _node_depth(node: dict) -> int:
    """Derivation depth of a node (raw_input = 0); tolerates a missing column."""
    try:
        return int(node.get("depth") or 0)
    except (TypeError, ValueError):
        return 0


def _is_valid_premise(partner: dict, node_a: dict, statuses: dict[str, str]) -> bool:
    """Whether `partner` may pair with `node_a` as a premise.

    raw_input partners are always eligible (subject to the same-source guard). An
    inference partner is eligible only if it is a verified, high-confidence,
    not-too-deep prior conclusion — see [[inference-forward-chaining-design]].
    Verification (Pass 2) stays raw-only, so derived nodes feed reasoning as
    premises but never count as corroborating evidence.
    """
    if partner.get("source_url") and partner["source_url"] == node_a.get("source_url"):
        return False  # same article => not independent
    category = partner.get("node_category")
    if category == "raw_input":
        return True
    if category == "inference":
        if statuses.get(partner["id"]) != "corroborated":
            return False
        if _to_float(partner.get("confidence"), 0.0) < PROMOTE_THRESHOLD:
            return False
        # A premise at depth d yields a conclusion at depth d+1; keep it <= cap.
        return _node_depth(partner) < MAX_DERIVATION_DEPTH
    return False


def _propagate_confidence(base_conf: float, node_a: dict, node_b: dict) -> float:
    """Cap a conclusion's confidence by its weakest DERIVED premise (min-chain).

    Chaining can't manufacture certainty: a conclusion built on a prior inference
    is no more confident than that inference. raw_input premises don't cap here —
    their credibility is already handled by the coverage/verification machinery.
    """
    derived = [
        _to_float(n.get("confidence"), 1.0)
        for n in (node_a, node_b)
        if n.get("node_category") == "inference"
    ]
    return min([base_conf, *derived]) if derived else base_conf


def _derivation_roots(node_a: dict, node_b: dict) -> list[str]:
    """Transitive raw_input grounding of a pair: each raw premise contributes
    itself, each derived premise contributes its own raw ancestors."""
    roots: set[str] = set()
    for n in (node_a, node_b):
        if n.get("node_category") == "inference":
            roots.update(db.derivation_roots(n["id"]))
        else:
            roots.add(n["id"])
    return list(roots)


def _independence_capped_confidence(
    conf: float, verdict: dict, node_a: dict, node_b: dict, convergent: list[str]
) -> float:
    """Enforce min-chain monotonicity, with an independence exception.

    A derived conclusion may exceed its weakest DERIVED premise only when the lift
    is earned by evidence INDEPENDENT of that premise's own grounding: raw support
    that isn't already an ancestor of the premise, or a convergent inference whose
    roots barely overlap the premise's. Absent that, the confidence is hard-capped
    at the weakest derived premise — chaining alone can't manufacture certainty.
    """
    derived = [n for n in (node_a, node_b) if n.get("node_category") == "inference"]
    if not derived:
        return conf  # no chain => leave the existing machinery untouched
    premise_cap = min(_to_float(n.get("confidence"), 1.0) for n in derived)
    if conf <= premise_cap:
        return conf

    premise_roots: set[str] = set()
    for n in derived:
        premise_roots.update(db.derivation_roots(n["id"]))

    independent_support = any(s not in premise_roots for s in verdict["support_ids"])
    independent_conv = any(
        _jaccard(set(db.derivation_roots(cid)), premise_roots) <= INDEPENDENCE_MAX_OVERLAP
        for cid in convergent
    )
    if independent_support or independent_conv:
        return conf  # the lift is genuinely backed by independent evidence
    return min(conf, premise_cap)


# --- Pass 1: candidate pairing ----------------------------------------------
def _candidate_pairs(new_nodes: list[dict]) -> list[tuple[dict, dict, float]]:
    """Pair each new node with its best non-duplicate partners across the graph.

    Partners are gathered from the semantic, entity, and domain channels (whole
    graph), scored by cosine, filtered (similarity floor, duplicate ceiling,
    same-source independence), deduped by unordered id pair, and capped.
    """
    pairs: dict[frozenset, tuple[dict, dict, float]] = {}
    for node_a in new_nodes:
        a_emb = _node_embedding(node_a)
        cand_ids: set[str] = set()

        for m in db.match_nodes(a_emb, MIN_PAIR_SIMILARITY, PAIR_CANDIDATE_K + 5):
            cand_ids.add(m["id"])
        for eid in _entity_ids(node_a["id"]):
            for n in db.nodes_by_entity(eid, node_a["id"], PAIR_CANDIDATE_K):
                cand_ids.add(n["id"])
        for tid in _topic_ids(node_a["id"]):
            for n in db.nodes_by_topic(tid, node_a["id"], PAIR_CANDIDATE_K):
                cand_ids.add(n["id"])
        cand_ids.discard(node_a["id"])
        if not cand_ids:
            continue

        partners = db.nodes_by_ids(cand_ids)
        inf_ids = [p["id"] for p in partners if p.get("node_category") == "inference"]
        statuses = db.inference_statuses(inf_ids) if inf_ids else {}

        # When chaining off a derived premise, force synthesis (not restatement):
        # pair it only with events that add NEW grounding — never its own sources,
        # never a near-restatement.
        distinct_mode = SYNTH_DISTINCT_PAIRING and node_a.get("node_category") == "inference"
        premise_roots = set(db.derivation_roots(node_a["id"])) if distinct_mode else set()

        scored: list[tuple[float, dict]] = []
        for partner in partners:
            if not _is_valid_premise(partner, node_a, statuses):
                continue
            sim = _cosine(a_emb, _node_embedding(partner))
            if sim < MIN_PAIR_SIMILARITY or sim >= DUPLICATE_THRESHOLD:
                continue
            if distinct_mode and (partner["id"] in premise_roots or sim >= SYNTH_MAX_PARTNER_SIM):
                continue  # its own source / near-restatement => no new conclusion
            scored.append((sim, partner))

        scored.sort(key=lambda t: t[0], reverse=True)
        for sim, partner in scored[:PAIR_CANDIDATE_K]:
            key = frozenset((node_a["id"], partner["id"]))
            if key not in pairs:
                pairs[key] = (node_a, partner, sim)

    ranked = sorted(pairs.values(), key=lambda t: t[2], reverse=True)
    return ranked[:MAX_PAIRS_PER_BATCH]


# --- Pass 2: adversarial verification ---------------------------------------
def _hybrid_retrieve(
    alternatives: list[dict], node_a: dict, node_b: dict, inference_content: str = ""
) -> list[dict]:
    """Evidence pool: cosine over each alternative's evidence_signature UNION a
    structural pull (actor / entities / edges / time window) around the sources.

    Ordered by channel priority (signature-semantic first, time-window last),
    deduped preserving order, raw_input only, capped for the classifier prompt.

    When RERANK_EVIDENCE is on, a wider pool (EVIDENCE_RERANK_POOL) is reordered by
    a cross-encoder against `inference_content` before truncating to the same cap.
    """
    ordered_ids: list[str] = []
    seen: set[str] = {node_a["id"], node_b["id"]}

    def _add(node_id: str) -> None:
        if node_id not in seen:
            seen.add(node_id)
            ordered_ids.append(node_id)

    for alt in alternatives:
        sig = (alt or {}).get("evidence_signature")
        if not sig:
            continue
        for m in db.match_nodes(embed(sig), MIN_PAIR_SIMILARITY, EVIDENCE_RETRIEVAL_K):
            _add(m["id"])

    for node in (node_a, node_b):
        for n in db.nodes_by_actor(node.get("actor"), 10):
            _add(n["id"])
        for eid in _entity_ids(node["id"]):
            for n in db.nodes_by_entity(eid, node["id"], 10):
                _add(n["id"])
        for nid in db.neighbor_node_ids(node["id"]):
            _add(nid)

    start, end = _time_window(node_a, node_b)
    window = db.nodes_in_time_window(start, end, 50)
    if SCOPED_RETRIEVAL and window:
        # Keep only window nodes that share an actor or entity with the premises —
        # the unscoped pull floods the classifier with same-era, unrelated stories.
        prem_actors = {node_a.get("actor"), node_b.get("actor")} - {None}
        prem_entities = set(_entity_ids(node_a["id"])) | set(_entity_ids(node_b["id"]))
        ent_map = db.node_entity_map([n["id"] for n in window])
        window = [
            n for n in window
            if n.get("actor") in prem_actors
            or (ent_map.get(n["id"], set()) & prem_entities)
        ]
    for n in window:
        _add(n["id"])

    rows = {r["id"]: r for r in db.nodes_by_ids(ordered_ids)}
    evidence = [
        rows[i] for i in ordered_ids
        if i in rows and rows[i].get("node_category") == "raw_input"
    ]
    if BELIEF_REVISION:
        # Staleness: an event_announcement whose expires_at has passed no longer
        # counts as live evidence.
        now = datetime.now(timezone.utc)
        fresh = []
        for e in evidence:
            exp = _parse_ts(e.get("expires_at"))
            if exp and exp < now:
                continue
            fresh.append(e)
        evidence = fresh

    if RERANK_EVIDENCE and inference_content and evidence:
        import rerank  # local import: only loaded when reranking is enabled
        pool = evidence[:EVIDENCE_RERANK_POOL]  # rerank a wider pool, keep the same cap
        scores = rerank.score(inference_content, [e["content"] for e in pool])
        pool = [e for _, e in sorted(zip(scores, pool), key=lambda t: t[0], reverse=True)]
        return pool[:EVIDENCE_CLASSIFY_CAP]

    return evidence[:EVIDENCE_CLASSIFY_CAP]


def _coverage(node_a: dict, node_b: dict) -> float:
    """Density of related corpus around the sources, normalised to [0,1].

    Counts raw_input nodes in the time window that share an actor or subject with
    either source. High coverage => an absence of defeaters is informative.
    """
    start, end = _time_window(node_a, node_b)
    window = db.nodes_in_time_window(start, end, 200)
    actors = {node_a.get("actor"), node_b.get("actor")} - {None}
    subjects = {node_a.get("subject"), node_b.get("subject")} - {None}
    if ENTITY_COVERAGE:
        # Match on resolved entities (plus the string fallback), so surface-form
        # variants of one story still count toward density.
        prem_entities = set(_entity_ids(node_a["id"])) | set(_entity_ids(node_b["id"]))
        ent_map = db.node_entity_map([n["id"] for n in window])
        related_ids = [
            n["id"] for n in window
            if (ent_map.get(n["id"], set()) & prem_entities)
            or n.get("actor") in actors or n.get("subject") in subjects
        ]
    else:
        related_ids = [
            n["id"] for n in window
            if n.get("actor") in actors or n.get("subject") in subjects
        ]
    # Weight each related node: by source reliability (when enabled) and down-weight
    # superseded (overtaken) reports so stale density can't manufacture coverage.
    weights = db.source_weights(related_ids) if USE_SOURCE_WEIGHTS else {}
    stale = db.superseded_node_ids(related_ids) if SUPERSESSION else set()
    related = sum(
        weights.get(i, 1.0) * (SUPERSEDED_COVERAGE_WEIGHT if i in stale else 1.0)
        for i in related_ids
    )
    return min(1.0, related / COVERAGE_SATURATION)


def _coverage_ceiling(coverage: float) -> float:
    """Hard cap on confidence as a function of coverage.

    Thin coverage (0) caps at 0.40; full coverage (1) allows up to 0.95. This is
    the invariant: 'nothing contradicted it' can never yield high confidence
    unless the corpus around the claim is actually dense. Graded mode re-anchors
    the line to the measured calibration curve (thin-coverage claims were
    empirically far truer than 0.40 implied).
    """
    if GRADED_VERDICT:
        return GRADED_CEILING_FLOOR + GRADED_CEILING_SLOPE * coverage
    return 0.40 + 0.55 * coverage


def _classify_evidence_nli(inference_content: str, retrieved: list[dict]) -> list[dict]:
    """NLI path: label each retrieved node by entailment against the inference.

    entailment >= NLI_ENTAIL_THRESHOLD => supports_inference; contradiction >=
    NLI_CONTRA_THRESHOLD => supports_alternative (a defeater); else irrelevant.
    The defeater's alternative_index is left None — NLI judges the inference
    directly, not which competing alternative a node supports; a non-empty defeater
    set already drives the verdict to 'contested'.
    """
    import nli  # local import: only loaded when the NLI path is selected
    out: list[dict] = []
    for i, node in enumerate(retrieved):
        scores = nli.entail_scores(node.get("content", ""), inference_content)
        if scores["entailment"] >= NLI_ENTAIL_THRESHOLD:
            out.append({"index": i, "label": "supports_inference"})
        elif scores["contradiction"] >= NLI_CONTRA_THRESHOLD:
            out.append({"index": i, "label": "supports_alternative", "alternative_index": None})
        else:
            out.append({"index": i, "label": "irrelevant"})
    return out


def _classify_evidence(
    inference_content: str, alternatives: list[dict], retrieved: list[dict]
) -> list[dict]:
    """Select the evidence-classification path per USE_NLI_CLASSIFIER.

    "llm"  -> the existing LLM classifier (default, baseline; failover untouched).
    "nli"  -> local entailment model only.
    "both" -> label only where LLM and NLI agree; disagreement abstains to
              'irrelevant', which surfaces downstream as more 'unverified'.
    """
    mode = USE_NLI_CLASSIFIER
    if mode == "nli":
        return _classify_evidence_nli(inference_content, retrieved)
    if mode == "both":
        llm = {
            c.get("index"): c
            for c in classify_evidence(inference_content, alternatives, retrieved)
            if isinstance(c.get("index"), int)
        }
        nli_labels = {c["index"]: c for c in _classify_evidence_nli(inference_content, retrieved)}
        out: list[dict] = []
        for i in range(len(retrieved)):
            lc, nc = llm.get(i), nli_labels.get(i)
            if lc and nc and lc.get("label") == nc.get("label"):
                out.append(lc)  # agreement: keep the LLM record (carries alternative_index)
            else:
                out.append({"index": i, "label": "irrelevant"})  # abstain on disagreement
        return out
    return classify_evidence(inference_content, alternatives, retrieved)  # "llm" / default


def _should_contest(defeater_ids: list, support_ids: list) -> bool:
    """Whether defeater evidence should flip the verdict to 'contested'.

    "strict" (default, baseline): any defeater contests. "weighted": contest only
    when the defeaters aren't dominated by supports — so a single misclassified node
    can't flip a well-supported claim. Threshold via DEFEATER_SUPPORT_RATIO.
    """
    if not defeater_ids:
        return False
    if DEFEATER_POLICY == "weighted":
        return len(defeater_ids) >= max(1, math.ceil(DEFEATER_SUPPORT_RATIO * len(support_ids)))
    return True  # strict: baseline behaviour


def _debate_defeaters(
    content: str,
    node_a: dict,
    node_b: dict,
    support_ids: list[str],
    defeater_recs: list[dict],
    retrieved: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Escalation tier: opponent -> rebuttal -> mediator over borderline defeaters.

    Returns (surviving_defeater_recs, audit). Every turn is grounded in the cited
    node contents; a defeater survives only if its objection is SUSTAINED (a
    proposer concession always sustains). Fail-safe: on any LLM failure, ALL
    defeaters are kept — the debate can only ever make the verdict more lenient
    than baseline when the mediator explicitly overrules an objection.
    """
    content_by_id = {n["id"]: n.get("content", "") for n in retrieved}
    texts = [content_by_id.get(r["node_id"], "(unknown)") for r in defeater_recs]
    try:
        objections = debate_object(content, texts)
        active = [i for i, o in enumerate(objections) if o]
        audit: list[dict] = []
        sustained: list[dict] = []

        if active:
            premises = [node_a.get("content", ""), node_b.get("content", "")]
            supports_sample = [content_by_id.get(s, "") for s in support_ids[:3]]
            obj_texts = [f"{objections[i]}  [cited evidence: {texts[i]}]" for i in active]
            rebuttals = debate_rebut(content, premises, supports_sample, obj_texts)
            exchanges = [
                f"OBJECTION: {objections[i]} | EVIDENCE: {texts[i]} | PROPOSER: "
                + ("CONCEDES" if rebuttals[k]["concede"] else rebuttals[k]["response"])
                for k, i in enumerate(active)
            ]
            rulings = debate_adjudicate(content, exchanges)
            for k, i in enumerate(active):
                ruling = "sustained" if rebuttals[k]["concede"] else rulings[k]["ruling"]
                audit.append({
                    "node_id": defeater_recs[i]["node_id"],
                    "objection": objections[i],
                    "concede": rebuttals[k]["concede"],
                    "rebuttal": rebuttals[k]["response"],
                    "ruling": ruling,
                    "rationale": rulings[k]["rationale"],
                })
                if ruling == "sustained":
                    sustained.append(defeater_recs[i])

        for i, o in enumerate(objections):
            if not o:  # the opponent itself found nothing this evidence grounds
                audit.append({
                    "node_id": defeater_recs[i]["node_id"],
                    "objection": None,
                    "ruling": "overruled",
                    "rationale": "opponent found no objection grounded in the evidence",
                })
        return sustained, audit
    except Exception as exc:
        return defeater_recs, [{"error": f"debate failed, defeaters kept: {exc}"}]


def _verify_inference(
    inference: dict, node_a: dict, node_b: dict, base_conf: float
) -> dict:
    """Pass 2 a–e: alternatives -> hybrid retrieval -> classification -> status.

    Returns the verdict dict consumed by _persist_inference.
    """
    content = inference["content"]

    # Pass 1.5 — soundness gate: does the conclusion actually FOLLOW from its two
    # premises, or overreach (reversed cause/effect, an unpinned date/quantity,
    # unwarranted certainty)? A premise-only check, before the expensive Pass 2, so
    # an unsound conclusion is demoted to 'unverified' without paying for retrieval +
    # classification. Fail-safe: an inconclusive/failed check passes through (baseline).
    if SOUNDNESS_GATE:
        sc = check_soundness(node_a.get("content", ""), node_b.get("content", ""), content)
        if sc and not sc.get("sound", True):
            return {
                "status": "unverified",
                "confidence": min(base_conf, UNVERIFIED_CONFIDENCE_CAP),
                "coverage": 0.0,
                "support_ids": [],
                "defeater_ids": [],
                "alternatives": [],
                "debate": None,
                "unsound": sc.get("flaw"),
            }

    # TODO(out-of-scope): corpus-grounded reportability — estimate each alternative's
    # reportability from how often such events actually surface in the corpus,
    # rather than the LLM's prior. Hooks in here on `alternatives`.
    alternatives = enumerate_alternatives(content, MAX_ALTERNATIVES)[:MAX_ALTERNATIVES]
    retrieved = _hybrid_retrieve(alternatives, node_a, node_b, content)
    classifications = _classify_evidence(content, alternatives, retrieved)

    support_ids: list[str] = []
    defeater_recs: list[dict] = []  # {"node_id", "alt"} — alt = alternative_index
    for c in classifications:
        idx = c.get("index")
        if not isinstance(idx, int) or not (0 <= idx < len(retrieved)):
            continue
        node_id = retrieved[idx]["id"]
        label = c.get("label")
        if label == "supports_inference":
            support_ids.append(node_id)
        elif label == "supports_alternative":
            ai = c.get("alternative_index")
            defeater_recs.append({"node_id": node_id, "alt": ai if isinstance(ai, int) else None})

    if SUPERSESSION and (support_ids or defeater_recs):
        # Overtaken reports are history, not live evidence — they can neither
        # corroborate nor contest.
        stale = db.superseded_node_ids(support_ids + [r["node_id"] for r in defeater_recs])
        if stale:
            support_ids = [i for i in support_ids if i not in stale]
            defeater_recs = [r for r in defeater_recs if r["node_id"] not in stale]

    # Debate escalation: only borderline verdicts (support AND defeaters present)
    # get the opponent/rebuttal/mediator treatment; sustained objections remain.
    debate_audit = None
    if DEBATE_VERDICTS and support_ids and defeater_recs:
        defeater_recs, debate_audit = _debate_defeaters(
            content, node_a, node_b, support_ids, defeater_recs, retrieved
        )

    defeater_ids = [r["node_id"] for r in defeater_recs]
    # An alternative counts as evidenced only if a SURVIVING defeater supports it.
    alt_supported = {r["alt"] for r in defeater_recs if r["alt"] is not None}

    coverage = _coverage(node_a, node_b)

    # Empty high-reportability alternatives = informative silence (counts FOR the
    # inference); empty low-reportability alternatives count for nothing.
    empty_high = [
        _to_float(a.get("reportability"), 0.0)
        for i, a in enumerate(alternatives)
        if i not in alt_supported and _to_float(a.get("reportability"), 0.0) >= HIGH_REPORTABILITY
    ]

    # TODO(out-of-scope): replace this hand-tuned status/confidence arithmetic with a
    # learned, calibrated model once a labeled set exists (features: support/defeater
    # counts, coverage, reportability, rerank/NLI scores). Keep this branch as the
    # baseline fallback.
    # Graded mode: strong direct support (>= GRADED_MIN_SUPPORTS) corroborates even
    # below the coverage floor — the binary floor was measured as the recall wall,
    # and thin coverage still bounds *confidence* via the coverage ceiling.
    if GRADED_VERDICT:
        sufficient_support = bool(support_ids) and (
            len(support_ids) >= GRADED_MIN_SUPPORTS or coverage >= COVERAGE_CORROBORATION_MIN
        )
    else:
        sufficient_support = bool(support_ids) and coverage >= COVERAGE_CORROBORATION_MIN

    if _should_contest(defeater_ids, support_ids):
        status = "contested"
        conf = base_conf * DEFEATER_PENALTY
    elif sufficient_support and empty_high:
        status = "corroborated"
        mean_rep = sum(empty_high) / len(empty_high)
        bonus_scale = mean_rep
        if USE_SOURCE_WEIGHTS:
            # Corroboration from reliable sources earns more of the bonus.
            sw = db.source_weights(support_ids)
            if sw:
                bonus_scale *= sum(sw.values()) / len(sw)
        conf = min(1.0, base_conf + CORROBORATION_BONUS * coverage * bonus_scale)
    else:
        status = "unverified"
        conf = min(base_conf, GRADED_UNVERIFIED_CAP if GRADED_VERDICT else UNVERIFIED_CONFIDENCE_CAP)

    # Always apply the coverage gate, regardless of branch.
    conf = min(conf, _coverage_ceiling(coverage))

    # Annotate alternatives for storage/audit.
    for i, a in enumerate(alternatives):
        a["had_support"] = i in alt_supported

    return {
        "status": status,
        "confidence": conf,
        "coverage": coverage,
        "support_ids": support_ids,
        "defeater_ids": defeater_ids,
        "alternatives": alternatives,
        "debate": debate_audit,
    }


# --- Pass 3: convergence (independence-guarded) -----------------------------
def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _find_duplicate(inf_embedding: list[float], source_roots: list[str]) -> Optional[str]:
    """An existing inference this one merely RESTATES from overlapping grounding.

    Near-identical content (cosine >= MERGE_THRESHOLD) whose raw grounding overlaps
    this one's (Jaccard > INDEPENDENCE_MAX_OVERLAP) — i.e. redundant, not independent.
    Returns its id, or None. Independent re-derivations are deliberately NOT treated
    as duplicates: those are corroboration, handled by _detect_convergence.
    """
    src = set(source_roots)
    for m in db.match_inferences(inf_embedding, MERGE_THRESHOLD, 5):
        existing_roots = set(db.derivation_roots(m["id"]))
        if existing_roots and _jaccard(src, existing_roots) > INDEPENDENCE_MAX_OVERLAP:
            return m["id"]
    return None


def _detect_convergence(inf_embedding: list[float], source_roots: list[str]) -> list[str]:
    """Existing inferences that independently re-derive this one.

    Independence is judged on transitive RAW grounding (derivation_roots), not the
    direct premises: a match counts only if its raw-ancestor set is largely DISJOINT
    from this inference's (Jaccard <= INDEPENDENCE_MAX_OVERLAP). This is what stops a
    conclusion re-derived from its own descendants from posing as corroboration.
    """
    convergent: list[str] = []
    src = set(source_roots)
    matches = db.match_inferences(inf_embedding, CONVERGENCE_THRESHOLD, 10)
    # Fixed behaviour: a re-derivation only counts as corroborating convergence if it
    # is itself 'corroborated' (this also makes the stored converged_with honest).
    statuses = {} if CONVERGENCE_LEGACY else db.inference_statuses([m["id"] for m in matches])
    for m in matches:
        existing_roots = set(db.derivation_roots(m["id"]))
        if not existing_roots:
            continue
        if _jaccard(src, existing_roots) > INDEPENDENCE_MAX_OVERLAP:
            continue
        if not CONVERGENCE_LEGACY and statuses.get(m["id"]) != "corroborated":
            continue
        convergent.append(m["id"])
    return convergent


def _convergence_confidence(
    verdict: dict, node_a: dict, node_b: dict, emb: list[float], source_roots: list[str]
) -> tuple[float, list[str]]:
    """Apply the convergence bonus (then the independence cap) to a verdict's conf.

    The bonus only strengthens an already-'corroborated' verdict — so it can never
    lift an 'unverified' conclusion above its cap or boost a 'contested' one past its
    defeater penalty. ENCELADUS_CONVERGENCE_LEGACY restores the old unconditional add.
    Shared by _persist_inference and eval._predict so both see the same confidence.
    """
    convergent = _detect_convergence(emb, source_roots)
    conf = verdict["confidence"]
    if convergent and (CONVERGENCE_LEGACY or verdict["status"] == "corroborated"):
        conf = min(
            _coverage_ceiling(verdict["coverage"]),
            conf + CONVERGENCE_BONUS * len(convergent),
        )
    # Min-chain monotonicity (with independence exception): a derived conclusion
    # can't out-confidence its premises on the strength of shared evidence alone.
    conf = _independence_capped_confidence(conf, verdict, node_a, node_b, convergent)
    return conf, convergent


# --- Phase 3: belief revision (re-open verdicts as new evidence arrives) ------
def _reverify_affected(new_nodes: list[dict], exclude_ids: set) -> int:
    """Re-verify existing inferences the new raw nodes bear on. Returns the count.

    Two triggers per new node: (a) it SUPERSEDES a node some inference derives
    from — that premise is overtaken, re-open unconditionally; (b) it is
    semantically close to an inference (>= REVERIFY_SIMILARITY) — potential new
    support or defeater. Budgeted to REVERIFY_MAX_PER_BATCH per batch, highest
    priority first; `exclude_ids` skips inferences created in this same batch.
    """
    if not _revision_ok:
        return 0
    candidates: dict[str, float] = {}
    for node in new_nodes:
        emb = _node_embedding(node)
        for m in db.match_inferences(emb, REVERIFY_SIMILARITY, 5):
            if m["id"] not in exclude_ids:
                sim = _to_float(m.get("similarity"), 0.0)
                candidates[m["id"]] = max(candidates.get(m["id"], 0.0), sim)
        if SUPERSESSION:
            for old_id in db.supersedes_targets(node["id"]):
                for inf_id in db.inferences_deriving_from(old_id):
                    if inf_id not in exclude_ids:
                        candidates[inf_id] = 2.0  # premise overtaken: top priority
    ranked = sorted(candidates.items(), key=lambda t: t[1], reverse=True)
    revised = 0
    for inf_id, _prio in ranked[:REVERIFY_MAX_PER_BATCH]:
        if _reverify_one(inf_id):
            revised += 1
    return revised


def _reverify_one(inf_id: str) -> bool:
    """Re-run Pass 2 (+ convergence) for one stored inference and update its verdict.

    Reuses the inference's original premises and base_confidence; the verdict,
    confidence, evidence edges, and converged_with are all recomputed against the
    CURRENT corpus, and inference_meta.revised_at records the revision.
    """
    rows = db.nodes_by_ids([inf_id])
    if not rows:
        return False
    inf_node = rows[0]
    premises = db.nodes_by_ids(db.derives_from_targets(inf_id))
    if len(premises) < 2:
        return False
    meta = db.get_inference_meta(inf_id) or {}
    base_conf = _to_float(meta.get("base_confidence"), 0.6)

    verdict = _verify_inference({"content": inf_node["content"]}, premises[0], premises[1], base_conf)
    conf, convergent = _convergence_confidence(
        verdict, premises[0], premises[1],
        embed(inf_node["content"]), _derivation_roots(premises[0], premises[1]),
    )
    global _revision_ok
    try:
        db.update_inference_verdict(
            inf_id,
            status=verdict["status"],
            confidence=conf,
            coverage=verdict["coverage"],
            support_node_ids=verdict["support_ids"],
            defeater_node_ids=verdict["defeater_ids"],
            alternatives=verdict["alternatives"],
            converged_with=convergent,
            revised_at=datetime.now(timezone.utc).isoformat(),
            debate=verdict.get("debate"),
        )
    except Exception as exc:  # inference_meta.revised_at missing — degrade loudly, once
        _revision_ok = False
        print(f"[warn] verdict update rejected ({exc}); belief revision disabled "
              f"for this run — apply schema.sql (section 11) to enable")
        return False
    # Refresh the verification edges to mirror the new verdict.
    db.delete_edges_from(inf_id, ["corroborated_by", "contradicts", "converges_with"])
    for sid in verdict["support_ids"]:
        db.insert_edge(inf_id, sid, "corroborated_by")
    for did in verdict["defeater_ids"]:
        db.insert_edge(inf_id, did, "contradicts")
    for cid in convergent:
        db.insert_edge(inf_id, cid, "converges_with")
    return True


def _persist_inference(
    inference: dict, node_a: dict, node_b: dict, base_conf: float, verdict: dict
) -> dict:
    """Insert the inference node + meta + verification/convergence edges.

    Returns the persisted node row augmented with its verification `status`, so the
    deepening loop can feed trustworthy conclusions back in as premises.
    """
    content = inference["content"]
    emb = embed(content)
    # Independence is judged on transitive raw grounding, so a derived premise
    # contributes its own raw ancestors rather than itself.
    source_roots = _derivation_roots(node_a, node_b)

    # Detect convergence + finalize confidence. Shared with eval._predict so the
    # eval measures the confidence actually stored (incl. the convergence step).
    conf, convergent = _convergence_confidence(verdict, node_a, node_b, emb, source_roots)

    depth = 1 + max(_node_depth(node_a), _node_depth(node_b))
    inf_row = db.insert_node(
        node_category="inference",
        node_kind="derived",          # verdict lives in inference_meta.status, not node_kind
        content=content,
        actor=node_a.get("actor"),
        subject=node_a.get("subject"),
        confidence=conf,
        depth=depth,
        embedding=emb,
    )
    inf_id = inf_row["id"]

    db.insert_inference_meta(
        inf_id,
        status=verdict["status"],
        base_confidence=base_conf,
        coverage=verdict["coverage"],
        support_node_ids=verdict["support_ids"],
        defeater_node_ids=verdict["defeater_ids"],
        alternatives=verdict["alternatives"],
        converged_with=convergent,
        debate=verdict.get("debate"),
    )

    # Provenance + verification graph.
    db.insert_edge(inf_id, node_a["id"], "derives_from")
    db.insert_edge(inf_id, node_b["id"], "derives_from")
    for sid in verdict["support_ids"]:
        db.insert_edge(inf_id, sid, "corroborated_by")
    for did in verdict["defeater_ids"]:
        db.insert_edge(inf_id, did, "contradicts")
    for cid in convergent:
        db.insert_edge(inf_id, cid, "converges_with")

    # Return the persisted row (+ status) so the deepening loop can decide whether
    # this conclusion is trustworthy enough to itself become a premise next pass.
    inf_row["status"] = verdict["status"]
    return inf_row


def ingest_from_newsapi(query: str, page_size: int = 10) -> list[str]:
    """Fetch recent English articles for `query` and ingest each one.

    Returns the ids of the raw-input nodes that were created.
    """
    api_key = os.environ.get("NEWSAPI_KEY")
    if not api_key:
        raise RuntimeError("NEWSAPI_KEY must be set in the environment (.env).")

    resp = requests.get(
        NEWSAPI_URL,
        params={
            "apiKey": api_key,
            "q": query,
            "pageSize": page_size,
            "language": "en",
            "sortBy": "publishedAt",
        },
        timeout=30,
    )
    resp.raise_for_status()
    articles = resp.json().get("articles", [])

    node_ids: list[str] = []
    for i, article in enumerate(articles):
        title = article.get("title") or ""
        description = article.get("description") or ""
        url = article.get("url")
        try:
            node_ids.append(ingest_text(title + ". " + description, url))
            print(f"[{i + 1}/{len(articles)}] ingested: {title[:80]}")
        except Exception as exc:  # keep going on a single bad article
            print(f"[{i + 1}/{len(articles)}] skipped ({url}): {exc}")
        time.sleep(0.5)  # be gentle with NewsAPI / the LLM + embedding APIs

    return node_ids


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print('usage: python ingestion.py "<text to ingest>"')
        print("       python ingestion.py --infer    # flush the inference batch")
        print("       python ingestion.py --topics   # place pre-existing topics in the DAG")
        raise SystemExit(1)
    if sys.argv[1] == "--infer":
        print(run_inference_batch(force=True))
    elif sys.argv[1] == "--topics":
        print(backfill_topic_hierarchy())
    else:
        print(ingest_text(sys.argv[1]))
