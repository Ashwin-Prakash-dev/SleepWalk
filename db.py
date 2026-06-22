"""Supabase data-access layer for the Enceladus knowledge graph.

Wraps the Supabase Python client with typed helpers for the three tables
(entities, nodes, edges) and the `match_nodes` similarity-search RPC.

Credentials are read from the environment (see .env / .env.example):
    SUPABASE_URL, SUPABASE_KEY
"""
from __future__ import annotations

import os
from typing import Any, Optional, Sequence

from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()

# --- allowed values, mirrored from the CHECK constraints in schema.sql -------
NODE_CATEGORIES: frozenset[str] = frozenset({"raw_input", "inference"})

NODE_KINDS: frozenset[str] = frozenset({
    "fact", "claim", "position", "event_announcement",
    "prediction", "denial", "agreement", "contradiction", "derived",
})

EDGE_TYPES: frozenset[str] = frozenset({
    "same_subject", "same_actor", "semantically_similar",
    "derives_from", "contradicts",
    "corroborated_by", "converges_with",
})

# An embedding is a 1536-element list of floats (vector(1536) in Postgres).
Embedding = Sequence[float]

_client: Optional[Client] = None


def client() -> Client:
    """Return a lazily-created, process-wide Supabase client."""
    global _client
    if _client is None:
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        if not url or not key:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_KEY must be set in the environment "
                "(copy .env.example to .env and fill them in)."
            )
        _client = create_client(url, key)
    return _client


def _first(resp: Any) -> dict[str, Any]:
    """Return the first row of an insert/select response."""
    if not resp.data:
        raise RuntimeError("Supabase returned no rows for the operation.")
    return resp.data[0]


# --- entities ----------------------------------------------------------------
def insert_entity(
    name: str,
    aliases: Optional[Sequence[str]] = None,
    embedding: Optional[Embedding] = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"name": name}
    if aliases is not None:
        payload["aliases"] = list(aliases)
    if embedding is not None:
        payload["embedding"] = list(embedding)
    return _first(client().table("entities").insert(payload).execute())


def find_entity(name: str) -> Optional[dict[str, Any]]:
    resp = client().table("entities").select("*").eq("name", name).limit(1).execute()
    return resp.data[0] if resp.data else None


def find_or_create_entity(
    name: str,
    aliases: Optional[Sequence[str]] = None,
    embedding: Optional[Embedding] = None,
) -> dict[str, Any]:
    """Return the existing entity with this (unique) name, or create it."""
    existing = find_entity(name)
    if existing is not None:
        return existing
    return insert_entity(name, aliases, embedding)


def add_entity_alias(entity_id: str, alias: str) -> dict[str, Any]:
    """Append `alias` to an entity's aliases array (no-op if already present).

    Used by embedding-based resolution to record a newly-seen surface form on the
    entity it resolved to, so future lookups hit the cheap exact/alias path.
    """
    row = _first(
        client().table("entities").select("id,aliases").eq("id", entity_id).execute()
    )
    aliases = list(row.get("aliases") or [])
    if alias in aliases:
        return row
    aliases.append(alias)
    return _first(
        client().table("entities").update({"aliases": aliases}).eq("id", entity_id).execute()
    )


def match_entities(
    query_embedding: Embedding,
    match_threshold: float = 0.85,
    match_count: int = 5,
) -> list[dict[str, Any]]:
    """Call the `match_entities` SQL function: entities near a query embedding.

    Mirrors `match_nodes`. Returns rows ordered by descending similarity.
    """
    resp = client().rpc(
        "match_entities",
        {
            "query_embedding": list(query_embedding),
            "match_threshold": match_threshold,
            "match_count": match_count,
        },
    ).execute()
    return resp.data or []


# --- nodes -------------------------------------------------------------------
def insert_node(
    node_category: str,
    node_kind: str,
    content: str,
    *,
    actor: Optional[str] = None,
    entity_id: Optional[str] = None,
    subject: Optional[str] = None,
    confidence: float = 0.8,
    source_url: Optional[str] = None,
    event_date: Optional[str] = None,
    expires_at: Optional[str] = None,
    depth: int = 0,
    source_weight: Optional[float] = None,
    embedding: Optional[Embedding] = None,
) -> dict[str, Any]:
    if node_category not in NODE_CATEGORIES:
        raise ValueError(f"node_category must be one of {sorted(NODE_CATEGORIES)}")
    if node_kind not in NODE_KINDS:
        raise ValueError(f"node_kind must be one of {sorted(NODE_KINDS)}")

    payload: dict[str, Any] = {
        "node_category": node_category,
        "node_kind": node_kind,
        "content": content,
        "confidence": confidence,
        "depth": depth,
    }
    optional = {
        "actor": actor,
        "entity_id": entity_id,
        "subject": subject,
        "source_url": source_url,
        "event_date": event_date,
        "expires_at": expires_at,
        # Only written when provided, so the column stays unrequired for the
        # baseline path (and absent DBs don't break on insert).
        "source_weight": source_weight,
    }
    payload.update({k: v for k, v in optional.items() if v is not None})
    if embedding is not None:
        payload["embedding"] = list(embedding)
    return _first(client().table("nodes").insert(payload).execute())


# --- edges -------------------------------------------------------------------
def insert_edge(
    source_id: str,
    target_id: str,
    edge_type: str,
    weight: float = 1.0,
) -> dict[str, Any]:
    if edge_type not in EDGE_TYPES:
        raise ValueError(f"edge_type must be one of {sorted(EDGE_TYPES)}")
    payload = {
        "source_id": source_id,
        "target_id": target_id,
        "edge_type": edge_type,
        "weight": weight,
    }
    return _first(client().table("edges").insert(payload).execute())


# --- node <-> entity links ---------------------------------------------------
def insert_node_entity(
    node_id: str,
    entity_id: str,
    role: Optional[str] = None,
) -> dict[str, Any]:
    """Link a node to an entity with a role (actor/target/mentioned).

    Additive to nodes.entity_id — records every participating entity in the
    node_entities join table. Upserts on the (node_id, entity_id) primary key so
    re-linking the same pair is idempotent.
    """
    payload: dict[str, Any] = {"node_id": node_id, "entity_id": entity_id}
    if role is not None:
        payload["role"] = role
    return _first(client().table("node_entities").upsert(payload).execute())


# --- topics ------------------------------------------------------------------
def insert_topic(
    name: str,
    aliases: Optional[Sequence[str]] = None,
    embedding: Optional[Embedding] = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"name": name}
    if aliases is not None:
        payload["aliases"] = list(aliases)
    if embedding is not None:
        payload["embedding"] = list(embedding)
    return _first(client().table("topics").insert(payload).execute())


def find_topic(name: str) -> Optional[dict[str, Any]]:
    resp = client().table("topics").select("*").eq("name", name).limit(1).execute()
    return resp.data[0] if resp.data else None


def add_topic_alias(topic_id: str, alias: str) -> dict[str, Any]:
    row = _first(
        client().table("topics").select("id,aliases").eq("id", topic_id).execute()
    )
    aliases = list(row.get("aliases") or [])
    if alias in aliases:
        return row
    aliases.append(alias)
    return _first(
        client().table("topics").update({"aliases": aliases}).eq("id", topic_id).execute()
    )


def match_topics(
    query_embedding: Embedding,
    match_threshold: float = 0.80,
    match_count: int = 5,
) -> list[dict[str, Any]]:
    resp = client().rpc(
        "match_topics",
        {
            "query_embedding": list(query_embedding),
            "match_threshold": match_threshold,
            "match_count": match_count,
        },
    ).execute()
    return resp.data or []


def insert_node_topic(node_id: str, topic_id: str) -> dict[str, Any]:
    payload: dict[str, Any] = {"node_id": node_id, "topic_id": topic_id}
    return _first(client().table("node_topics").upsert(payload).execute())


# --- topic hierarchy (overlapping DAG over the flat topics) ------------------
def insert_topic_relation(child_id: str, parent_id: str) -> None:
    """Record a child IS-A parent edge in the topic DAG (idempotent)."""
    if child_id == parent_id:
        return
    client().table("topic_relations").upsert(
        {"child_id": child_id, "parent_id": parent_id}
    ).execute()


def topic_parents(topic_id: str) -> list[str]:
    """Direct parent topic ids of `topic_id`."""
    rows = (
        client().table("topic_relations").select("parent_id")
        .eq("child_id", topic_id).execute().data or []
    )
    return [r["parent_id"] for r in rows]


def topic_children(topic_id: str) -> list[str]:
    """Direct child topic ids of `topic_id`."""
    rows = (
        client().table("topic_relations").select("child_id")
        .eq("parent_id", topic_id).execute().data or []
    )
    return [r["child_id"] for r in rows]


def topic_descendant_ids(root: str) -> list[str]:
    """All topic ids at or below `root` (root included), via the recursive RPC."""
    resp = client().rpc("topic_descendants", {"root": root}).execute()
    return [r["id"] for r in (resp.data or [])]


def topic_ancestor_ids(start_id: str) -> list[str]:
    """All topic ids at or above `start_id` (start included), via the recursive RPC."""
    resp = client().rpc("topic_ancestors", {"start_id": start_id}).execute()
    return [r["id"] for r in (resp.data or [])]


def set_topic_root(topic_id: str, is_root: bool = True) -> None:
    """Mark (or unmark) a topic as a curated top-level domain."""
    client().table("topics").update({"is_root": is_root}).eq("id", topic_id).execute()


def list_root_topics() -> list[dict[str, Any]]:
    """Curated top-level domain topics (is_root = true)."""
    return (
        client().table("topics").select("id,name,aliases")
        .eq("is_root", True).order("name").execute().data or []
    )


def nodes_under_topic(root: str, match_count: int = 50) -> list[dict[str, Any]]:
    """Read-time rollup: recent raw_input nodes tagged with `root` or any descendant."""
    resp = client().rpc(
        "nodes_under_topic", {"root": root, "match_count": match_count}
    ).execute()
    return resp.data or []


def nodes_by_entity(entity_id: str, exclude_id: str, limit: int = 10) -> list[dict[str, Any]]:
    """Recent raw_input nodes involving entity_id, for inference pool expansion."""
    id_rows = (
        client()
        .table("node_entities")
        .select("node_id")
        .eq("entity_id", entity_id)
        .neq("node_id", exclude_id)
        .limit(50)
        .execute()
        .data or []
    )
    if not id_rows:
        return []
    ids = [r["node_id"] for r in id_rows]
    return (
        client()
        .table("nodes")
        .select("id,node_kind,actor,subject,confidence,content")
        .in_("id", ids)
        .eq("node_category", "raw_input")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
        .data or []
    )


def nodes_by_topic(topic_id: str, exclude_id: str, limit: int = 10) -> list[dict[str, Any]]:
    """Recent raw_input nodes in a topic domain, for inference pool expansion."""
    id_rows = (
        client()
        .table("node_topics")
        .select("node_id")
        .eq("topic_id", topic_id)
        .neq("node_id", exclude_id)
        .limit(50)
        .execute()
        .data or []
    )
    if not id_rows:
        return []
    ids = [r["node_id"] for r in id_rows]
    return (
        client()
        .table("nodes")
        .select("id,node_kind,actor,subject,confidence,content")
        .in_("id", ids)
        .eq("node_category", "raw_input")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
        .data or []
    )


# --- similarity search -------------------------------------------------------
def match_nodes(
    query_embedding: Embedding,
    match_threshold: float = 0.75,
    match_count: int = 15,
) -> list[dict[str, Any]]:
    """Call the `match_nodes` SQL function and return matching node rows."""
    resp = client().rpc(
        "match_nodes",
        {
            "query_embedding": list(query_embedding),
            "match_threshold": match_threshold,
            "match_count": match_count,
        },
    ).execute()
    return resp.data or []


# --- streams (read-time entity-pair channel) ---------------------------------
def stream_between(
    a: str,
    b: str,
    max_count: int = 100,
) -> list[dict[str, Any]]:
    """Call the `stream_between` SQL function: nodes involving BOTH entities.

    Read-time prototype — derived from node_entities on every call; there is no
    materialized streams table.
    """
    resp = client().rpc(
        "stream_between",
        {"a": a, "b": b, "max_count": max_count},
    ).execute()
    return resp.data or []


def stream_between_names(
    name_a: str,
    name_b: str,
    max_count: int = 100,
) -> list[dict[str, Any]]:
    """Convenience overload: resolve two entity names to ids, then stream_between.

    Returns an empty list if either name does not match an existing entity.
    """
    a = find_entity(name_a)
    b = find_entity(name_b)
    if a is None or b is None:
        return []
    return stream_between(a["id"], b["id"], max_count)


# --- batched inference: queue, retrieval, metadata ---------------------------
# Columns returned to the inference engine — everything it reasons over except
# the 1536-dim embedding (embeddings are recomputed from `content` when needed,
# which sidesteps parsing the pgvector wire format back into a list).
INFERENCE_NODE_COLUMNS = (
    "id,node_category,node_kind,actor,subject,confidence,content,"
    "source_url,event_date,created_at,depth"
)


def nodes_unprocessed_count() -> int:
    """Count raw_input nodes not yet seen by the batched inference engine."""
    resp = (
        client()
        .table("nodes")
        .select("id", count="exact")
        .eq("node_category", "raw_input")
        .eq("inference_processed", False)
        .execute()
    )
    return resp.count or 0


def fetch_unprocessed_nodes(limit: int = 200) -> list[dict[str, Any]]:
    """Oldest-first raw_input nodes with inference_processed = false."""
    return (
        client()
        .table("nodes")
        .select(INFERENCE_NODE_COLUMNS)
        .eq("node_category", "raw_input")
        .eq("inference_processed", False)
        .order("created_at", desc=False)
        .limit(limit)
        .execute()
        .data or []
    )


def mark_nodes_processed(ids: Sequence[str]) -> None:
    """Flag nodes as processed so the next batch doesn't re-pair them."""
    if not ids:
        return
    client().table("nodes").update({"inference_processed": True}).in_(
        "id", list(ids)
    ).execute()


def source_weights(ids: Sequence[str]) -> dict[str, float]:
    """Map node_id -> source_weight for the given ids (default 1.0 when null).

    Fetched on demand (only when source weighting is enabled) so the column is not
    required by the baseline path. Selecting it does require the column to exist.
    """
    if not ids:
        return {}
    rows = (
        client().table("nodes").select("id,source_weight").in_("id", list(ids)).execute().data or []
    )
    return {r["id"]: (r.get("source_weight") if r.get("source_weight") is not None else 1.0) for r in rows}


def nodes_by_ids(ids: Sequence[str], limit: int = 200) -> list[dict[str, Any]]:
    """Fetch node rows (no embedding) for a set of ids."""
    if not ids:
        return []
    return (
        client()
        .table("nodes")
        .select(INFERENCE_NODE_COLUMNS)
        .in_("id", list(ids))
        .limit(limit)
        .execute()
        .data or []
    )


def nodes_by_actor(actor: str, limit: int = 20) -> list[dict[str, Any]]:
    """Recent raw_input nodes with this exact actor (structural evidence pull)."""
    if not actor:
        return []
    return (
        client()
        .table("nodes")
        .select(INFERENCE_NODE_COLUMNS)
        .eq("actor", actor)
        .eq("node_category", "raw_input")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
        .data or []
    )


def nodes_in_time_window(
    start_iso: str, end_iso: str, limit: int = 200
) -> list[dict[str, Any]]:
    """raw_input nodes whose created_at falls in [start_iso, end_iso]."""
    return (
        client()
        .table("nodes")
        .select(INFERENCE_NODE_COLUMNS)
        .eq("node_category", "raw_input")
        .gte("created_at", start_iso)
        .lte("created_at", end_iso)
        .limit(limit)
        .execute()
        .data or []
    )


def neighbor_node_ids(node_id: str) -> list[str]:
    """One-hop edge neighbors of a node (either direction), excluding itself."""
    edges = (
        client()
        .table("edges")
        .select("source_id,target_id")
        .or_(f"source_id.eq.{node_id},target_id.eq.{node_id}")
        .execute()
        .data or []
    )
    ids = {e["source_id"] for e in edges} | {e["target_id"] for e in edges}
    ids.discard(node_id)
    return list(ids)


def derives_from_targets(inference_id: str) -> list[str]:
    """The source nodes an inference was derived from (its derives_from targets)."""
    edges = (
        client()
        .table("edges")
        .select("target_id")
        .eq("source_id", inference_id)
        .eq("edge_type", "derives_from")
        .execute()
        .data or []
    )
    return [e["target_id"] for e in edges]


def derivation_roots(node_id: str) -> list[str]:
    """Transitive raw_input ancestors of a node (just itself, if it is raw_input).

    Walks derives_from edges down to the leaves via the recursive RPC. This is the
    grounding set the independence/convergence guard compares on.
    """
    resp = client().rpc("derivation_roots", {"start_id": node_id}).execute()
    return [r["id"] for r in (resp.data or [])]


def inference_statuses(ids: Sequence[str]) -> dict[str, str]:
    """Map inference node_id -> verification status for the given ids.

    Used to gate which inferences may serve as premises (only 'corroborated' ones).
    """
    if not ids:
        return {}
    rows = (
        client()
        .table("inference_meta")
        .select("node_id,status")
        .in_("node_id", list(ids))
        .execute()
        .data or []
    )
    return {r["node_id"]: r["status"] for r in rows}


def match_inferences(
    query_embedding: Embedding,
    match_threshold: float = 0.90,
    match_count: int = 10,
) -> list[dict[str, Any]]:
    """Call the `match_inferences` SQL function: inference nodes near an embedding."""
    resp = client().rpc(
        "match_inferences",
        {
            "query_embedding": list(query_embedding),
            "match_threshold": match_threshold,
            "match_count": match_count,
        },
    ).execute()
    return resp.data or []


def insert_inference_meta(
    node_id: str,
    status: str,
    base_confidence: float,
    coverage: float,
    support_node_ids: Sequence[str],
    defeater_node_ids: Sequence[str],
    alternatives: Any,
    converged_with: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Upsert the verification verdict + provenance for an inference node."""
    payload: dict[str, Any] = {
        "node_id": node_id,
        "status": status,
        "base_confidence": base_confidence,
        "coverage": coverage,
        "support_node_ids": list(support_node_ids or []),
        "defeater_node_ids": list(defeater_node_ids or []),
        "alternatives": alternatives or [],
        "converged_with": list(converged_with or []),
    }
    return _first(client().table("inference_meta").upsert(payload).execute())
