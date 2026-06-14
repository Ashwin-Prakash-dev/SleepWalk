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

import os
import time
from typing import Any, Optional

import requests
from dotenv import load_dotenv

import db
from embeddings import embed
from llm_service import extract_node, run_inference

load_dotenv()

NEWSAPI_URL = "https://newsapi.org/v2/everything"
# Cosine threshold for semantic edges + inference context. Calibrated for the
# local all-MiniLM-L6-v2 model, whose related-text scores (~0.45-0.65) run lower
# than OpenAI's (~0.8+). Raise toward 0.75 if you switch back to OpenAI embeddings.
SIMILARITY_THRESHOLD = 0.45
STRUCTURAL_EDGE_LIMIT = 10
INFERENCE_CONTEXT_SIZE = 15


# --- small helpers -----------------------------------------------------------
def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _pg_quote(value: str) -> str:
    """Escape a value for embedding inside a PostgREST filter string."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _resolve_entity(actor: str) -> str:
    """Return the entity id for `actor`, creating the entity if needed.

    Matches `entities` where name ILIKE actor OR aliases @> ARRAY[actor].
    """
    safe = _pg_quote(actor)
    res = (
        db.client()
        .table("entities")
        .select("*")
        .or_(f'name.ilike."{safe}",aliases.cs.{{"{safe}"}}')
        .limit(1)
        .execute()
    )
    if res.data:
        return res.data[0]["id"]
    return db.insert_entity(name=actor, aliases=[actor])["id"]


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


def _link_entities(
    node_id: str,
    primary_entity_id: Optional[str],
    extracted_entities: Optional[list[dict]],
) -> None:
    """Populate node_entities for a node (additive; nodes.entity_id is untouched).

    Links the primary actor (role 'actor', via the already-resolved
    `primary_entity_id`) plus every entity from the extraction's `entities` list,
    each resolved through the shared `_resolve_entity` alias logic. Deduped per
    entity, with the primary actor winning so it always keeps role 'actor'.
    """
    roles: dict[str, str] = {}
    if primary_entity_id:
        roles[primary_entity_id] = "actor"
    for item in extracted_entities or []:
        name = (item or {}).get("name")
        if not name:
            continue
        entity_id = _resolve_entity(name)
        roles.setdefault(entity_id, (item or {}).get("role") or "mentioned")
    for entity_id, role in roles.items():
        db.insert_node_entity(node_id, entity_id, role)


# --- pipeline ----------------------------------------------------------------
def ingest_text(text: str, source_url: str = None) -> str:
    """Run the full ingestion pipeline on one piece of text.

    Returns the id of the inserted raw-input node.
    """
    # Step 1 — structured extraction.
    node = extract_node(text, source_url)
    actor = node.get("actor") or None
    subject = node.get("subject") or None

    # Step 2 — embed the core content.
    embedding = embed(node["content"])

    # Step 3 — entity resolution.
    entity_id = _resolve_entity(actor) if actor else None

    # Step 4 — insert the raw-input node.
    new_row = db.insert_node(
        node_category="raw_input",
        node_kind=node["node_kind"],
        content=node["content"],
        actor=actor,
        entity_id=entity_id,
        subject=subject,
        confidence=_to_float(node.get("confidence"), 0.8),
        source_url=source_url,
        expires_at=node.get("expires_at"),
        embedding=embedding,
    )
    new_id = new_row["id"]

    # Step 4b — multi-entity links (additive). The primary actor (entity_id) and
    # every extracted entity land in node_entities; nodes.entity_id is unchanged.
    _link_entities(new_id, entity_id, node.get("entities"))

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

    # Step 7 — inference.
    _run_inference_step(node, new_id, actor, subject, similar[:INFERENCE_CONTEXT_SIZE])

    return new_id


def _run_inference_step(
    node: dict,
    new_id: str,
    actor: Optional[str],
    subject: Optional[str],
    similar: list[dict],
) -> None:
    """Generate inference nodes and their derives_from/contradicts edges."""
    for inf in run_inference(node, similar):
        content = inf.get("content")
        if not content:
            continue
        inf_kind = inf.get("inference_kind")

        # inference_kind ∈ {contradiction, derives_from, supports, tension}, but the
        # nodes CHECK constraint only allows 'contradiction'/'derived' here — map the
        # rest to 'derived'. The relationship itself is carried by the edge type.
        node_kind = inf_kind if inf_kind in db.NODE_KINDS else "derived"
        edge_type = "contradicts" if inf_kind == "contradiction" else "derives_from"

        inf_row = db.insert_node(
            node_category="inference",
            node_kind=node_kind,
            content=content,
            actor=actor,
            subject=subject,
            confidence=_to_float(inf.get("confidence"), 0.7),
            embedding=embed(content),
        )
        inf_id = inf_row["id"]

        # Edges to the stored nodes that support this inference.
        for idx in inf.get("source_node_indices") or []:
            if isinstance(idx, int) and 0 <= idx < len(similar):
                db.insert_edge(inf_id, similar[idx]["id"], edge_type)

        # And a derives_from edge back to the node that triggered the inference.
        db.insert_edge(inf_id, new_id, "derives_from")


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
        raise SystemExit(1)
    print(ingest_text(sys.argv[1]))
