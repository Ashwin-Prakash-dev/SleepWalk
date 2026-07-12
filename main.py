"""FastAPI surface for the Enceladus knowledge graph.

Run:  python main.py        (or: uvicorn main:app --port 8000)

Endpoints:
  GET  /dashboard         single-page UI for browsing inferences/events/entities
  GET  /dashboard/data    aggregate JSON powering the dashboard
  POST /ingest            ingest one piece of text
  POST /ingest/news       ingest a batch from NewsAPI
  POST /ingest/poll       one incremental domain-stream cycle (Phase 5)
  POST /infer/run         flush the batched inference engine
  GET  /nodes             list nodes (filterable)
  GET  /nodes/{id}/graph  one-hop provenance trace around a node
  GET  /inferences        list inference nodes
  GET  /entities          list entities + aliases
  GET  /contested         disputed map: contested inferences clustered by topic
  GET  /frontier          unknown map: coverage gaps + un-derived links
  GET  /domain/{topic}    one topic's full rollup (raw + derived, by status)
  POST /ask               Analysis of Competing Hypotheses over the graph (decision support)
"""
from __future__ import annotations

import os
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import ach
import db
import domain_stream
import frontier
from ingestion import ingest_from_newsapi, ingest_text, run_inference_batch

# Columns returned to clients — everything except the 1536-dim embedding.
NODE_COLUMNS = (
    "id,node_category,node_kind,content,actor,entity_id,subject,"
    "confidence,source_url,event_date,expires_at,created_at"
)

# /inferences `kind` filter -> stored node_kind. The relationship lives on the
# edge; inference nodes are stored as 'contradiction' or 'derived'.
INFERENCE_KIND_MAP = {"contradiction": "contradiction", "derives_from": "derived"}

app = FastAPI(title="Enceladus", description="Geopolitical knowledge graph")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- request bodies ----------------------------------------------------------
class IngestBody(BaseModel):
    text: str
    source_url: Optional[str] = None


class NewsBody(BaseModel):
    query: str
    page_size: int = 10


# --- endpoints ---------------------------------------------------------------
@app.get("/")
def health() -> dict:
    return {"status": "ok", "service": "enceladus"}


# --- dashboard ---------------------------------------------------------------
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    """Serve the single-page dashboard."""
    path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    with open(path, encoding="utf-8") as fh:
        return HTMLResponse(fh.read())


@app.get("/dashboard/data")
def dashboard_data() -> dict:
    """One aggregate payload powering the dashboard.

    Joins inference nodes with their inference_meta verdicts and resolves the
    source / support / defeater / convergence node ids to their content, so the
    client renders the full verification picture without N+1 round trips.
    """
    c = db.client()
    nodes = (
        c.table("nodes")
        .select("id,node_category,node_kind,actor,subject,confidence,content,source_url,created_at")
        .order("created_at", desc=True)
        .execute()
        .data
    )
    content_by_id = {n["id"]: n["content"] for n in nodes}
    raw = [n for n in nodes if n["node_category"] == "raw_input"]
    inf_nodes = [n for n in nodes if n["node_category"] == "inference"]

    meta = {m["node_id"]: m for m in c.table("inference_meta").select("*").execute().data}
    edges = c.table("edges").select("source_id,target_id,edge_type").execute().data

    sources_by_inf: dict[str, list[str]] = {}
    for e in edges:
        if e["edge_type"] == "derives_from":
            sources_by_inf.setdefault(e["source_id"], []).append(e["target_id"])

    def brief(ids: Optional[list]) -> list[dict]:
        return [{"id": i, "content": content_by_id.get(i, "(unknown)")} for i in (ids or [])]

    inferences = []
    for n in inf_nodes:
        m = meta.get(n["id"], {})
        inferences.append({
            "id": n["id"],
            "content": n["content"],
            "actor": n["actor"],
            "subject": n["subject"],
            "confidence": n["confidence"],
            "status": m.get("status", "unknown"),
            "base_confidence": m.get("base_confidence"),
            "coverage": m.get("coverage"),
            "sources": brief(sources_by_inf.get(n["id"])),
            "support": brief(m.get("support_node_ids")),
            "defeaters": brief(m.get("defeater_node_ids")),
            "converged_with": brief(m.get("converged_with")),
            "alternatives": m.get("alternatives") or [],
        })
    inferences.sort(key=lambda x: x["confidence"] or 0, reverse=True)

    tally: dict[str, int] = {}
    for i in inferences:
        tally[i["status"]] = tally.get(i["status"], 0) + 1

    entities = c.table("entities").select("name,aliases,created_at").order("name").execute().data
    topics = c.table("topics").select("name,aliases,created_at").order("name").execute().data

    return {
        "stats": {
            "raw": len(raw),
            "inference": len(inf_nodes),
            "entities": len(entities),
            "topics": len(topics),
            "edges": len(edges),
            "status_tally": tally,
        },
        "inferences": inferences,
        "events": raw,
        "entities": entities,
        "topics": topics,
    }


@app.post("/ingest")
def ingest(body: IngestBody) -> dict:
    try:
        node_id = ingest_text(body.text, body.source_url)
    except Exception as exc:  # extraction / embedding / DB failure
        raise HTTPException(status_code=500, detail=str(exc))

    row = (
        db.client()
        .table("nodes")
        .select("id,node_kind,actor,subject,confidence")
        .eq("id", node_id)
        .single()
        .execute()
        .data
    )
    return {
        "node_id": node_id,
        "node_kind": row["node_kind"],
        "actor": row["actor"],
        "subject": row["subject"],
        "confidence": row["confidence"],
    }


@app.post("/ingest/news")
def ingest_news(body: NewsBody) -> dict:
    try:
        node_ids = ingest_from_newsapi(body.query, body.page_size)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"ingested_count": len(node_ids), "node_ids": node_ids}


class PollBody(BaseModel):
    queries: Optional[list[str]] = None
    page_size: int = 8


@app.post("/ingest/poll")
def ingest_poll(body: PollBody) -> dict:
    """One incremental domain-stream cycle: fetch -> skip seen -> ingest -> infer/revise."""
    try:
        return domain_stream.poll(body.queries, body.page_size)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# --- frontier / completeness map (Phase 4; read-time, never materialized) -----
@app.get("/contested")
def contested() -> list[dict]:
    """The disputed map: contested inferences clustered by topic, with defeaters."""
    return frontier.contested_clusters()


@app.get("/frontier")
def frontier_map() -> dict:
    """The unknown map: per-topic coverage gaps + strong un-derived links."""
    g = frontier._graph()
    return {
        "coverage_gaps": frontier.coverage_gaps(g),
        "underived_links": frontier.underived_links(g),
    }


@app.get("/domain/{topic_name}")
def domain(topic_name: str) -> dict:
    """Everything the graph knows under one topic (DAG rollup + derived layer)."""
    view = frontier.domain_view(topic_name)
    if view is None:
        raise HTTPException(status_code=404, detail=f"topic {topic_name!r} not found")
    return view


class AskBody(BaseModel):
    question: str
    hypotheses: Optional[list[str]] = None
    max_hypotheses: int = 5


@app.post("/ask")
def ask(body: AskBody) -> dict:
    """Analysis of Competing Hypotheses over the graph — decision support, not a verdict."""
    try:
        return ach.ask(body.question, body.hypotheses, body.max_hypotheses)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/infer/run")
def infer_run() -> dict:
    """Manually flush the batched inference engine (force a partial batch)."""
    try:
        return run_inference_batch(force=True)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/nodes")
def list_nodes(
    actor: Optional[str] = None,
    subject: Optional[str] = None,
    node_kind: Optional[str] = None,
    limit: int = 50,
) -> list[dict]:
    query = db.client().table("nodes").select(NODE_COLUMNS)
    if actor:
        query = query.eq("actor", actor)
    if subject:
        query = query.eq("subject", subject)
    if node_kind:
        query = query.eq("node_kind", node_kind)
    return query.order("created_at", desc=True).limit(limit).execute().data


@app.get("/nodes/{node_id}/graph")
def node_graph(node_id: str) -> dict:
    node = (
        db.client().table("nodes").select(NODE_COLUMNS).eq("id", node_id).execute().data
    )
    if not node:
        raise HTTPException(status_code=404, detail="node not found")

    edges = (
        db.client()
        .table("edges")
        .select("*")
        .or_(f"source_id.eq.{node_id},target_id.eq.{node_id}")
        .execute()
        .data
    )

    neighbor_ids = {e["source_id"] for e in edges} | {e["target_id"] for e in edges}
    neighbor_ids.discard(node_id)

    connected = []
    if neighbor_ids:
        connected = (
            db.client()
            .table("nodes")
            .select(NODE_COLUMNS)
            .in_("id", list(neighbor_ids))
            .execute()
            .data
        )

    return {"node": node[0], "edges": edges, "connected_nodes": connected}


@app.get("/inferences")
def list_inferences(
    kind: Optional[str] = None,
    min_confidence: float = 0.6,
    limit: int = 20,
) -> list[dict]:
    query = (
        db.client()
        .table("nodes")
        .select(NODE_COLUMNS)
        .eq("node_category", "inference")
        .gte("confidence", min_confidence)
    )
    if kind:
        query = query.eq("node_kind", INFERENCE_KIND_MAP.get(kind, kind))
    return query.order("confidence", desc=True).limit(limit).execute().data


@app.get("/entities")
def list_entities() -> list[dict]:
    return (
        db.client()
        .table("entities")
        .select("id,name,aliases,created_at")
        .order("created_at", desc=True)
        .execute()
        .data
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
