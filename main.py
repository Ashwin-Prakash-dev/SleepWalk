"""FastAPI surface for the Enceladus knowledge graph.

Run:  python main.py        (or: uvicorn main:app --port 8000)

Endpoints:
  POST /ingest            ingest one piece of text
  POST /ingest/news       ingest a batch from NewsAPI
  GET  /nodes             list nodes (filterable)
  GET  /nodes/{id}/graph  one-hop provenance trace around a node
  GET  /inferences        list inference nodes
  GET  /entities          list entities + aliases
"""
from __future__ import annotations

from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import db
from ingestion import ingest_from_newsapi, ingest_text

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
