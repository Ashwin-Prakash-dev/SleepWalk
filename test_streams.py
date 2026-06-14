"""Tests for multi-entity extraction + the streams read-time prototype.

Live test, in the same style as test_llm.py: hits Groq for extraction and
Supabase for the node_entities writes and the stream_between RPC. Requires a
populated .env (GROQ_API_KEY, SUPABASE_URL, SUPABASE_KEY) and schema.sql applied.

Run:  python test_streams.py
"""
from __future__ import annotations

import json

import db
from ingestion import ingest_text
from llm_service import extract_node

TWO_ACTOR_SENTENCE = (
    "The United States warned it would impose fresh sanctions on Iran "
    "if the nuclear negotiations fail."
)

# ── Test 1: extraction yields a roles-tagged entities list ──────────
print("TEST 1: multi-entity extraction")
result = extract_node(TWO_ACTOR_SENTENCE)
print(json.dumps(result, indent=2))

# `actor` is preserved (additive change) ...
assert result["actor"] is not None, "primary actor must still be populated"
# ... and `entities` is a non-empty, roles-tagged list.
entities = result.get("entities")
assert isinstance(entities, list), "entities must be a list"
assert len(entities) >= 1, "entities should be non-empty"
for e in entities:
    assert e.get("name"), "each entity needs a name"
    assert e.get("role") in ("actor", "target", "mentioned"), f"bad role: {e.get('role')}"
assert any(e["role"] == "actor" for e in entities), "the primary actor should appear with role 'actor'"
print("PASS\n")

# ── Test 2: a two-actor sentence yields >=2 node_entities rows ──────
print("TEST 2: node_entities wiring")
node_id = ingest_text(TWO_ACTOR_SENTENCE)
links = (
    db.client().table("node_entities").select("*").eq("node_id", node_id).execute().data
)
print(json.dumps(links, indent=2))
assert len(links) >= 2, f"expected >=2 node_entities rows, got {len(links)}"
assert any(l["role"] == "actor" for l in links), "primary actor row missing"
entity_ids = [l["entity_id"] for l in links]
print("PASS\n")

# ── Test 3: stream_between returns the node for both entities ───────
print("TEST 3: stream_between")
a, b = entity_ids[0], entity_ids[1]
rows = db.stream_between(a, b)
print(json.dumps(rows, indent=2, default=str))
assert any(r["id"] == node_id for r in rows), (
    "the node should appear in the stream between two of its entities"
)
print("PASS\n")

# cleanup: drop the node we created (node_entities + edges cascade off nodes).
db.client().table("nodes").delete().eq("id", node_id).execute()

print("=" * 40)
print("All stream tests passed.")
