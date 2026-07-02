"""Tests for embedding-based entity resolution (ingestion._resolve_entity).

Live test, same style as test_llm.py: hits the local embedding model and Supabase.
Requires .env (SUPABASE_URL, SUPABASE_KEY) and schema.sql applied (specifically the
match_entities function and entities_embedding_idx).

Uses synthetic, unique actor names so it never collides with or deletes real data.

Run:  python test_entities.py
"""
from __future__ import annotations

import db
from embeddings import embed
from ingestion import ENTITY_MATCH_THRESHOLD, _resolve_entity

CANON = "Zorbia Federation"                   # synthetic primary actor
VARIANT = "the Zorbia Federation"             # sibling surface form (exact match misses)
DISTINCT = "quarterly banana export tariffs"  # semantically far — must NOT merge

c = db.client()


def _cleanup() -> None:
    for name in (CANON, DISTINCT):
        c.table("entities").delete().eq("name", name).execute()


# clean slate, in case a previous run left rows behind
_cleanup()

print(f"ENTITY_MATCH_THRESHOLD = {ENTITY_MATCH_THRESHOLD}")

# seed one entity WITH an embedding (mirrors what _resolve_entity step 3 stores)
canon_id = db.insert_entity(CANON, aliases=[CANON], embedding=embed(CANON))["id"]
print(f"seeded {CANON!r} -> {canon_id}")

# ── Test 1: exact match resolves to the same entity ─────────────────
print("TEST 1: exact resolution")
assert _resolve_entity(CANON) == canon_id
print("PASS\n")

# ── Test 2: a sibling surface form collapses onto the same entity ───
print("TEST 2: embedding collapse + alias enrichment")
resolved = _resolve_entity(VARIANT)
assert resolved == canon_id, f"{VARIANT!r} should resolve to {canon_id}, got {resolved}"
# the surface form is now recorded as an alias, so next time hits the cheap path
aliases = db.find_entity(CANON)["aliases"]
assert VARIANT in aliases, f"{VARIANT!r} should have been added as an alias: {aliases}"
print("PASS\n")

# ── Test 3: a distinct actor does NOT merge (guards against false merges) ──
print("TEST 3: distinct actor stays separate")
distinct_id = _resolve_entity(DISTINCT)
assert distinct_id != canon_id, "a semantically distant name must not merge into the actor"
print("PASS\n")

_cleanup()
print("=" * 40)
print("All entity-resolution tests passed.")
