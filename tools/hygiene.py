"""Graph hygiene: singleton-entity report/prune + near-duplicate topic merge.

    python tools/hygiene.py                    # report only (dry run, default)
    python tools/hygiene.py --prune-singletons # delete orphan/singleton junk entities
    python tools/hygiene.py --merge-topics     # merge near-duplicate topics

Real-news ingestion accumulates noise: one-mention entities ("ANZ strategist",
"Report") that fragment streams/coverage, and near-duplicate topics ("trade" vs
"international trade") that fragment the DAG. Both operations print exactly what
they would do; nothing is deleted or merged without the explicit flag.

Pruning rule: an entity is prunable when it is linked to <= SINGLETON_MAX_LINKS
nodes AND is not the primary actor (nodes.entity_id) of any node. Topic merge
rule: cosine >= TOPIC_MERGE_THRESHOLD between topic-name embeddings; the topic
with fewer node links merges into the larger (links repointed, aliases carried
over, DAG relations moved, loser deleted).
"""
from __future__ import annotations

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on sys.path

import db
from embeddings import embed

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

SINGLETON_MAX_LINKS = 1
TOPIC_MERGE_THRESHOLD = 0.90  # deliberately above TOPIC_MATCH_THRESHOLD (0.82)


def _cosine(a, b) -> float:
    return sum(x * y for x, y in zip(a, b))


# --- singleton entities --------------------------------------------------------
def singleton_entities() -> list[dict]:
    """Entities with <= SINGLETON_MAX_LINKS node links and no primary-actor use."""
    c = db.client()
    entities = c.table("entities").select("id,name,aliases").execute().data or []
    links = c.table("node_entities").select("entity_id").execute().data or []
    link_counts = Counter(r["entity_id"] for r in links)
    primary = {
        r["entity_id"]
        for r in (c.table("nodes").select("entity_id").not_.is_("entity_id", "null").execute().data or [])
    }
    return [
        e for e in entities
        if link_counts.get(e["id"], 0) <= SINGLETON_MAX_LINKS and e["id"] not in primary
    ]


def prune_singletons(execute: bool) -> None:
    victims = singleton_entities()
    print(f"=== singleton entities ({len(victims)}) ===")
    for e in victims:
        print(f"  {e['name']}")
    if not victims:
        return
    if not execute:
        print("(dry run — pass --prune-singletons to delete)")
        return
    c = db.client()
    ids = [e["id"] for e in victims]
    c.table("node_entities").delete().in_("entity_id", ids).execute()
    c.table("entities").delete().in_("id", ids).execute()
    print(f"pruned {len(ids)} entities")


# --- near-duplicate topics -----------------------------------------------------
def duplicate_topic_pairs() -> list[tuple[dict, dict, float]]:
    """(keep, merge, similarity) pairs above TOPIC_MERGE_THRESHOLD.

    Embeddings are recomputed from names (cheap, local model) rather than parsing
    the stored pgvector wire format. Keep = the more-linked topic.
    """
    c = db.client()
    topics = c.table("topics").select("id,name,aliases,is_root").execute().data or []
    links = Counter(r["topic_id"] for r in (c.table("node_topics").select("topic_id").execute().data or []))
    vecs = {t["id"]: embed(t["name"]) for t in topics}
    pairs, merged = [], set()
    for i, a in enumerate(topics):
        for b in topics[i + 1:]:
            if a["id"] in merged or b["id"] in merged:
                continue
            sim = _cosine(vecs[a["id"]], vecs[b["id"]])
            if sim < TOPIC_MERGE_THRESHOLD:
                continue
            # Keep the busier topic; a root always wins over a non-root.
            keep, merge = (a, b) if (a.get("is_root"), links.get(a["id"], 0)) >= (b.get("is_root"), links.get(b["id"], 0)) else (b, a)
            if merge.get("is_root"):
                continue  # never merge away a curated root
            pairs.append((keep, merge, sim))
            merged.add(merge["id"])
    return pairs


def merge_topics(execute: bool) -> None:
    pairs = duplicate_topic_pairs()
    print(f"\n=== near-duplicate topics ({len(pairs)} merge pairs, cosine >= {TOPIC_MERGE_THRESHOLD}) ===")
    for keep, merge, sim in pairs:
        print(f"  {merge['name']!r} -> {keep['name']!r}  (sim {sim:.2f})")
    if not pairs:
        return
    if not execute:
        print("(dry run — pass --merge-topics to merge)")
        return
    c = db.client()
    for keep, merge, _sim in pairs:
        # Repoint node links (upsert dodges (node_id, topic_id) PK collisions).
        rows = c.table("node_topics").select("node_id").eq("topic_id", merge["id"]).execute().data or []
        for r in rows:
            c.table("node_topics").upsert({"node_id": r["node_id"], "topic_id": keep["id"]}).execute()
        c.table("node_topics").delete().eq("topic_id", merge["id"]).execute()
        # Move DAG relations (skip self-loops), then carry aliases and delete.
        for rel, col, other in (("child_id", "parent_id", keep), ("parent_id", "child_id", keep)):
            rows = c.table("topic_relations").select(col).eq(rel, merge["id"]).execute().data or []
            for r in rows:
                if r[col] != keep["id"]:
                    c.table("topic_relations").upsert({rel: keep["id"], col: r[col]}).execute()
            c.table("topic_relations").delete().eq(rel, merge["id"]).execute()
        for alias in [merge["name"], *(merge.get("aliases") or [])]:
            db.add_topic_alias(keep["id"], alias)
        c.table("topics").delete().eq("id", merge["id"]).execute()
    print(f"merged {len(pairs)} topics")


if __name__ == "__main__":
    prune_singletons("--prune-singletons" in sys.argv)
    merge_topics("--merge-topics" in sys.argv)
