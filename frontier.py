"""Frontier / completeness map: what's settled, disputed, and unknown (Phase 4).

Read-time only — like streams, nothing here is materialized. Three views over the
current graph:

- contested_clusters(): the DISPUTED map — contested inferences grouped by the
  topics of their premises, with the defeating evidence attached. On real news,
  disagreement is signal; this surfaces it as a product instead of noise.
- coverage_gaps(): the UNKNOWN map — per topic (DAG rollup), how dense the raw
  corpus is and how much of the derived layer is actually settled. Thin,
  low-corroboration topics = "what we don't know yet".
- underived_links(): the un-REASONED map — strongly similar raw pairs no
  inference was ever derived from; the engine's own backlog of open questions.
- domain_view(topic): one topic's full picture (rollup nodes + the inferences
  grounded in them, by status).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

import db

THIN_COVERAGE_NODES = 4      # rollup raw-node count below which a topic is "thin"
UNDERIVED_MIN_WEIGHT = 0.60  # semantic-edge weight floor for "should have been reasoned about"


def _graph() -> dict[str, Any]:
    """One read of everything the maps need (small corpus; read-time is fine)."""
    c = db.client()
    nodes = {
        n["id"]: n
        for n in (c.table("nodes")
                  .select("id,node_category,node_kind,actor,subject,confidence,content,event_date")
                  .execute().data or [])
    }
    edges = c.table("edges").select("source_id,target_id,edge_type,weight").execute().data or []
    meta = {m["node_id"]: m for m in (c.table("inference_meta").select("*").execute().data or [])}
    node_topics = c.table("node_topics").select("node_id,topic_id").execute().data or []
    topics = {t["id"]: t for t in (c.table("topics").select("id,name,is_root").execute().data or [])}

    premises = defaultdict(list)   # inference id -> premise node ids
    for e in edges:
        if e["edge_type"] == "derives_from":
            premises[e["source_id"]].append(e["target_id"])
    topics_of = defaultdict(set)   # node id -> topic ids
    for r in node_topics:
        topics_of[r["node_id"]].add(r["topic_id"])
    return {"nodes": nodes, "edges": edges, "meta": meta,
            "topics": topics, "topics_of": topics_of, "premises": premises}


def _inference_topics(inf_id: str, g: dict) -> set:
    """An inference inherits the topics of its (transitively raw) premises."""
    out: set = set()
    stack = list(g["premises"].get(inf_id, []))
    seen = set(stack)
    while stack:
        nid = stack.pop()
        out |= g["topics_of"].get(nid, set())
        for pid in g["premises"].get(nid, []):
            if pid not in seen:
                seen.add(pid)
                stack.append(pid)
    return out


def contested_clusters(g: Optional[dict] = None) -> list[dict]:
    """Contested inferences grouped by topic, most-disputed topics first."""
    g = g or _graph()
    clusters: dict[str, list[dict]] = defaultdict(list)
    for inf_id, m in g["meta"].items():
        if m.get("status") != "contested" or inf_id not in g["nodes"]:
            continue
        node = g["nodes"][inf_id]
        item = {
            "id": inf_id,
            "content": node["content"],
            "confidence": node["confidence"],
            "defeaters": [
                {"id": d, "content": g["nodes"].get(d, {}).get("content", "(unknown)")}
                for d in (m.get("defeater_node_ids") or [])
            ],
            "supports": len(m.get("support_node_ids") or []),
        }
        topic_ids = _inference_topics(inf_id, g) or {None}
        for tid in topic_ids:
            name = g["topics"].get(tid, {}).get("name", "(untopiced)") if tid else "(untopiced)"
            clusters[name].append(item)
    return [
        {"topic": name, "contested_count": len(items), "items": items}
        for name, items in sorted(clusters.items(), key=lambda kv: -len(kv[1]))
    ]


def coverage_gaps(g: Optional[dict] = None) -> list[dict]:
    """Per topic: raw density + verdict mix; thinnest/least-settled first.

    A topic with few raw nodes, or many inferences but few corroborated, is a
    place where the model knows it doesn't know.
    """
    g = g or _graph()
    raw_by_topic: dict[str, int] = defaultdict(int)
    for nid, tids in g["topics_of"].items():
        if g["nodes"].get(nid, {}).get("node_category") == "raw_input":
            for tid in tids:
                raw_by_topic[tid] += 1
    verdicts: dict[str, dict] = defaultdict(lambda: defaultdict(int))
    for inf_id, m in g["meta"].items():
        for tid in _inference_topics(inf_id, g):
            verdicts[tid][m.get("status", "?")] += 1

    rows = []
    for tid, topic in g["topics"].items():
        raw = raw_by_topic.get(tid, 0)
        v = verdicts.get(tid, {})
        total_inf = sum(v.values())
        corroborated = v.get("corroborated", 0)
        rows.append({
            "topic": topic["name"],
            "is_root": bool(topic.get("is_root")),
            "raw_nodes": raw,
            "inferences": total_inf,
            "corroborated": corroborated,
            "contested": v.get("contested", 0),
            "unverified": v.get("unverified", 0),
            "thin": raw < THIN_COVERAGE_NODES,
            "settled_ratio": (corroborated / total_inf) if total_inf else 0.0,
        })
    rows.sort(key=lambda r: (not r["thin"], r["settled_ratio"], -r["raw_nodes"]))
    return rows


def underived_links(g: Optional[dict] = None, limit: int = 20) -> list[dict]:
    """Strong semantic raw-pairs with no inference derived from both — the
    engine's open questions, ranked by similarity."""
    g = g or _graph()
    derived_pairs = {
        frozenset(p) for p in g["premises"].values() if len(p) == 2
    }
    out, seen = [], set()
    sem = sorted(
        (e for e in g["edges"] if e["edge_type"] == "semantically_similar"
         and (e.get("weight") or 0) >= UNDERIVED_MIN_WEIGHT),
        key=lambda e: -(e.get("weight") or 0),
    )
    for e in sem:
        pair = frozenset((e["source_id"], e["target_id"]))
        if pair in seen or pair in derived_pairs:
            continue
        a, b = (g["nodes"].get(i) for i in pair)
        if not a or not b or a["node_category"] != "raw_input" or b["node_category"] != "raw_input":
            continue
        seen.add(pair)
        out.append({
            "similarity": e.get("weight"),
            "a": {"id": a["id"], "content": a["content"]},
            "b": {"id": b["id"], "content": b["content"]},
        })
        if len(out) >= limit:
            break
    return out


def domain_view(topic_name: str) -> Optional[dict]:
    """Everything the graph knows under one topic (DAG rollup + derived layer)."""
    topic = db.find_topic(topic_name)
    if not topic:
        return None
    g = _graph()
    rollup_nodes = db.nodes_under_topic(topic["id"], 500)
    rollup_ids = {n["id"] for n in rollup_nodes}
    inferences = []
    for inf_id, m in g["meta"].items():
        prem = set(g["premises"].get(inf_id, []))
        if prem & rollup_ids and inf_id in g["nodes"]:
            n = g["nodes"][inf_id]
            inferences.append({
                "id": inf_id, "content": n["content"], "confidence": n["confidence"],
                "status": m.get("status"), "revised_at": m.get("revised_at"),
            })
    inferences.sort(key=lambda i: -(i["confidence"] or 0))
    return {
        "topic": topic["name"],
        "raw_nodes": rollup_nodes,
        "inferences": inferences,
        "status_tally": dict(
            (s, sum(1 for i in inferences if i["status"] == s))
            for s in ("corroborated", "contested", "unverified")
        ),
    }
