"""Inspect the topic DAG: roots -> ... -> leaves, with rollup node counts.

    python topics_tree.py            # print the hierarchy
    python topics_tree.py --orphans  # list topics with no parent that aren't roots

Each line shows [direct N / rollup M]: N nodes tagged with that exact topic, M
nodes tagged with it OR any descendant (the read-time rollup). A topic reachable
by more than one parent (a DAG diamond) is printed under each but expanded once.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on sys.path

import db


def _topics() -> dict:
    rows = db.client().table("topics").select("id,name,is_root").execute().data or []
    return {r["id"]: r for r in rows}


def _children_map() -> dict:
    rels = (
        db.client().table("topic_relations").select("child_id,parent_id").execute().data
        or []
    )
    kids: dict[str, list[str]] = {}
    for r in rels:
        kids.setdefault(r["parent_id"], []).append(r["child_id"])
    return kids


def _direct_counts() -> dict:
    rows = db.client().table("node_topics").select("topic_id").execute().data or []
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["topic_id"]] = counts.get(r["topic_id"], 0) + 1
    return counts


def print_tree() -> None:
    topics = _topics()
    kids = _children_map()
    direct = _direct_counts()
    has_parent = {c for kid_list in kids.values() for c in kid_list}
    expanded: set[str] = set()

    def name_of(tid: str) -> str:
        return topics.get(tid, {}).get("name", "?")

    def walk(tid: str, prefix: str, path: frozenset) -> None:
        t = topics.get(tid)
        if not t:
            return
        flag = " *root" if t.get("is_root") else ""
        if tid in path:                       # back-edge => cycle; stop descending
            print(f"{prefix}{t['name']}{flag}  (cycle)")
            return
        rollup = len(db.nodes_under_topic(tid, 1000))
        line = f"{prefix}{t['name']}{flag}  [direct {direct.get(tid, 0)} / rollup {rollup}]"
        if tid in expanded:                   # shared node in a diamond; don't re-expand
            print(line + "  (shown above)")
            return
        print(line)
        expanded.add(tid)
        for cid in sorted(kids.get(tid, []), key=name_of):
            walk(cid, prefix + "  ", path | {tid})

    roots = sorted((t for t in topics.values() if t.get("is_root")), key=lambda t: t["name"])
    print("=== TOPIC DAG (roots -> leaves) ===")
    for r in roots:
        walk(r["id"], "", frozenset())

    unplaced = sorted(
        (t for t in topics.values() if not t.get("is_root") and t["id"] not in has_parent),
        key=lambda t: t["name"],
    )
    if unplaced:
        print("\n=== UNPLACED (no parent, not a root) — run `python ingestion.py --topics` ===")
        for t in unplaced:
            walk(t["id"], "", frozenset())


def print_orphans() -> None:
    topics = _topics()
    kids = _children_map()
    has_parent = {c for kid_list in kids.values() for c in kid_list}
    orphans = [
        t for t in topics.values() if not t.get("is_root") and t["id"] not in has_parent
    ]
    for t in sorted(orphans, key=lambda t: t["name"]):
        print(t["name"])


if __name__ == "__main__":
    if "--orphans" in sys.argv:
        print_orphans()
    else:
        print_tree()
