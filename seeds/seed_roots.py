"""Seed curated top-level topic roots, then (re)place the topic DAG.

    python seed_roots.py            # seed roots, place any not-yet-placed topics
    python seed_roots.py --replace  # clear existing relations + is_root, then re-place

Curated roots bias classify_topic_parent onto a small shared vocabulary, so the
hierarchy converges onto these instead of sprouting academic-discipline mega-roots
like 'social sciences'. Keep the set small and concretely top-level.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on sys.path

import db
from embeddings import embed
from ingestion import backfill_topic_hierarchy

ROOTS = ["energy", "security", "economics", "technology", "diplomacy", "environment"]
_NIL = "00000000-0000-0000-0000-000000000000"


def clear_hierarchy() -> None:
    """Drop all parent edges and clear is_root so every topic is re-placed fresh."""
    db.client().table("topic_relations").delete().neq("child_id", _NIL).execute()
    db.client().table("topics").update({"is_root": False}).neq("id", _NIL).execute()
    print("cleared topic_relations + is_root flags")


def seed_roots() -> None:
    for name in ROOTS:
        existing = db.find_topic(name)
        if existing:
            db.set_topic_root(existing["id"])
            print(f"  root (existing): {name}")
        else:
            row = db.insert_topic(name=name, aliases=[name], embedding=embed(name))
            db.set_topic_root(row["id"])
            print(f"  root (created):  {name}")


def main() -> None:
    if "--replace" in sys.argv:
        clear_hierarchy()
    print("seeding roots ...")
    seed_roots()
    print("placing topics ...")
    print(backfill_topic_hierarchy())


if __name__ == "__main__":
    main()
