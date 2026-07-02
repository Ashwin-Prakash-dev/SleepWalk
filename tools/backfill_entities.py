"""Backfill entities.embedding for rows created before embedding-based resolution.

Embedding-based entity resolution (ingestion._resolve_entity, step 2) can only
match against entities that already have an embedding. Entities created by earlier
ingests have a NULL embedding; this script embeds each one's name so resolution
works on already-ingested data.

Idempotent: by default only rows with a NULL embedding are touched. Use --all to
re-embed every entity (e.g. after changing the embedding model).

Run:  python backfill_entities.py
      python backfill_entities.py --all
"""
from __future__ import annotations

import sys

import db
from embeddings import embed


def main() -> None:
    re_all = "--all" in sys.argv
    c = db.client()

    query = c.table("entities").select("id,name")
    if not re_all:
        query = query.is_("embedding", "null")
    rows = query.execute().data

    print(f"entities to embed: {len(rows)}")
    for i, row in enumerate(rows, 1):
        emb = embed(row["name"])
        c.table("entities").update({"embedding": emb}).eq("id", row["id"]).execute()
        print(f"[{i}/{len(rows)}] {row['name']}")
    print("done")


if __name__ == "__main__":
    main()
