"""Seed the knowledge graph with sample headlines.

    python seed.py            # ingest the samples on top of existing data
    python seed.py --reset    # wipe entities/nodes/edges first, then ingest

The samples are clustered (Iran nuclear, Russia/Ukraine, China/Taiwan) with
shared actors and subjects plus a claim/denial pair, so the structural,
semantic, and inference edges all have something to connect.
"""
from __future__ import annotations

import sys

import db
from ingestion import ingest_text

SAMPLES: list[tuple[str, str]] = [
    # --- Iran nuclear cluster ---
    ("Iran announced it will resume nuclear talks with European powers in Geneva next month.",
     "https://example.com/iran-talks"),
    ("Iran's foreign minister said the country's nuclear program is entirely peaceful and denied any weapons ambitions.",
     "https://example.com/iran-denial"),
    ("The United States warned it would impose fresh sanctions on Iran if the nuclear negotiations fail.",
     "https://example.com/us-iran-sanctions"),
    ("Israel said it would not rule out military action against Iran's nuclear facilities.",
     "https://example.com/israel-iran"),
    ("The IAEA reported that Iran has increased its stockpile of enriched uranium beyond agreed limits.",
     "https://example.com/iaea-iran"),
    # --- Russia / Ukraine cluster (note the claim/denial pair) ---
    ("Russia announced a partial withdrawal of troops from the eastern front near Kharkiv.",
     "https://example.com/russia-withdrawal"),
    ("Ukraine's military denied that Russian forces had withdrawn, calling it a tactical repositioning.",
     "https://example.com/ukraine-denial"),
    ("The European Union pledged an additional 5 billion euros in military aid to Ukraine.",
     "https://example.com/eu-ukraine-aid"),
    # --- China / Taiwan cluster ---
    ("China conducted large-scale military drills around Taiwan following a foreign diplomatic visit.",
     "https://example.com/china-drills"),
    ("Taiwan's defense ministry said it detected 20 Chinese aircraft crossing the median line.",
     "https://example.com/taiwan-aircraft"),
    ("The United States reaffirmed its commitment to Taiwan's defense amid rising regional tensions.",
     "https://example.com/us-taiwan"),
]


def reset() -> None:
    """Delete all rows (edges cascade from nodes; entities deleted last)."""
    nil = "00000000-0000-0000-0000-000000000000"
    c = db.client()
    c.table("edges").delete().neq("id", nil).execute()
    c.table("nodes").delete().neq("id", nil).execute()
    c.table("entities").delete().neq("id", nil).execute()
    print("reset: cleared entities, nodes, edges")


def main() -> None:
    if "--reset" in sys.argv:
        reset()

    for i, (text, url) in enumerate(SAMPLES, 1):
        try:
            node_id = ingest_text(text, url)
            print(f"[{i}/{len(SAMPLES)}] {node_id}  {text[:70]}")
        except Exception as exc:
            print(f"[{i}/{len(SAMPLES)}] FAILED: {exc}  ({text[:50]})")

    c = db.client()
    nodes = c.table("nodes").select("id", count="exact").execute().count
    raw = c.table("nodes").select("id", count="exact").eq("node_category", "raw_input").execute().count
    inf = c.table("nodes").select("id", count="exact").eq("node_category", "inference").execute().count
    entities = c.table("entities").select("id", count="exact").execute().count
    edges = c.table("edges").select("id", count="exact").execute().count
    print(f"\ndone — entities: {entities} | nodes: {nodes} (raw: {raw}, inference: {inf}) | edges: {edges}")


if __name__ == "__main__":
    main()
