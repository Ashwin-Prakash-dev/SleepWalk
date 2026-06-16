"""Ingest the frozen real-news snapshot — deterministic and reproducible.

    python seed_news.py                    # ingest all (cleaned) articles
    python seed_news.py --reset            # wipe everything first, then ingest
    python seed_news.py --reset --limit 60 # balanced subset of ~60 across clusters

Reads seed_data/news_snapshot.jsonl (produced once by fetch_news_snapshot.py) and
feeds each article through the normal ingestion pipeline, carrying the article's
real publishedAt as the node's event_date so coverage / time-window / convergence
reason over actual dates. Curated topic roots are seeded first (as in seed_large)
so the topic DAG stays clean.

Light cleaning strips syndication boilerplate and drops aggregator digests so the
extractor sees clean single-event text.
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import OrderedDict

import db
import seed_large  # reuse reset()
import seed_roots
from ingestion import ingest_text, run_inference_batch

SNAPSHOT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "seed_data", "news_snapshot.jsonl"
)
# Sources that publish multi-headline digests rather than single-event articles.
AGGREGATOR_SOURCES = {"Slashdot.org", "Google News", "Yahoo Entertainment"}
MIN_DESC_CHARS = 30


def _clean_description(desc: str) -> str:
    """Strip syndication boilerplate ('The post … appeared first on …') and cruft."""
    desc = re.split(r"\n+The post ", desc)[0]               # common WordPress footer
    desc = re.sub(r"\s*The post .*?appeared first on .*$", "", desc)  # if no newline
    desc = re.sub(r"\s*\[\+\d+ chars\]\s*$", "", desc)      # NewsAPI truncation marker
    return desc.strip().rstrip("…").strip()


def _valid(row: dict) -> bool:
    if (row.get("source") or "") in AGGREGATOR_SOURCES:
        return False
    return len(_clean_description(row.get("description", ""))) >= MIN_DESC_CHARS


def _select_balanced(rows: list[dict], limit: int) -> list[dict]:
    """Round-robin across clusters (the `query` field) up to `limit`."""
    groups: OrderedDict[str, list[dict]] = OrderedDict()
    for r in rows:
        groups.setdefault(r.get("query", "?"), []).append(r)
    selected, idx = [], 0
    while len(selected) < limit:
        progressed = False
        for lst in groups.values():
            if idx < len(lst):
                selected.append(lst[idx])
                progressed = True
                if len(selected) >= limit:
                    break
        if not progressed:
            break
        idx += 1
    return selected


def _load_articles(limit: int | None) -> list[dict]:
    if not os.path.exists(SNAPSHOT_PATH):
        print(f"{SNAPSHOT_PATH} not found — run `python fetch_news_snapshot.py` first.")
        raise SystemExit(1)
    rows = []
    with open(SNAPSHOT_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows = [r for r in rows if _valid(r)]
    if limit:
        rows = _select_balanced(rows, limit)
    return rows


def main() -> None:
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    articles = _load_articles(limit)

    if "--reset" in sys.argv:
        seed_large.reset()

    print("seeding topic roots ...")
    seed_roots.seed_roots()

    print(f"ingesting {len(articles)} cleaned articles ...")
    for i, a in enumerate(articles, 1):
        text = f"{a['title']}. {_clean_description(a['description'])}"
        try:
            node_id = ingest_text(text, a.get("url"), event_date=a.get("published_at"))
            print(f"[{i}/{len(articles)}] {node_id}  {a['title'][:64]}")
        except Exception as exc:
            print(f"[{i}/{len(articles)}] FAILED: {exc}  ({a['title'][:44]})")

    print("\nflushing inference tail ...")
    print(run_inference_batch(force=True))

    c = db.client()
    nodes    = c.table("nodes").select("id", count="exact").execute().count
    raw      = c.table("nodes").select("id", count="exact").eq("node_category", "raw_input").execute().count
    inf      = c.table("nodes").select("id", count="exact").eq("node_category", "inference").execute().count
    entities = c.table("entities").select("id", count="exact").execute().count
    topics   = c.table("topics").select("id", count="exact").execute().count
    edges    = c.table("edges").select("id", count="exact").execute().count
    print(
        f"\ndone — entities: {entities} | topics: {topics} | "
        f"nodes: {nodes} (raw: {raw}, inference: {inf}) | edges: {edges}"
    )


if __name__ == "__main__":
    main()
