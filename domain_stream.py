"""Continuous, domain-scoped ingestion driver (Phase 5) — makes the graph live.

    python domain_stream.py --once               # one poll cycle over the default queries
    python domain_stream.py --once --page-size 6
    python domain_stream.py --query "Iran oil"   # scope to specific queries (repeatable)

Each cycle: fetch fresh articles for the domain's query set -> skip everything
already ingested (source_url idempotency) -> ingest the new ones (real
publishedAt as event_date; source weight when ENCELADUS_SOURCE_WEIGHTS=1) ->
flush the inference engine, which also runs belief revision, so new evidence
both creates conclusions and RE-OPENS existing verdicts. The frontier endpoints
then reflect the updated map.

Scheduling is external by design (cron / Task Scheduler / the /ingest/poll
endpoint / an agent loop): the driver is a single idempotent cycle.
"""
from __future__ import annotations

import os
import sys

import requests
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import db
import ingestion
import sources
from ingestion import ingest_text, run_inference_batch
from seeds import seed_news  # cleaning + aggregator filtering (namespace package)

load_dotenv()

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

NEWSAPI_URL = "https://newsapi.org/v2/everything"

# Default domain scope — mirrors the seed snapshot's themes. Override with --query.
DEFAULT_QUERIES = [
    "Iran nuclear deal", "Strait of Hormuz shipping", "Russia oil exports sanctions",
    "China semiconductor export controls", "OPEC oil production",
    "European natural gas prices", "electric vehicle battery supply",
]


def _fetch(query: str, api_key: str, page_size: int) -> list[dict]:
    resp = requests.get(
        NEWSAPI_URL,
        params={"apiKey": api_key, "q": query, "pageSize": page_size,
                "language": "en", "sortBy": "publishedAt"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("articles", [])


def poll(queries: list[str] | None = None, page_size: int = 8) -> dict:
    """One incremental cycle. Returns a summary dict (also printed by the CLI)."""
    api_key = os.environ.get("NEWSAPI_KEY")
    if not api_key:
        raise RuntimeError("NEWSAPI_KEY must be set in the environment (.env).")
    queries = queries or DEFAULT_QUERIES

    candidates: list[dict] = []
    seen_urls: set = set()
    for q in queries:
        try:
            articles = _fetch(q, api_key, page_size)
        except Exception as exc:
            print(f"  query {q!r} failed: {exc}")
            continue
        for a in articles:
            row = {
                "title": (a.get("title") or "").strip(),
                "description": (a.get("description") or "").strip(),
                "url": a.get("url"),
                "source": (a.get("source") or {}).get("name"),
                "published_at": a.get("publishedAt"),
            }
            if not row["title"] or row["title"] == "[Removed]" or not row["url"]:
                continue
            if row["url"] in seen_urls or not seed_news._valid(row):
                continue
            seen_urls.add(row["url"])
            candidates.append(row)

    already = db.existing_source_urls([c["url"] for c in candidates])
    fresh = [c for c in candidates if c["url"] not in already]

    ingested = 0
    for a in fresh:
        text = f"{a['title']}. {seed_news._clean_description(a['description'])}"
        weight = sources.weight_for(a.get("source")) if ingestion.USE_SOURCE_WEIGHTS else None
        try:
            ingest_text(text, a["url"], event_date=a.get("published_at"), source_weight=weight)
            ingested += 1
            print(f"  + {a['title'][:70]}")
        except Exception as exc:
            print(f"  ! failed: {exc}  ({a['title'][:44]})")

    batch = run_inference_batch(force=True) if ingested else {"skipped": 0}
    summary = {
        "fetched": len(candidates),
        "already_ingested": len(candidates) - len(fresh),
        "new_ingested": ingested,
        "inference": batch,
    }
    print(f"\ncycle: {summary}")
    return summary


if __name__ == "__main__":
    qs = [sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--query"]
    ps = int(sys.argv[sys.argv.index("--page-size") + 1]) if "--page-size" in sys.argv else 8
    poll(qs or None, ps)
