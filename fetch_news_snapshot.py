"""Fetch a real-news snapshot ONCE and freeze it to a committed file.

    python fetch_news_snapshot.py            # fetch + write seed_data/news_snapshot.jsonl
    python fetch_news_snapshot.py --page-size 30

This is the only non-deterministic step: it hits NewsAPI live (needs NEWSAPI_KEY).
The resulting JSONL is committed and becomes the FIXED corpus that seed_news.py
ingests deterministically — so every later ingest / eval run is reproducible even
though the underlying news feed keeps changing.

Queries are themed to stay dense within clusters (so corroboration/convergence
have material), mirroring the synthetic seed's themes but with real events.
"""
from __future__ import annotations

import json
import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()

NEWSAPI_URL = "https://newsapi.org/v2/everything"
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seed_data", "news_snapshot.jsonl")

# Themed queries — dense clusters with deliberate cross-links (oil/energy/sanctions).
QUERIES = [
    "Iran nuclear talks",
    "Russia Ukraine war",
    "China Taiwan tensions",
    "semiconductor export controls",
    "OPEC oil prices",
    "Europe natural gas energy",
    "electric vehicle critical minerals",
    "climate renewable energy transition",
]


def _fetch(query: str, api_key: str, page_size: int) -> list[dict]:
    resp = requests.get(
        NEWSAPI_URL,
        params={
            "apiKey": api_key,
            "q": query,
            "pageSize": page_size,
            "language": "en",
            "sortBy": "publishedAt",
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("articles", [])


def main() -> None:
    api_key = os.environ.get("NEWSAPI_KEY")
    if not api_key:
        print("NEWSAPI_KEY must be set in the environment (.env). Aborting.")
        raise SystemExit(1)

    page_size = 20
    if "--page-size" in sys.argv:
        page_size = int(sys.argv[sys.argv.index("--page-size") + 1])

    seen_urls: set[str] = set()
    rows: list[dict] = []
    for q in QUERIES:
        try:
            articles = _fetch(q, api_key, page_size)
        except Exception as exc:
            print(f"  query {q!r} failed: {exc}")
            continue
        kept = 0
        for a in articles:
            title = (a.get("title") or "").strip()
            desc = (a.get("description") or "").strip()
            url = a.get("url")
            # Skip removed/empty articles and cross-query duplicates.
            if not title or title == "[Removed]" or not desc or not url or url in seen_urls:
                continue
            seen_urls.add(url)
            rows.append({
                "title": title,
                "description": desc,
                "url": url,
                "source": (a.get("source") or {}).get("name"),
                "published_at": a.get("publishedAt"),
                "query": q,
            })
            kept += 1
        print(f"  {q!r}: kept {kept}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nwrote {len(rows)} unique articles to {OUT_PATH}")
    print("Commit this file — it is the frozen corpus. Then run: python seed_news.py --reset")


if __name__ == "__main__":
    main()
