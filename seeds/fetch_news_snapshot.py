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
import re
import sys

import requests
from dotenv import load_dotenv

load_dotenv()

NEWSAPI_URL = "https://newsapi.org/v2/everything"
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seed_data", "news_snapshot.jsonl")

# Many DISTINCT-but-linked sub-topics across domains, so genuine multi-hop chains
# can form (e.g. pipeline strike -> oil exports -> gas prices) rather than one
# breaking story dominating. Cross-links are deliberate (oil/energy/sanctions/trade).
QUERIES = [
    "Iran nuclear deal", "Iran sanctions oil", "Strait of Hormuz shipping", "Israel Iran conflict",
    "Russia Ukraine war", "Russia oil exports sanctions", "European natural gas prices", "Nord Stream pipeline",
    "China Taiwan military", "China semiconductor export controls", "China rare earth minerals", "TSMC chip manufacturing",
    "OPEC oil production", "global oil prices", "US Federal Reserve interest rates", "inflation energy prices",
    "electric vehicle battery supply", "lithium critical minerals mining", "renewable energy solar wind", "LNG exports Europe",
]

# Lightweight stopwords so near-duplicate title detection keys on content words.
_STOP = {"the", "a", "an", "to", "of", "in", "on", "for", "and", "as", "is", "at",
         "with", "by", "from", "amid", "after", "over", "says", "new", "us", "u.s."}


def _title_key(title: str) -> frozenset:
    return frozenset(re.findall(r"[a-z0-9]+", title.lower())) - _STOP


def _near_dup(key: frozenset, kept: list[frozenset]) -> bool:
    """True if `key` token-overlaps an already-kept title (Jaccard > 0.6)."""
    for k in kept:
        union = len(key | k)
        if union and len(key & k) / union > 0.6:
            return True
    return False


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

    page_size = 12
    if "--page-size" in sys.argv:
        page_size = int(sys.argv[sys.argv.index("--page-size") + 1])
    # Duplicate ARTICLES are corroboration (they feed coverage), so keep them by
    # default. --dedupe-titles collapses same-event reports if you want breadth only.
    dedupe_titles = "--dedupe-titles" in sys.argv

    seen_urls: set[str] = set()
    kept_titles: list[frozenset] = []
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
            # Skip removed/empty articles and cross-query URL duplicates.
            if not title or title == "[Removed]" or not desc or not url or url in seen_urls:
                continue
            # Optionally skip near-duplicate articles (same event from multiple outlets).
            key = _title_key(title)
            if dedupe_titles and _near_dup(key, kept_titles):
                continue
            seen_urls.add(url)
            kept_titles.append(key)
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
