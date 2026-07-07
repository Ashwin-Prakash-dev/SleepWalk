"""Shared article cleaning/filtering for news ingestion (seeds + domain stream).

Strips syndication boilerplate and drops aggregator digests so the extractor
sees clean single-event text. Used by seeds/seed_news.py and domain_stream.py.
"""
from __future__ import annotations

import re

# Sources that publish multi-headline digests rather than single-event articles.
AGGREGATOR_SOURCES = {"Slashdot.org", "Google News", "Yahoo Entertainment"}
MIN_DESC_CHARS = 30


def clean_description(desc: str) -> str:
    """Strip syndication boilerplate ('The post … appeared first on …') and cruft."""
    desc = re.split(r"\n+The post ", desc)[0]               # common WordPress footer
    desc = re.sub(r"\s*The post .*?appeared first on .*$", "", desc)  # if no newline
    desc = re.sub(r"\s*\[\+\d+ chars\]\s*$", "", desc)      # NewsAPI truncation marker
    return desc.strip().rstrip("…").strip()


def valid_article(row: dict) -> bool:
    """Reject aggregator digests and empty/near-empty descriptions."""
    if (row.get("source") or "") in AGGREGATOR_SOURCES:
        return False
    return len(clean_description(row.get("description", ""))) >= MIN_DESC_CHARS
