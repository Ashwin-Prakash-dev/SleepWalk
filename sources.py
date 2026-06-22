"""Source credibility weights for evidence handling.

A small, explicit map from news source name to a [0,1] reliability weight, used
(when ingestion.USE_SOURCE_WEIGHTS is on) so that coverage/corroboration from
reputable wires counts for more than churn from low-signal aggregators/blogs.

Deliberately simple and hand-maintained — no model, no API. Unknown sources get
DEFAULT_WEIGHT (moderate trust). Tune freely; this is a prior, not ground truth.
"""
from __future__ import annotations

from typing import Optional

DEFAULT_WEIGHT = 0.6

# Higher = more reliable. Names match NewsAPI `source.name` values where known.
SOURCE_WEIGHTS: dict[str, float] = {
    # Wires / high-reliability
    "Reuters": 1.0, "Associated Press": 1.0, "AP News": 1.0,
    "Bloomberg": 0.95, "Financial Times": 0.95, "The Wall Street Journal": 0.95,
    "BBC News": 0.95, "NPR": 0.9, "The Economist": 0.95,
    "The New York Times": 0.9, "The Washington Post": 0.9, "The Guardian": 0.85,
    "Nikkei": 0.9, "Al Jazeera English": 0.8, "CNBC": 0.85, "Politico": 0.85,
    # Mid
    "Business Insider": 0.6, "Forbes": 0.6, "Yahoo Entertainment": 0.5,
    # Low-signal / aggregators / niche blogs
    "Slashdot.org": 0.45, "Crypto Briefing": 0.35, "Antiwar.com": 0.4,
    "Biztoc.com": 0.35, "Globalsecurity.org": 0.5,
}


def weight_for(source: Optional[str]) -> float:
    """Reliability weight in [0,1] for a source name (DEFAULT_WEIGHT if unknown)."""
    if not source:
        return DEFAULT_WEIGHT
    return SOURCE_WEIGHTS.get(source.strip(), DEFAULT_WEIGHT)
