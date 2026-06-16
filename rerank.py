"""Local cross-encoder reranker (no API key required).

Scores (query, candidate) pairs with a cross-encoder so retrieval can reorder a
wide candidate pool by true relevance before truncating to the classifier cap.
Lazy-loaded and process-cached exactly like embeddings._get_model, so the ~80MB
model downloads once on first use and is reused thereafter.

Used only when ingestion.RERANK_EVIDENCE is on; the baseline path never imports it.
"""
from __future__ import annotations

from typing import Optional, Sequence

from sentence_transformers import CrossEncoder

RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"  # relevance scorer

_model: Optional[CrossEncoder] = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(RERANK_MODEL_NAME)
    return _model


def score(query: str, candidates: Sequence[str]) -> list[float]:
    """Relevance score of each candidate against the query (higher = more relevant)."""
    if not candidates:
        return []
    pairs = [(query, c) for c in candidates]
    return [float(s) for s in _get_model().predict(pairs)]
