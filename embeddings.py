"""Embedding helper (local, no API key required).

Uses a local sentence-transformers model and zero-pads the output to 1536
dimensions so it drops into the existing vector(1536) schema unchanged.

Why padding: the `embedding` columns, the ivfflat index, and the match_nodes()
function are all declared vector(1536) (originally sized for OpenAI
text-embedding-3-small). all-MiniLM-L6-v2 produces 384-dim vectors; appending
zeros to reach 1536 is lossless for cosine similarity — zeros don't change dot
products or L2 norms — so no schema migration is required.

Caveat: every vector stored in the DB must come from the same model for
similarity to be meaningful. If you switch models (or back to OpenAI), re-embed
all existing rows.
"""
from __future__ import annotations

from typing import Optional, Sequence

from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dim, fast, ~80MB download on first use
NATIVE_DIM = 384
DIMENSIONS = 1536  # padded width — matches schema.sql vector(1536)

_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _pad(vec: list[float]) -> list[float]:
    """Right-pad a native embedding with zeros to DIMENSIONS (lossless for cosine)."""
    if len(vec) >= DIMENSIONS:
        return vec[:DIMENSIONS]
    return vec + [0.0] * (DIMENSIONS - len(vec))


def embed(text: str) -> list[float]:
    """Return a 1536-dim embedding for a single string."""
    vec = _get_model().encode(text, normalize_embeddings=True).tolist()
    return _pad(vec)


def embed_batch(texts: Sequence[str]) -> list[list[float]]:
    """Return 1536-dim embeddings for many strings in one batch."""
    vecs = _get_model().encode(list(texts), normalize_embeddings=True)
    return [_pad(v.tolist()) for v in vecs]
