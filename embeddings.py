"""Embedding helper.

Generates 1536-dimensional vectors with OpenAI's `text-embedding-3-small`,
which matches the vector(1536) columns in schema.sql. Swap the model here if
you change the schema's vector dimensionality.

Reads OPENAI_API_KEY from the environment (see .env).
"""
from __future__ import annotations

from typing import Sequence

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "text-embedding-3-small"  # 1536 dimensions
DIMENSIONS = 1536

_oai: OpenAI | None = None


def _openai() -> OpenAI:
    global _oai
    if _oai is None:
        _oai = OpenAI()  # picks up OPENAI_API_KEY from the environment
    return _oai


def embed(text: str) -> list[float]:
    """Return the embedding for a single string."""
    resp = _openai().embeddings.create(model=MODEL, input=text)
    return resp.data[0].embedding


def embed_batch(texts: Sequence[str]) -> list[list[float]]:
    """Return embeddings for many strings in one request."""
    resp = _openai().embeddings.create(model=MODEL, input=list(texts))
    # OpenAI preserves input order, but sort on index to be safe.
    return [item.embedding for item in sorted(resp.data, key=lambda d: d.index)]
