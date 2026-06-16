"""Local natural-language-inference scorer (no API key required).

For an (evidence, claim) pair, returns calibrated entailment / contradiction /
neutral probabilities from a cross-encoder NLI model. Lets evidence be classified
by entailment instead of (or alongside) the LLM classifier.

Lazy-loaded and process-cached like embeddings._get_model. Used only when
ingestion.USE_NLI_CLASSIFIER is "nli" or "both"; the LLM path never imports it.

Note: cross-encoder/nli-deberta-v3-base outputs three logits in the fixed order
[contradiction, entailment, neutral] (the model's documented label mapping); we
softmax them into probabilities.
"""
from __future__ import annotations

import math
from typing import Any, Optional

from sentence_transformers import CrossEncoder

NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"
_LABELS = ("contradiction", "entailment", "neutral")  # model's output order

_model: Optional[CrossEncoder] = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(NLI_MODEL_NAME)
    return _model


def _softmax(logits: Any) -> list[float]:
    vals = [float(x) for x in logits]
    m = max(vals)
    exps = [math.exp(v - m) for v in vals]
    total = sum(exps) or 1.0
    return [e / total for e in exps]


def entail_scores(premise: str, hypothesis: str) -> dict[str, float]:
    """P(label) for label in {entailment, contradiction, neutral}.

    'Does `premise` (the evidence) entail `hypothesis` (the inference)?'
    """
    logits = _get_model().predict([(premise, hypothesis)])[0]
    probs = _softmax(logits)
    return dict(zip(_LABELS, probs))
