"""LLM extraction and inference for the Enceladus knowledge graph.

Two entry points, both backed by Claude via the Anthropic Python SDK:

- extract_node(text, source_url=None) -> dict
    Pull one structured node out of a piece of raw text.

- run_inference(new_node, similar_nodes) -> list[dict]
    Compare a new node against semantically similar stored nodes and surface
    logical relationships (contradiction, derives_from, supports, tension).

Model: claude-sonnet-4-6 (as requested). Reads ANTHROPIC_API_KEY from the
environment (see .env).
"""
from __future__ import annotations

import json
from typing import Any, Optional
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

...
MODEL = "llama-3.3-70b-versatile"
_client: Optional[Groq] = None

def client() -> Groq:
    global _client
    if _client is None:
        _client = Groq()
    return _client

# --- prompts (verbatim) ------------------------------------------------------
EXTRACT_SYSTEM_PROMPT = (
    "You are a structured information extractor. Always respond with valid JSON "
    "only. No preamble, no explanation, no markdown."
)

EXTRACT_USER_PROMPT = """Extract structured information from the following text and return a JSON object with exactly these fields:
- actor: string or null (the entity making the statement or primarily involved)
- subject: string (2-4 word topic tag, e.g. 'missile capabilities', 'World Cup hosting')
- node_kind: one of exactly: fact, claim, position, event_announcement, prediction, denial, agreement
- content: string (one sentence summary of the core information)
- confidence: float between 0.0 and 1.0 based on source credibility and certainty of language
- expires_at: ISO date string or null (only for event_announcements with a clear future date)
- denies_claim: string or null (if node_kind is denial, briefly describe what claim is being denied)

Text to extract from:
{text}"""

INFERENCE_SYSTEM_PROMPT = (
    "You are a geopolitical inference engine. Analyze logical relationships "
    "between events. Always respond with valid JSON only. No preamble, no markdown."
)

INFERENCE_USER_PROMPT = """A new event has been recorded. Compare it against the stored events below and identify logical inferences.

NEW EVENT:
Actor: {actor}
Kind: {node_kind}
Subject: {subject}
Content: {content}

STORED EVENTS (ordered by relevance):
{formatted_list_of_similar_nodes}

Return a JSON array of inference objects. Each object must have:
- inference_kind: one of: contradiction, derives_from, supports, tension
- content: string (one clear sentence stating the inference)
- confidence: float 0.0 to 1.0
- source_node_indices: array of integers (0-based indices into the stored events list that support this inference)
- reasoning: string (one sentence explaining why)

Only return inferences with confidence above 0.6. Return empty array if none are significant."""


# --- helpers -----------------------------------------------------------------
def _message_text(response: Any) -> str:
    """Concatenate the text blocks of a Messages API response."""
    return "".join(b.text for b in response.content if b.type == "text").strip()


def _strip_code_fences(text: str) -> str:
    """Remove a ```json ... ``` wrapper if the model added one despite instructions."""
    if text.startswith("```"):
        # Drop the opening fence line (``` or ```json) and the trailing fence.
        text = text.split("\n", 1)[1] if "\n" in text else ""
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


def _complete_json(system: str, prompt: str, *, max_tokens: int) -> Any:
    last_error: Optional[Exception] = None
    for _ in range(2):
        response = client().chat.completions.create(
            model=MODEL,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        try:
            text = response.choices[0].message.content.strip()
            return json.loads(_strip_code_fences(text))
        except json.JSONDecodeError as exc:
            last_error = exc
    raise ValueError(f"Did not return valid JSON after one retry: {last_error}")

def _format_similar_nodes(nodes: list[dict]) -> str:
    """Render stored nodes as an indexed list the model can reference by index."""
    if not nodes:
        return "(none)"
    lines = []
    for i, n in enumerate(nodes):
        actor = n.get("actor") or "unknown"
        kind = n.get("node_kind") or "unknown"
        subject = n.get("subject") or "unknown"
        content = n.get("content", "")
        lines.append(f"[{i}] actor={actor} | kind={kind} | subject={subject}\n    {content}")
    return "\n".join(lines)


# --- public API --------------------------------------------------------------
def extract_node(text: str, source_url: str = None) -> dict:
    """Extract one structured node from raw text.

    Returns the parsed JSON object. When `source_url` is provided it is attached
    as `source_url` on the returned dict so the caller can persist provenance.
    """
    prompt = EXTRACT_USER_PROMPT.format(text=text)
    node = _complete_json(
        EXTRACT_SYSTEM_PROMPT,
        prompt,
        max_tokens=1024,
    )
    if source_url is not None and isinstance(node, dict):
        node["source_url"] = source_url
    return node


def run_inference(new_node: dict, similar_nodes: list[dict]) -> list[dict]:
    """Infer logical relationships between a new node and similar stored nodes.

    Returns a list of inference objects (possibly empty).
    """
    prompt = INFERENCE_USER_PROMPT.format(
        actor=new_node.get("actor") or "unknown",
        node_kind=new_node.get("node_kind") or "unknown",
        subject=new_node.get("subject") or "unknown",
        content=new_node.get("content", ""),
        formatted_list_of_similar_nodes=_format_similar_nodes(similar_nodes),
    )
    result = _complete_json(
        INFERENCE_SYSTEM_PROMPT,
        prompt,
        max_tokens=4096,
    )
    if isinstance(result, list):
        return result
    # Tolerate an object wrapper like {"inferences": [...]}.
    if isinstance(result, dict):
        for key in ("inferences", "results", "data"):
            value = result.get(key)
            if isinstance(value, list):
                return value
    return []
