"""LLM extraction and inference for the Enceladus knowledge graph.

Backed by two providers behind a small failover router (see `_complete_json`):
Groq (llama-3.3-70b-versatile, primary) and Google Gemini (gemini-2.0-flash,
secondary). The secondary exists purely to spread load and survive Groq's daily
token cap — calls fail over to the next provider on a rate-limit/quota error.

Entry points:

- extract_node(text, source_url=None) -> dict
    Pull one structured node out of a piece of raw text. Keeps the single primary
    `actor` field and additionally carries an `entities` list of {"name", "role"}
    objects plus a `domains` list — additive, so existing consumers are unaffected.

- run_inference(new_node, similar_nodes) -> list[dict]   [legacy, unused by pipeline]
    Compare a new node against similar stored nodes and surface relationships.

- reason_pair / enumerate_alternatives / classify_evidence
    The three staged calls of the batched, adversarially-verified inference engine
    (ingestion.run_inference_batch).

Reads GROQ_API_KEY and GEMINI_API_KEY from the environment (see .env).
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Optional

from dotenv import load_dotenv
from groq import Groq

# Set ENCELADUS_LLM_DEBUG=1 to log which provider served each call (observability
# during bring-up; silent by default).
_DEBUG = bool(os.environ.get("ENCELADUS_LLM_DEBUG"))

load_dotenv()

# --- providers ---------------------------------------------------------------
GROQ_MODEL = "llama-3.3-70b-versatile"   # primary; strongest at strict JSON
GEMINI_MODEL = "gemini-2.5-flash"        # secondary; chosen over Gemma for JSON reliability

_groq_client: Optional[Groq] = None
_gemini_client: Any = None


def client() -> Groq:
    """Lazily-created, process-wide Groq client (kept for backwards compat)."""
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq()
    return _groq_client


def _gemini() -> Any:
    """Lazily-created Google Gemini client (reads GEMINI_API_KEY/GOOGLE_API_KEY)."""
    global _gemini_client
    if _gemini_client is None:
        from google import genai

        _gemini_client = genai.Client()
    return _gemini_client

# --- prompts (verbatim) ------------------------------------------------------
EXTRACT_SYSTEM_PROMPT = (
    "You are a structured information extractor. Always respond with valid JSON "
    "only. No preamble, no explanation, no markdown."
)

EXTRACT_USER_PROMPT = """Extract structured information from the following text and return a JSON object with exactly these fields:
- actor: string or null (the entity making the statement or primarily involved)
- subject: string (2-4 word primary topic tag — the single most salient theme, e.g. 'missile capabilities', 'oil exports')
- domains: array of strings, each 2-4 words naming a distinct thematic dimension of the event (always include the subject's domain; add others only if clearly present — e.g. ["energy", "military conflict"] for an event where drone strikes cause oil export cuts). Minimum 1 element, typically 1-3.
- node_kind: one of exactly: fact, claim, position, event_announcement, prediction, denial, agreement
- content: string (one sentence summary of the core information)
- confidence: float between 0.0 and 1.0 based on source credibility and certainty of language
- expires_at: ISO date string or null (only for event_announcements with a clear future date)
- denies_claim: string or null (if node_kind is denial, briefly describe what claim is being denied)
- entities: array of objects, one per distinct entity (country, organization, or person) named in the text, each {"name": string, "role": one of exactly "actor", "target", "mentioned"}. The primary actor above MUST also appear here with role "actor"; use "target" for an entity the action is directed at, and "mentioned" for any other entity referenced.

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
def _strip_code_fences(text: str) -> str:
    """Remove a ```json ... ``` wrapper if the model added one despite instructions."""
    if text.startswith("```"):
        # Drop the opening fence line (``` or ```json) and the trailing fence.
        text = text.split("\n", 1)[1] if "\n" in text else ""
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


def _complete_groq(system: str, prompt: str, max_tokens: int) -> str:
    response = client().chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return (response.choices[0].message.content or "").strip()


def _complete_gemini(system: str, prompt: str, max_tokens: int) -> str:
    from google.genai import types

    # Disable "thinking" (gemini-2.5-flash enables it by default): thinking tokens
    # otherwise eat the max_output_tokens budget and truncate the JSON response.
    response = _gemini().models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=0,
            max_output_tokens=max_tokens,
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return (response.text or "").strip()


_PROVIDERS = {"groq": _complete_groq, "gemini": _complete_gemini}
_DEFAULT_ORDER = ("groq", "gemini")


def _provider_order(prefer: Optional[str]) -> tuple[str, ...]:
    """Provider try-order; `prefer` (if valid) is moved to the front."""
    if prefer in _PROVIDERS:
        return (prefer,) + tuple(p for p in _DEFAULT_ORDER if p != prefer)
    return _DEFAULT_ORDER


def _complete_json(
    system: str, prompt: str, *, max_tokens: int, prefer: Optional[str] = None
) -> Any:
    """Complete a JSON request, failing over across providers.

    For each provider in order: retry once on a JSON parse error, but on any
    provider-side error (rate-limit / quota / API failure) move on to the next
    provider. Raises only when every provider has been exhausted. `prefer` biases
    which provider is tried first (e.g. "gemini" for cheap/bulky Pass-1 calls,
    to spare the scarce Groq token budget for Pass-2 verification).
    """
    last_error: Optional[Exception] = None
    for name in _provider_order(prefer):
        complete = _PROVIDERS[name]
        for _ in range(2):  # one retry per provider, for transient JSON glitches
            try:
                text = complete(system, prompt, max_tokens)
                parsed = json.loads(_strip_code_fences(text))
                if _DEBUG:
                    print(f"[llm] served by {name}", file=sys.stderr)
                return parsed
            except json.JSONDecodeError as exc:
                last_error = exc
                continue  # retry the same provider
            except Exception as exc:  # rate-limit / quota / API error → next provider
                last_error = exc
                break
    raise ValueError(f"No provider returned valid JSON: {last_error}")

COREFERENCE_SYSTEM_PROMPT = (
    "You are an entity coreference resolver. "
    "Coreference means the same real-world referent: 'Russian forces' and 'Russia' are the "
    "same state actor; 'Russia' and 'Ukraine' are distinct states and must never be merged "
    "even when they appear together. "
    "Respond with valid JSON only: {\"match\": <integer> | null}"
)

COREFERENCE_USER_PROMPT = """Mention to resolve: "{mention}"
Source sentence: "{context}"

Candidate entities (0-indexed):
{candidates}

Does the mention refer to the same real-world entity as one of the candidates?
- Same referent → return the 0-based index. (e.g. "Russian forces" → "Russia", "US" → "United States", "Tehran" → "Iran", "Iran's foreign minister" → "Iran")
- Distinct entity, or not enough information → return null. (e.g. "Russia" when candidates include only "Ukraine")

Return JSON only: {{"match": <integer> | null}}"""


def _format_candidates(candidates: list[dict]) -> str:
    lines = []
    for i, c in enumerate(candidates):
        aliases = [a for a in (c.get("aliases") or []) if a != c["name"]]
        alias_str = f" (also: {', '.join(aliases[:4])})" if aliases else ""
        lines.append(f"[{i}] {c['name']}{alias_str}")
    return "\n".join(lines)


def resolve_entity_coreference(
    mention: str,
    context: str,
    candidates: list[dict],
) -> Optional[int]:
    """Return 0-based index of the candidate the mention coreferents with, or None.

    candidates: list of dicts with 'name', 'aliases', 'id' (match_entities output).
    Context is capped at 300 chars; max_tokens=50 keeps the call cheap.
    """
    if not candidates:
        return None
    prompt = COREFERENCE_USER_PROMPT.format(
        mention=mention,
        context=context[:300],
        candidates=_format_candidates(candidates),
    )
    try:
        result = _complete_json(COREFERENCE_SYSTEM_PROMPT, prompt, max_tokens=50)
        if isinstance(result, dict):
            match = result.get("match")
            if isinstance(match, int):
                return match
    except ValueError:
        pass
    return None


TOPIC_PARENT_SYSTEM_PROMPT = (
    "You organize topics into a hierarchy of broader domains. Given a narrow topic, "
    "name the single broader domain it is a kind of (an IS-A parent). Always respond "
    'with valid JSON only: {"parent": <string> | null}.'
)

TOPIC_PARENT_USER_PROMPT = """Narrow topic: "{topic}"

Existing broader domains (prefer one of these if it genuinely fits):
{domains}

Name the single broader domain that "{topic}" is a kind of — strictly MORE GENERAL than the topic itself (2-4 words). Examples: "chip exports" -> "semiconductors"; "semiconductors" -> "technology"; "naval deployment" -> "military"; "oil exports" -> "energy".

Rules:
- The parent must be broader than the topic, never a synonym or a rephrasing of it.
- If one of the existing domains above fits, return that exact string.
- If "{topic}" is already a broad top-level domain (e.g. "economics", "energy", "security"), return null.

Return JSON only: {{"parent": <string> | null}}"""


def classify_topic_parent(topic: str, existing_domains: list[str]) -> Optional[str]:
    """Return the broader domain `topic` is a kind of, or None if it's top-level.

    `existing_domains` (current root / high-level topic names) is offered to the
    model to bias convergence onto a shared vocabulary rather than inventing
    near-duplicate parents. Biased to Gemini to spare the scarce Groq token budget.
    """
    domains = "\n".join(f"- {d}" for d in existing_domains) or "(none yet)"
    prompt = TOPIC_PARENT_USER_PROMPT.format(topic=topic, domains=domains)
    try:
        result = _complete_json(
            TOPIC_PARENT_SYSTEM_PROMPT, prompt, max_tokens=60, prefer="gemini"
        )
    except ValueError:
        return None
    if isinstance(result, dict):
        parent = result.get("parent")
        if isinstance(parent, str):
            parent = parent.strip()
            if parent and parent.lower() != topic.strip().lower():
                return parent
    return None


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

    Returns the parsed JSON object, e.g.::

        {"actor": "United States", "subject": "Iran sanctions",
         "node_kind": "claim", "content": "...", "confidence": 0.9,
         "expires_at": null, "denies_claim": null,
         "entities": [{"name": "United States", "role": "actor"},
                      {"name": "Iran", "role": "target"}]}

    The top-level `actor` (primary actor) is preserved; `entities` is additive.
    When `source_url` is provided it is attached as `source_url` on the returned
    dict so the caller can persist provenance.
    """
    prompt = EXTRACT_USER_PROMPT.replace("{text}", text)
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


# --- batched inference: staged calls -----------------------------------------
def _as_list(result: Any, *keys: str) -> list[dict]:
    """Coerce an LLM result into a list, tolerating a {key: [...]} wrapper."""
    if isinstance(result, list):
        return result
    if isinstance(result, dict):
        for key in keys:
            value = result.get(key)
            if isinstance(value, list):
                return value
    return []


REASON_PAIR_SYSTEM_PROMPT = (
    "You are a reasoning engine. Given two events, state the single strongest NEW "
    "logical inference that follows from combining them. Always respond with valid "
    "JSON only. No preamble, no markdown."
)

REASON_PAIR_USER_PROMPT = """Two premises have been observed. State the single strongest NEW logical inference that follows from COMBINING them — a conclusion neither premise states on its own (a + b => c). It need NOT be a contradiction or agreement; it can be any causal link, implication, or consequence.

A premise tagged "derived inference" is a PRIOR CONCLUSION, not a direct observation: treat it as provisional, and let your confidence reflect that the chain is only as strong as its weakest link.

PREMISE A ({a_origin}):
actor={a_actor} | kind={a_kind} | subject={a_subject}
{a_content}

PREMISE B ({b_origin}):
actor={b_actor} | kind={b_kind} | subject={b_subject}
{b_content}

Return a JSON object:
- content: string (one clear sentence stating the inferred conclusion), or null if no meaningful new inference follows
- confidence: float 0.0 to 1.0 (how strongly the two premises jointly support this conclusion)
- reasoning: string (one sentence explaining the logical step)

If the premises are unrelated, or one merely restates the other, return {{"content": null}}."""


ALTERNATIVES_SYSTEM_PROMPT = (
    "You are an adversarial verifier. For a proposed inference, enumerate competing "
    "explanations that would account for the same observation. Always respond with "
    "valid JSON only. No preamble, no markdown."
)

ALTERNATIVES_USER_PROMPT = """A proposed inference has been made:
"{inference}"

Enumerate up to {max_alternatives} COMPETING explanations — alternative accounts that, if true, would explain the same underlying observation differently and thereby undercut this inference. For each alternative provide:
- explanation: string (one sentence stating the competing account)
- evidence_signature: string (what a news report SUPPORTING this alternative would say, phrased AS the content of such a report so it can be matched against a corpus)
- reportability: float 0.0 to 1.0 (if this alternative were true, how likely is it that evidence would appear in a news corpus — high for overt public actions, low for secret or private causes)

Return a JSON array of alternative objects. Return an empty array [] if there are no credible competing explanations."""


CLASSIFY_SYSTEM_PROMPT = (
    "You are an evidence classifier. Classify each retrieved item as supporting the "
    "inference, supporting a competing alternative, or irrelevant. Always respond "
    "with valid JSON only. No preamble, no markdown."
)

CLASSIFY_USER_PROMPT = """INFERENCE:
"{inference}"

COMPETING ALTERNATIVES (0-indexed):
{alternatives}

RETRIEVED NODES (0-indexed):
{nodes}

For each retrieved node, classify it relative to the inference and the alternatives:
- "supports_inference": the node is evidence FOR the inference.
- "supports_alternative": the node is evidence for one of the competing alternatives (a defeater); set alternative_index to that alternative's 0-based index.
- "irrelevant": the node supports neither the inference nor any alternative.

Return a JSON array with one object per retrieved node, each:
- index: integer (the node's 0-based index above)
- label: one of "supports_inference", "supports_alternative", "irrelevant"
- alternative_index: integer or null (required when label is "supports_alternative")"""


def _format_alternatives(alternatives: list[dict]) -> str:
    if not alternatives:
        return "(none)"
    lines = []
    for i, a in enumerate(alternatives):
        lines.append(f"[{i}] {a.get('explanation', '')}")
    return "\n".join(lines)


def _origin_label(node: dict) -> str:
    """Provenance tag for a premise: an observation vs a prior (derived) conclusion."""
    if node.get("node_category") == "inference":
        try:
            conf = float(node.get("confidence"))
        except (TypeError, ValueError):
            conf = 0.0
        return f"derived inference, prior conclusion at confidence {conf:.2f}"
    return "observed report"


REASON_PAIR_NOVELTY_CLAUSE = """

NOVELTY REQUIREMENT: the conclusion must introduce a NEW fact, cause, or consequence that NEITHER premise states, and that is not merely a paraphrase, summary, or generalization of either premise. If the only thing that follows is a restatement of a premise, return {{"content": null}}."""


def reason_pair(node_a: dict, node_b: dict, require_novel: bool = False) -> dict:
    """Pass 1: open-ended logical inference combining two premises (a + b => c).

    Returns {content, confidence, reasoning}, or {} if no meaningful inference
    follows. Biased to the Gemini backend to spare the scarce Groq token budget.
    When `require_novel` is set, the conclusion must go beyond restating either
    premise (used when chaining off a derived premise, to force synthesis).
    """
    prompt = REASON_PAIR_USER_PROMPT.format(
        a_origin=_origin_label(node_a),
        a_actor=node_a.get("actor") or "unknown",
        a_kind=node_a.get("node_kind") or "unknown",
        a_subject=node_a.get("subject") or "unknown",
        a_content=node_a.get("content", ""),
        b_origin=_origin_label(node_b),
        b_actor=node_b.get("actor") or "unknown",
        b_kind=node_b.get("node_kind") or "unknown",
        b_subject=node_b.get("subject") or "unknown",
        b_content=node_b.get("content", ""),
    )
    if require_novel:
        prompt += REASON_PAIR_NOVELTY_CLAUSE
    try:
        result = _complete_json(
            REASON_PAIR_SYSTEM_PROMPT, prompt, max_tokens=512, prefer="gemini"
        )
    except ValueError:
        return {}
    if isinstance(result, dict) and result.get("content"):
        return result
    return {}


def enumerate_alternatives(inference_content: str, max_alternatives: int = 3) -> list[dict]:
    """Pass 2a: competing explanations, each with evidence_signature + reportability."""
    prompt = ALTERNATIVES_USER_PROMPT.format(
        inference=inference_content, max_alternatives=max_alternatives
    )
    try:
        result = _complete_json(ALTERNATIVES_SYSTEM_PROMPT, prompt, max_tokens=1024)
    except ValueError:
        return []
    return _as_list(result, "alternatives", "results", "data")


def classify_evidence(
    inference_content: str, alternatives: list[dict], nodes: list[dict]
) -> list[dict]:
    """Pass 2c: label each retrieved node supports_inference/alternative/irrelevant."""
    if not nodes:
        return []
    prompt = CLASSIFY_USER_PROMPT.format(
        inference=inference_content,
        alternatives=_format_alternatives(alternatives),
        nodes=_format_similar_nodes(nodes),
    )
    try:
        result = _complete_json(CLASSIFY_SYSTEM_PROMPT, prompt, max_tokens=2048)
    except ValueError:
        return []
    return _as_list(result, "classifications", "results", "data")


# --- labeling aid (NOT part of the pipeline) ---------------------------------
JUDGE_SYSTEM_PROMPT = (
    "You are an independent fact-checking analyst. Judge whether a proposed "
    "conclusion is correct, given its premises and your own knowledge. You are NOT "
    "evaluating any system's verdict — only the conclusion itself. Always respond "
    "with valid JSON only. No preamble, no markdown."
)

JUDGE_USER_PROMPT = """A conclusion was inferred from the premises below.

PREMISES:
{premises}

CONCLUSION:
"{conclusion}"

Judge the CONCLUSION independently:
- "true": well-supported — it follows from the premises and/or matches what you know to be the case.
- "false": wrong, contradicted, or an unwarranted leap from the premises.
- "unverifiable": not enough information to judge it either way.

Return JSON only: {{"verdict": "true" | "false" | "unverifiable", "reason": "<one sentence>"}}"""


def judge_inference(conclusion: str, premises: list[str]) -> dict:
    """Independent true/false/unverifiable judgment of a conclusion given its premises.

    A labeling AID only (used by eval/label_dump.py --llm-judge) — uses the same
    failover router, and is NEVER a substitute for a human label. Returns
    {"verdict": str, "reason": str}, or {} on failure. Biased to Gemini to spare
    the Groq budget.
    """
    prem = "\n".join(f"- {p}" for p in premises) or "(none)"
    prompt = JUDGE_USER_PROMPT.format(premises=prem, conclusion=conclusion)
    try:
        result = _complete_json(JUDGE_SYSTEM_PROMPT, prompt, max_tokens=200, prefer="gemini")
    except ValueError:
        return {}
    if isinstance(result, dict) and result.get("verdict") in ("true", "false", "unverifiable"):
        return {"verdict": result["verdict"], "reason": str(result.get("reason", ""))}
    return {}
