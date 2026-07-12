"""Analysis of Competing Hypotheses — ask the graph a question (query-time).

    python ach.py "Is Veridia in a monetary tightening cycle?"
    python ach.py "..." --hyp "It is tightening" --hyp "It is easing"

The goal-directed counterpart to the bottom-up ingestion engine. Given a question
(and optional analyst-seeded hypotheses), it:

  1. enumerates MUTUALLY EXCLUSIVE competing hypotheses (seeds + model expansion);
  2. retrieves relevant raw evidence from the graph (semantic KNN over the question
     and each hypothesis) — grounded observations only, never derived inferences;
  3. builds the evidence x hypothesis matrix (each cell consistent/inconsistent/na);
  4. ranks hypotheses by LEAST DISCONFIRMED — fewest weighted inconsistencies, not
     most confirmations (the ACH insight that guards against confirmation bias),
     weighting evidence by source reliability;
  5. surfaces the single most diagnostic MISSING observation to seek next.

Deliberately DECISION-SUPPORT, not decision-making: it ranks and shows the evidence
and the gap; it never asserts a verdict. Read-time only, like streams/frontier —
nothing is materialized. Persisting questions/hypotheses over time is a follow-up.
"""
from __future__ import annotations

import sys
from typing import Optional

import db
from embeddings import embed
from llm_service import (
    classify_against_hypotheses,
    discriminating_evidence,
    generate_hypotheses,
)

EVIDENCE_MATCH_THRESHOLD = 0.45   # MiniLM-scale (matches ingestion.SIMILARITY_THRESHOLD)
EVIDENCE_PER_QUERY_K = 8          # KNN pulled per (question + each hypothesis)
EVIDENCE_CAP = 18                 # max evidence rows scored (bounds LLM calls)


def _retrieve_evidence(question: str, hypotheses: list[str]) -> list[dict]:
    """Raw-input nodes semantically near the question or any hypothesis (deduped)."""
    ids: list[str] = []
    seen: set[str] = set()
    for text in [question, *hypotheses]:
        for m in db.match_nodes(embed(text), EVIDENCE_MATCH_THRESHOLD, EVIDENCE_PER_QUERY_K):
            if m["id"] not in seen:
                seen.add(m["id"])
                ids.append(m["id"])
    rows = {r["id"]: r for r in db.nodes_by_ids(ids)}
    evidence = [rows[i] for i in ids if i in rows and rows[i].get("node_category") == "raw_input"]
    return evidence[:EVIDENCE_CAP]


def ask(question: str, seeds: Optional[list[str]] = None, max_hypotheses: int = 5) -> dict:
    """Run ACH over the graph for `question`. Returns the ranked map + evidence gap."""
    hyps = generate_hypotheses(question, seeds or [], max_hypotheses)
    if not hyps:
        return {"question": question, "error": "no hypotheses could be formed"}

    evidence = _retrieve_evidence(question, hyps)
    weights = db.source_weights([e["id"] for e in evidence])

    # Build the matrix and tally weighted inconsistencies (disconfirmation) per hypothesis.
    scores = [{"hypothesis": h, "disconfirmation": 0.0, "consistent": 0.0,
               "inconsistent_evidence": []} for h in hyps]
    matrix = []
    for e in evidence:
        w = weights.get(e["id"], 1.0)
        stances = classify_against_hypotheses(e["content"], hyps)
        diagnostic = len(set(stances)) > 1  # discriminates between hypotheses
        for i, stance in enumerate(stances):
            if stance == "inconsistent":
                scores[i]["disconfirmation"] += w
                if diagnostic:
                    scores[i]["inconsistent_evidence"].append(e["content"])
            elif stance == "consistent":
                scores[i]["consistent"] += w
        matrix.append({"id": e["id"], "content": e["content"], "weight": round(w, 2),
                       "diagnostic": diagnostic, "stances": stances})

    # Least-disconfirmed first; ties broken by more consistent support.
    ranked = sorted(scores, key=lambda s: (s["disconfirmation"], -s["consistent"]))
    for s in ranked:
        s["disconfirmation"] = round(s["disconfirmation"], 2)
        s["consistent"] = round(s["consistent"], 2)

    gap = None
    if len(ranked) >= 2:
        gap = discriminating_evidence(question, ranked[0]["hypothesis"], ranked[1]["hypothesis"])

    return {
        "question": question,
        "leading_hypothesis": ranked[0]["hypothesis"],
        "hypotheses_ranked": ranked,
        "evidence_gap": gap,
        "evidence_count": len(evidence),
        "matrix": matrix,
        "note": ("Decision support, not a verdict: 'leading' = least-disconfirmed by the "
                 "retrieved evidence, not proven. Weigh the evidence gap before concluding."),
    }


def _print(result: dict) -> None:
    print(f"\nQUESTION: {result['question']}")
    if result.get("error"):
        print("  " + result["error"]); return
    print(f"\nHYPOTHESES (least-disconfirmed first, over {result['evidence_count']} evidence items):")
    for i, s in enumerate(result["hypotheses_ranked"]):
        lead = "  <- leading" if i == 0 else ""
        print(f"  [{i}] disconfirm={s['disconfirmation']:<5} support={s['consistent']:<5} {s['hypothesis']}{lead}")
        for ev in s["inconsistent_evidence"][:2]:
            print(f"        x {ev[:82]}")
    gap = result.get("evidence_gap") or {}
    if gap.get("evidence_to_seek"):
        print(f"\nMOST DIAGNOSTIC GAP — seek: {gap['evidence_to_seek']}")
        print(f"    -> favors #1: {gap.get('would_favor_h1','')[:80]}")
        print(f"    -> favors #2: {gap.get('would_favor_h2','')[:80]}")
    print(f"\n{result['note']}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    seeds = [sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--hyp"]
    if not args:
        print('usage: python ach.py "<question>" [--hyp "<hypothesis>" ...]')
        raise SystemExit(1)
    _print(ask(args[0], seeds or None))
