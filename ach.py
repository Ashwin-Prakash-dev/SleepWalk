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

    # Rank ENGAGED hypotheses first (some evidence bears on them), then least-
    # disconfirmed, then most support. A hypothesis no evidence touches is UNASSESSED,
    # not surviving — without this guard a vague, unfalsifiable hypothesis wins by
    # attracting zero inconsistencies (the classic least-disconfirmed pitfall).
    for s in scores:
        s["assessed"] = (s["disconfirmation"] + s["consistent"]) > 0
    ranked = sorted(scores, key=lambda s: (not s["assessed"], s["disconfirmation"], -s["consistent"]))
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


# --- standing questions (v2): persistent, auto-updating, flip-alerting --------
QUESTION_MATCH_THRESHOLD = 0.50  # new node this close to a hypothesis -> its question updates
QUESTIONS_PER_BATCH = 3          # LLM budget: questions updated per ingest batch
NODES_PER_QUESTION = 5           # LLM budget: new evidence items scored per question per batch


def _rescore(question_id: str) -> tuple[Optional[str], Optional[dict], list[dict]]:
    """Recompute scores/ranks for a question from its stored matrix cells.

    Returns (old_leading_hypothesis_id, new_leading_hypothesis_row, ranked_rows).
    Ranking: engaged (assessed) hypotheses first, then least-disconfirmed, then
    most support — same rule as the read-time ask().
    """
    q = db.get_question(question_id)
    hyps = db.question_hypotheses(question_id)
    cells = db.hypothesis_evidence_rows([h["id"] for h in hyps])
    totals: dict = {h["id"]: {"disconfirmation": 0.0, "support": 0.0} for h in hyps}
    for c in cells:
        t = totals.get(c["hypothesis_id"])
        if t is None:
            continue
        w = float(c.get("weight") or 1.0)
        if c["stance"] == "inconsistent":
            t["disconfirmation"] += w
        elif c["stance"] == "consistent":
            t["support"] += w
    for h in hyps:
        t = totals[h["id"]]
        h["disconfirmation"], h["support"] = round(t["disconfirmation"], 2), round(t["support"], 2)
        h["assessed"] = (t["disconfirmation"] + t["support"]) > 0
    ranked = sorted(hyps, key=lambda h: (not h["assessed"], h["disconfirmation"], -h["support"]))
    for rank, h in enumerate(ranked):
        db.update_hypothesis_score(
            h["id"], disconfirmation=h["disconfirmation"], support=h["support"],
            assessed=h["assessed"], rank=rank,
        )
    new_leader = ranked[0] if ranked and ranked[0]["assessed"] else None
    return (q or {}).get("leading_hypothesis_id"), new_leader, ranked


def open_question(question: str, seeds: Optional[list[str]] = None, max_hypotheses: int = 5) -> dict:
    """Create a standing question: hypotheses + initial matrix, persisted.

    From then on, every ingest batch routes relevant new evidence to it and
    re-ranks; a change of leading hypothesis is logged as a 'leader_changed'
    event (the flip alert).
    """
    from datetime import datetime, timezone

    hyps = generate_hypotheses(question, seeds or [], max_hypotheses)
    if not hyps:
        return {"error": "no hypotheses could be formed", "question": question}
    q = db.insert_question(question, embed(question))
    hyp_rows = [db.insert_hypothesis(q["id"], h, embed(h)) for h in hyps]

    evidence = _retrieve_evidence(question, hyps)
    weights = db.source_weights([e["id"] for e in evidence])
    contents = [h["content"] for h in hyp_rows]
    for e in evidence:
        stances = classify_against_hypotheses(e["content"], contents)
        for h, stance in zip(hyp_rows, stances):
            db.upsert_hypothesis_evidence(h["id"], e["id"], stance, weights.get(e["id"], 1.0))

    _, leader, ranked = _rescore(q["id"])
    gap = None
    if len(ranked) >= 2 and leader:
        gap = discriminating_evidence(question, ranked[0]["content"], ranked[1]["content"])
    db.update_question_state(
        q["id"], leading_hypothesis_id=(leader or {}).get("id"), evidence_gap=gap,
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    db.insert_question_event(q["id"], "created", {"hypotheses": hyps, "evidence_scored": len(evidence)})
    return question_view(q["id"])


def update_question(question_id: str, new_nodes: list[dict]) -> dict:
    """Score new evidence into a question's matrix; re-rank; flip-alert on change."""
    from datetime import datetime, timezone

    q = db.get_question(question_id)
    if not q or q.get("status") != "open":
        return {"updated": False}
    hyp_rows = db.question_hypotheses(question_id)
    if not hyp_rows:
        return {"updated": False}
    seen = db.question_evidence_ids(question_id)
    fresh = [n for n in new_nodes if n["id"] not in seen][:NODES_PER_QUESTION]
    if not fresh:
        return {"updated": False}

    weights = db.source_weights([n["id"] for n in fresh])
    contents = [h["content"] for h in hyp_rows]
    for n in fresh:
        stances = classify_against_hypotheses(n["content"], contents)
        for h, stance in zip(hyp_rows, stances):
            db.upsert_hypothesis_evidence(h["id"], n["id"], stance, weights.get(n["id"], 1.0))
    db.insert_question_event(question_id, "evidence_added",
                             {"nodes": [n["content"][:120] for n in fresh]})

    old_leading_id, leader, ranked = _rescore(question_id)
    flipped = bool(leader) and leader["id"] != old_leading_id and old_leading_id is not None
    gap = q.get("evidence_gap")
    if flipped and len(ranked) >= 2:
        gap = discriminating_evidence(q["question"], ranked[0]["content"], ranked[1]["content"])
        old = next((h["content"] for h in hyp_rows if h["id"] == old_leading_id), "(none)")
        db.insert_question_event(question_id, "leader_changed", {
            "old": old, "new": leader["content"],
            "trigger_evidence": [n["content"][:120] for n in fresh],
        })
        print(f"[flip] {q['question'][:60]} -> now leading: {leader['content'][:70]}")
    db.update_question_state(
        question_id, leading_hypothesis_id=(leader or {}).get("id"), evidence_gap=gap,
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    return {"updated": True, "flipped": flipped, "evidence_scored": len(fresh)}


def update_affected_questions(new_nodes: list[dict]) -> dict:
    """Route a batch's new raw nodes to the standing questions they bear on.

    Budgeted (QUESTIONS_PER_BATCH / NODES_PER_QUESTION). Fail-safe: if the
    standing-question tables don't exist yet, degrades to a no-op with a warning.
    """
    try:
        per_question: dict = {}
        best_sim: dict = {}
        for n in new_nodes:
            emb = embed(n.get("content", ""))
            for m in db.match_hypotheses(emb, QUESTION_MATCH_THRESHOLD, 5):
                qid = m["question_id"]
                per_question.setdefault(qid, []).append(n)
                best_sim[qid] = max(best_sim.get(qid, 0.0), float(m.get("similarity") or 0))
        updated = flips = 0
        for qid in sorted(per_question, key=lambda k: -best_sim[k])[:QUESTIONS_PER_BATCH]:
            res = update_question(qid, per_question[qid])
            updated += bool(res.get("updated"))
            flips += bool(res.get("flipped"))
        return {"updated": updated, "flips": flips}
    except Exception as exc:  # tables missing / transient DB failure -> no-op
        print(f"[warn] standing-question update skipped: {str(exc)[:90]}")
        return {"updated": 0, "flips": 0, "disabled": True}


def question_view(question_id: str) -> Optional[dict]:
    q = db.get_question(question_id)
    if not q:
        return None
    hyps = sorted(db.question_hypotheses(question_id),
                  key=lambda h: h.get("rank") if h.get("rank") is not None else 99)
    return {
        "id": q["id"],
        "question": q["question"],
        "status": q["status"],
        "leading_hypothesis": next((h["content"] for h in hyps if h["id"] == q.get("leading_hypothesis_id")), None),
        "hypotheses_ranked": [
            {"content": h["content"], "disconfirmation": h["disconfirmation"],
             "support": h["support"], "assessed": h["assessed"]}
            for h in hyps
        ],
        "evidence_gap": q.get("evidence_gap"),
        "events": db.question_events(question_id, 10),
        "updated_at": q.get("updated_at"),
        "note": ("Decision support, not a verdict: 'leading' = least-disconfirmed by "
                 "evidence seen so far; watch leader_changed events as the stream updates."),
    }


def _print(result: dict) -> None:
    print(f"\nQUESTION: {result['question']}")
    if result.get("error"):
        print("  " + result["error"]); return
    print(f"\nHYPOTHESES (least-disconfirmed first, over {result['evidence_count']} evidence items):")
    for i, s in enumerate(result["hypotheses_ranked"]):
        tag = "  <- leading" if i == 0 and s.get("assessed") else ("  (unassessed)" if not s.get("assessed") else "")
        print(f"  [{i}] disconfirm={s['disconfirmation']:<5} support={s['consistent']:<5} {s['hypothesis']}{tag}")
        for ev in s["inconsistent_evidence"][:2]:
            print(f"        x {ev[:82]}")
    gap = result.get("evidence_gap") or {}
    if gap.get("evidence_to_seek"):
        print(f"\nMOST DIAGNOSTIC GAP — seek: {gap['evidence_to_seek']}")
        print(f"    -> favors #1: {gap.get('would_favor_h1','')[:80]}")
        print(f"    -> favors #2: {gap.get('would_favor_h2','')[:80]}")
    print(f"\n{result['note']}")


def _print_view(view: Optional[dict]) -> None:
    if not view:
        print("question not found")
        return
    print(f"\nSTANDING QUESTION [{view['status']}]: {view['question']}")
    print(f"  leading: {view.get('leading_hypothesis') or '(none assessed yet)'}")
    for h in view["hypotheses_ranked"]:
        tag = "" if h["assessed"] else "  (unassessed)"
        print(f"    disconfirm={h['disconfirmation']:<5} support={h['support']:<5} {h['content'][:78]}{tag}")
    gap = view.get("evidence_gap") or {}
    if gap.get("evidence_to_seek"):
        print(f"  gap — seek: {gap['evidence_to_seek'][:100]}")
    for e in view.get("events", [])[:5]:
        print(f"  event: {e['event_type']} @ {e['created_at'][:19]}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    argv = sys.argv[1:]
    seeds = [argv[i + 1] for i, a in enumerate(argv) if a == "--hyp" and i + 1 < len(argv)]
    flag_values = {i + 1 for i, a in enumerate(argv) if a == "--hyp"}
    pos = [a for i, a in enumerate(argv) if not a.startswith("--") and i not in flag_values]

    if "--list" in argv:
        for q in db.list_questions():
            lead = q.get("leading_hypothesis_id") or "-"
            print(f"  [{q['status']}] {q['id'][:8]}  {q['question'][:70]}")
        raise SystemExit(0)
    if not pos:
        print('usage: python ach.py "<question>" [--hyp "<h>" ...] [--open] | --list')
        raise SystemExit(1)
    if "--open" in argv:
        _print_view(open_question(pos[0], seeds or None))
    else:
        _print(ask(pos[0], seeds or None))
