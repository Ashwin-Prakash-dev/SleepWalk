"""Dump existing inference nodes to eval/labels.jsonl for human truth-labeling.

    python eval/label_dump.py               # dump rows; `truth` left blank for humans
    python eval/label_dump.py --llm-judge   # also pre-fill a SUGGESTED truth to review

One row per inference: node_id, content, the derives_from premise contents, the
current inference_meta.status, current confidence, and an EMPTY `truth` field.
`truth` must be filled in by a human as one of {"true","false","unverifiable"} —
it is deliberately NOT auto-populated, because grading the pipeline against its
own output would make the evaluation circular.

--llm-judge adds a separate `truth_suggested` (+ reason) from an independent LLM
fact-check. It is an AID ONLY: run_eval.py reads `truth`, never `truth_suggested`,
so a row still counts only once a human copies/overrides the suggestion into
`truth`. Re-running this script preserves any `truth` already entered.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import db            # noqa: E402
import llm_service   # noqa: E402

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "labels.jsonl")
TRUTH_VALUES = ("true", "false", "unverifiable")


def _load_existing() -> dict[str, dict]:
    """Map node_id -> prior row, so human labels survive a re-dump."""
    if not os.path.exists(OUT_PATH):
        return {}
    out = {}
    with open(OUT_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                out[r.get("node_id")] = r
    return out


def main() -> None:
    judge_on = "--llm-judge" in sys.argv
    existing = _load_existing()

    infs = (
        db.client().table("nodes")
        .select("id,content,confidence")
        .eq("node_category", "inference")
        .order("id")
        .execute()
        .data or []
    )
    status_by_id = {
        m["node_id"]: m["status"]
        for m in (db.client().table("inference_meta")
                  .select("node_id,status").execute().data or [])
    }

    rows, preserved, judged = [], 0, 0
    for n in infs:
        prev = existing.get(n["id"], {})
        premises = [p.get("content", "") for p in db.nodes_by_ids(db.derives_from_targets(n["id"]))]

        suggested = prev.get("truth_suggested", "")
        reason = prev.get("truth_suggested_reason", "")
        if judge_on:
            j = llm_service.judge_inference(n.get("content", ""), premises)
            if j:
                suggested, reason = j["verdict"], j["reason"]
                judged += 1

        human_truth = (prev.get("truth") or "").strip()
        if human_truth:
            preserved += 1

        rows.append({
            "node_id": n["id"],
            "content": n.get("content", ""),
            "premises": premises,
            "status": status_by_id.get(n["id"]),
            "confidence": n.get("confidence"),
            "truth_suggested": suggested,          # LLM aid — NOT used by run_eval
            "truth_suggested_reason": reason,
            "truth": human_truth,                  # <-- HUMAN fills/confirms this
        })

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"wrote {len(rows)} inference rows to {OUT_PATH}  (preserved {preserved} human labels)")
    if judge_on:
        print(f"pre-filled {judged} `truth_suggested` values via independent LLM judge.")
        print("These are an AID ONLY and are NOT scored. Review each, then set `truth` yourself")
        print(f"to one of {TRUTH_VALUES} — run_eval.py reads `truth`, never `truth_suggested`.")
    else:
        print(f"ACTION REQUIRED: fill each row's empty `truth` with one of {TRUTH_VALUES}.")
        print("Leave `truth` blank to skip a row. Do NOT auto-populate it — labels must be")
        print("human-supplied, or the evaluation is circular (pipeline graded on its own output).")


if __name__ == "__main__":
    main()
