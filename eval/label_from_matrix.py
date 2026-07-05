"""Fill labels.jsonl `truth` from the seed_eval benchmark design.

    python eval/label_from_matrix.py

ONLY valid for the seed_eval diagnostic corpus. The scenarios are engineered and
entity-disjoint, so each inference's scenario — and with it the warranted truth
fixed by the benchmark design (docs/EVAL_SEED_NOTES.md expectation matrix) — is
recoverable from its content. This is benchmark ground truth, not pipeline output,
so scoring against it is NOT circular.

Caveat: the matrix truth targets each scenario's DESIGNED inference; adjacent
inferences the engine also formed inherit the scenario label, which can mislabel
edge cases. Rows where the independent LLM judge (truth_suggested) disagrees with
the matrix are printed for human review — the matrix label still wins, but the
list is the audit trail. Do not run this against real-news labels.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LABELS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "labels.jsonl")

# Scenario keyword -> designed truth, from the EVAL_SEED_NOTES expectation matrix.
# A/C: "false or unverifiable" in the notes -> "false" (both score identically for
# corroborated-vs-true; "false" gives the precision probe teeth).
MATRIX = {
    "veridia": "false",        # A — tightening narrative is overtaken
    "marran": "true",          # B — true positive
    "pelora": "false",         # C — topical-vs-causal trap
    "dram": "false",           # C (currency mentions without 'Pelora')
    "khelas": "unverifiable",  # D — hints of preparation, not evidence of escalation
    "tovar": "unverifiable",   # E — pure speculation
    "anvaria": "true",         # F — warranted & true but sparse (recall probe)
    "coastal": "true",         # F
    "sandar": "true",          # G — true, densely evidenced
}


def main() -> None:
    if not os.path.exists(LABELS_PATH):
        print(f"{LABELS_PATH} not found — run `python eval/label_dump.py` first.")
        raise SystemExit(1)

    rows = []
    with open(LABELS_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    labeled, unmatched, disagreements = 0, [], []
    for r in rows:
        text = (r.get("content") or "").lower()
        truth = next((t for kw, t in MATRIX.items() if kw in text), None)
        if truth is None:
            unmatched.append(r)
            continue
        r["truth"] = truth
        labeled += 1
        suggested = (r.get("truth_suggested") or "").strip()
        if suggested and suggested != truth:
            disagreements.append((truth, suggested, r.get("content", "")[:70]))

    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"labeled {labeled}/{len(rows)} rows from the benchmark matrix "
          f"({len(unmatched)} unmatched left blank)")
    for r in unmatched:
        print(f"  [unmatched] {r.get('content', '')[:80]}")
    if disagreements:
        print(f"\n{len(disagreements)} rows where the LLM judge disagrees with the "
              f"matrix (matrix wins; review these):")
        for truth, suggested, content in disagreements:
            print(f"  matrix={truth:<12} judge={suggested:<12} {content}")


if __name__ == "__main__":
    main()
