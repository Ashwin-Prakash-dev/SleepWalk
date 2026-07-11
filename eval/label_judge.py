"""Per-inference labeling via the independent judge, with a matrix cross-check.

    python eval/label_judge.py            # set truth = judge label; report disagreements
    python eval/label_judge.py --review   # print only matrix-vs-judge disagreements

Motivation: label_from_matrix.py assigns ONE truth to every inference in a
scenario, which mislabels scenarios that yield mixed-truth inferences (e.g.
Veridia's true rate-cut inferences scored 'false'). That systematically
UNDER-states precision. This labels each inference individually with the
independent judge (a different mechanism than the pipeline: it reasons from
premises + knowledge, not retrieval + adversarial evidence) and records the
matrix label alongside so the two can be compared.

Honesty caveats:
- The judge is a PROXY, not gold: it shares an LLM substrate with the pipeline,
  so it can agree where both are blind. Treat judge-graded precision as
  directional and strictly better than blanket-matrix, not as ground truth.
- The matrix-vs-judge disagreements (printed) are the rows a human should
  adjudicate to reach a true gold number — usually a small set.

Requires labels.jsonl to already carry `truth_suggested` (run
`python eval/label_dump.py --llm-judge` first).
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LABELS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "labels.jsonl")
TRUTH_VALUES = ("true", "false", "unverifiable")

# Reuse the scenario keyword -> matrix truth map (single source of truth).
from label_from_matrix import MATRIX  # noqa: E402


def _matrix_label(content: str):
    t = content.lower()
    return next((v for kw, v in MATRIX.items() if kw in t), None)


def main() -> None:
    review_only = "--review" in sys.argv
    rows = [json.loads(l) for l in open(LABELS_PATH, encoding="utf-8") if l.strip()]

    judged = agree = disagree = 0
    disagreements = []
    for r in rows:
        judge = (r.get("truth_suggested") or "").strip()
        matrix = _matrix_label(r.get("content", ""))
        r["truth_matrix"] = matrix  # record for the audit trail
        if judge in TRUTH_VALUES:
            if not review_only:
                r["truth"] = judge          # per-inference independent label = scored truth
            judged += 1
            if matrix in TRUTH_VALUES:
                if matrix == judge:
                    agree += 1
                else:
                    disagree += 1
                    disagreements.append((matrix, judge, r.get("content", "")[:74]))

    if not review_only:
        with open(LABELS_PATH, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_cmp = agree + disagree
    print(f"judge-labeled {judged}/{len(rows)} inferences"
          + ("" if review_only else " -> written as scored `truth`"))
    if total_cmp:
        print(f"matrix vs judge (where both label): agree {agree}/{total_cmp} "
              f"= {agree/total_cmp:.0%}, disagree {disagree}")
    print(f"\n{len(disagreements)} disagreements (matrix -> judge) — the rows to hand-adjudicate for gold:")
    for matrix, judge, content in disagreements:
        print(f"  matrix={matrix:<12} judge={judge:<12} {content}")


if __name__ == "__main__":
    main()
