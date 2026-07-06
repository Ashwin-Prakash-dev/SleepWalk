"""Evaluate the verification pipeline against human truth labels.

    python eval/run_eval.py             # evaluate the CURRENT config (env flags)
    python eval/run_eval.py --compare   # baseline / +rerank / +nli / +both table

Re-runs verification (Pass 2) over each labeled inference using the configured
evidence retrieval + classification path, then reports, against the human `truth`:
  - precision / recall / F1 of status=="corroborated" vs truth=="true"
  - a calibration table (confidence deciles: mean predicted vs empirical) + ECE
  - a 3x3 confusion matrix of predicted status vs truth

Deterministic over a fixed input set: rows are sorted by node_id, premises are
re-fetched from the (fixed) DB, and the LLM calls run at temperature 0 — so two
runs of the same config over the same labels are comparable.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import db          # noqa: E402
import ingestion   # noqa: E402

LABELS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "labels.jsonl")
TRUTH_VALUES = ("true", "false", "unverifiable")
STATUSES = ("corroborated", "contested", "unverified")

# Bug-fix comparison grid: (convergence_legacy, defeater_policy). Holds the other
# axis at its new default so each named change is isolated, plus the combined row.
CONFIGS = [
    ("legacy-convergence", True,  "strict"),
    ("fixed-convergence",  False, "strict"),
    ("strict-defeater",    False, "strict"),
    ("weighted-defeater",  False, "weighted"),
    ("fixed+weighted",     False, "weighted"),
]


def _load_labeled() -> list[dict]:
    if not os.path.exists(LABELS_PATH):
        return []
    rows = []
    with open(LABELS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            raw = (r.get("truth") or "").strip()
            if raw in TRUTH_VALUES:
                rows.append(r)
            elif raw:  # non-empty but not a valid label — still dropped, but say so
                print(f"[warn] dropping row with malformed truth {raw!r} (node {r.get('node_id')})")
    rows.sort(key=lambda r: r["node_id"])  # determinism over a fixed set
    return rows


def _base_conf(node_id: str) -> float:
    m = (db.client().table("inference_meta").select("base_confidence")
         .eq("node_id", node_id).limit(1).execute().data or [])
    if m and m[0].get("base_confidence") is not None:
        return float(m[0]["base_confidence"])
    return 0.6


def _predict(row: dict):
    """Re-run verification for one labeled inference under the current config."""
    premises = db.nodes_by_ids(db.derives_from_targets(row["node_id"]))
    if len(premises) < 2:
        return None  # need both premises to reconstruct the verification context
    node_a, node_b = premises[0], premises[1]
    verdict = ingestion._verify_inference(
        {"content": row["content"]}, node_a, node_b, _base_conf(row["node_id"])
    )
    # Mirror _persist_inference's finalization so the eval measures the confidence
    # actually stored (incl. the convergence step) — needed to see the Change-1 fix.
    emb = ingestion.embed(row["content"])
    source_roots = ingestion._derivation_roots(node_a, node_b)
    conf, _ = ingestion._convergence_confidence(verdict, node_a, node_b, emb, source_roots)
    return verdict["status"], float(conf)


def _evaluate(rows: list[dict]) -> list[tuple]:
    records = []  # (pred_status, pred_conf, truth)
    for r in rows:
        # A transient network/DB hiccup must not kill a long multi-call run:
        # retry the row twice, then skip it with a warning (n is reported per run).
        pred = None
        for attempt in range(3):
            try:
                pred = _predict(r)
                break
            except Exception as exc:
                if attempt == 2:
                    print(f"[warn] skipping row {r.get('node_id')} after 3 attempts: {exc}")
                else:
                    time.sleep(2 * (attempt + 1))
        if pred is not None:
            records.append((pred[0], pred[1], (r["truth"] or "").strip()))
    return records


def _prf(records):
    tp = sum(1 for s, _, t in records if s == "corroborated" and t == "true")
    fp = sum(1 for s, _, t in records if s == "corroborated" and t != "true")
    fn = sum(1 for s, _, t in records if s != "corroborated" and t == "true")
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def _calibration(records):
    bins = [[] for _ in range(10)]
    for _, c, t in records:
        bins[min(9, int(c * 10))].append((c, 1.0 if t == "true" else 0.0))
    n = len(records) or 1
    ece, table = 0.0, []
    for i, b in enumerate(bins):
        if not b:
            table.append((i / 10, (i + 1) / 10, 0, None, None))
            continue
        mean_pred = sum(c for c, _ in b) / len(b)
        emp_acc = sum(a for _, a in b) / len(b)
        ece += (len(b) / n) * abs(mean_pred - emp_acc)
        table.append((i / 10, (i + 1) / 10, len(b), mean_pred, emp_acc))
    return ece, table


def _confusion(records):
    cm = {s: {t: 0 for t in TRUTH_VALUES} for s in STATUSES}
    for s, _, t in records:
        if s in cm and t in cm[s]:
            cm[s][t] += 1
    return cm


def _config_label() -> str:
    return f"rerank={ingestion.RERANK_EVIDENCE} classifier={ingestion.USE_NLI_CLASSIFIER!r}"


def _report_single(records):
    p, r, f1 = _prf(records)
    ece, table = _calibration(records)
    cm = _confusion(records)
    print("=" * 64)
    print(f"EVAL  ({_config_label()})  n={len(records)}")
    print("-" * 64)
    print(f"corroborated-vs-true:  precision={p:.3f}  recall={r:.3f}  F1={f1:.3f}")
    print("\ncalibration (confidence decile -> mean predicted vs empirical P(true)):")
    print(f"  {'bin':>11} {'n':>4} {'mean_pred':>10} {'emp_acc':>9}")
    for lo, hi, cnt, mp, ea in table:
        if cnt:
            print(f"  [{lo:.1f},{hi:.1f}) {cnt:>4} {mp:>10.3f} {ea:>9.3f}")
    print(f"  ECE = {ece:.3f}")
    print("\nconfusion (rows=predicted status, cols=truth):")
    print(f"  {'':>13}" + "".join(f"{t:>14}" for t in TRUTH_VALUES))
    for s in STATUSES:
        print(f"  {s:>13}" + "".join(f"{cm[s][t]:>14}" for t in TRUTH_VALUES))
    print("=" * 64)


def _report_compare(rows):
    print("=" * 60)
    print(f"CONFIG COMPARISON  (n_labeled={len(rows)})")
    print("-" * 60)
    print(f"  {'config':>18} {'precision':>10} {'recall':>8} {'F1':>7} {'ECE':>7}")
    saved = (ingestion.CONVERGENCE_LEGACY, ingestion.DEFEATER_POLICY)
    try:
        for name, conv_legacy, defeater in CONFIGS:
            ingestion.CONVERGENCE_LEGACY, ingestion.DEFEATER_POLICY = conv_legacy, defeater
            records = _evaluate(rows)
            p, r, f1 = _prf(records)
            ece, _ = _calibration(records)
            print(f"  {name:>18} {p:>10.3f} {r:>8.3f} {f1:>7.3f} {ece:>7.3f}")
    finally:
        ingestion.CONVERGENCE_LEGACY, ingestion.DEFEATER_POLICY = saved
    print("=" * 60)


def main() -> None:
    rows = _load_labeled()
    if not rows:
        print(f"No labeled rows in {LABELS_PATH}.")
        print(f"Run `python eval/label_dump.py`, fill each row's `truth` "
              f"({'/'.join(TRUTH_VALUES)}), then re-run.")
        return
    if "--compare" in sys.argv:
        _report_compare(rows)
    else:
        _report_single(_evaluate(rows))


if __name__ == "__main__":
    main()
