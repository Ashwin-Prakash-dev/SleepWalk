"""Phase-1 forward-chaining gate: did derived premises chain in safely?

    python gate_check.py

Answers the four things the Phase-1 gate cares about (see
[[inference-forward-chaining-design]]):
  1. depth distribution of inference nodes (did any depth>=2 form?),
  2. each multi-hop inference and its premises,
  3. confidence monotonicity, judged by the INDEPENDENCE-AWARE rule the engine
     uses: a child may exceed its weakest derived premise only when the lift is
     backed by evidence independent of that premise's grounding — otherwise it's
     a true violation,
  4. convergence independence — no "self-echo" (converged pair sharing raw grounding).
Read-only.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on sys.path

import db
from embeddings import embed
from ingestion import INDEPENDENCE_MAX_OVERLAP, MERGE_THRESHOLD, _cosine, _jaccard, _to_float

try:  # node content can contain non-cp1252 chars; keep prints from crashing on Windows
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

c = db.client()
infs = (
    c.table("nodes").select("id,content,confidence,depth")
    .eq("node_category", "inference").execute().data or []
)
by_id = {n["id"]: n for n in infs}
meta = {
    m["node_id"]: m
    for m in (
        c.table("inference_meta")
        .select("node_id,support_node_ids,converged_with").execute().data or []
    )
}

# 1. depth distribution -------------------------------------------------------
dist: dict[int, int] = {}
for n in infs:
    d = int(n.get("depth") or 0)
    dist[d] = dist.get(d, 0) + 1
print("=== INFERENCE DEPTH DISTRIBUTION ===")
for d in sorted(dist):
    print(f"  depth {d}: {dist[d]}")

# 2 + 3. multi-hop inferences + independence-aware monotonicity ----------------
print("\n=== MULTI-HOP INFERENCES (depth >= 2) + their premises ===")
multi = [n for n in infs if int(n.get("depth") or 0) >= 2]
violations, permitted_lifts = [], []
novel_multi = 0  # depth>=2 conclusions that are NOT near-restatements of a derived premise
if not multi:
    print("  (none — no derived node was built on another inference yet)")
for n in multi:
    print(f"  [d{n['depth']} conf={n['confidence']:.2f}] {n['content'][:84]}")
    premise_ids = db.derives_from_targets(n["id"])
    prem = {r["id"]: r for r in db.nodes_by_ids(premise_ids)}
    derived_premises = [prem[p] for p in premise_ids
                        if prem.get(p, {}).get("node_category") == "inference"]
    for pid in premise_ids:
        p = prem.get(pid, {})
        cat = p.get("node_category", "?")
        pc = p.get("confidence")
        tag = "raw" if cat == "raw_input" else f"INFERENCE d{p.get('depth', 0)}"
        pcs = f"{pc:.2f}" if isinstance(pc, (int, float)) else "?"
        print(f"      <- [{tag} conf={pcs}] {p.get('content', '')[:70]}")

    if not derived_premises:
        continue
    # Novel = the conclusion isn't a near-restatement of any derived premise.
    n_emb = embed(n["content"])
    max_sim = max(_cosine(n_emb, embed(p["content"])) for p in derived_premises)
    if max_sim < MERGE_THRESHOLD:
        novel_multi += 1
    premise_cap = min(_to_float(p.get("confidence"), 1.0) for p in derived_premises)
    if n["confidence"] <= premise_cap + 1e-9:
        continue
    # Child exceeds its weakest derived premise — was the lift independently earned?
    premise_roots: set[str] = set()
    for p in derived_premises:
        premise_roots.update(db.derivation_roots(p["id"]))
    m = meta.get(n["id"], {})
    support = m.get("support_node_ids") or []
    converged = m.get("converged_with") or []
    independent_support = any(s not in premise_roots for s in support)
    independent_conv = any(
        _jaccard(set(db.derivation_roots(cid)), premise_roots) <= INDEPENDENCE_MAX_OVERLAP
        for cid in converged
    )
    rec = (n, premise_cap, independent_support, independent_conv)
    (permitted_lifts if (independent_support or independent_conv) else violations).append(rec)

multi_with_derived = sum(
    1 for n in multi
    if any(prem.get("node_category") == "inference"
           for prem in db.nodes_by_ids(db.derives_from_targets(n["id"])))
)
print("\n=== NOVEL-DEPTH RATE (multi-hop that isn't a restatement) ===")
print(f"  {novel_multi} of {multi_with_derived} multi-hop conclusions are novel "
      f"(cosine < {MERGE_THRESHOLD} to every derived premise)")

print("\n=== CONFIDENCE MONOTONICITY (independence-aware) ===")
if not violations:
    print("  OK — every lift above a derived premise is backed by independent evidence")
for n, cap, _is, _ic in violations:
    print(f"  VIOLATION: child {n['confidence']:.2f} > premise {cap:.2f}, NO independent backing")
    print(f"    {n['content'][:72]}")
for n, cap, is_sup, is_conv in permitted_lifts:
    why = "independent raw support" if is_sup else "independent convergence"
    print(f"  permitted lift: {n['confidence']:.2f} > premise {cap:.2f} ({why})")
    print(f"    {n['content'][:72]}")

# 4. convergence independence -------------------------------------------------
print("\n=== CONVERGENCE INDEPENDENCE (no self-echo) ===")
conv = (
    c.table("edges").select("source_id,target_id")
    .eq("edge_type", "converges_with").execute().data or []
)
if not conv:
    print("  (no convergence edges)")
self_echo = 0
for e in conv:
    ra = set(db.derivation_roots(e["source_id"]))
    rb = set(db.derivation_roots(e["target_id"]))
    j = _jaccard(ra, rb)
    flag = "  <-- SELF-ECHO" if j > INDEPENDENCE_MAX_OVERLAP else ""
    self_echo += 1 if flag else 0
    print(f"  jaccard(raw roots)={j:.2f}{flag}")
if conv:
    print(f"  -> {self_echo} self-echo edge(s) (want 0)")
