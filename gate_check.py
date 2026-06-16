"""Phase-1 forward-chaining gate: did derived premises chain in safely?

    python gate_check.py

Answers the four things the Phase-1 gate cares about (see
[[inference-forward-chaining-design]]):
  1. depth distribution of inference nodes (did any depth>=2 form?),
  2. each multi-hop inference and its premises,
  3. confidence monotonicity — no derived child more confident than a derived premise,
  4. convergence independence — no "self-echo" (converged pair sharing raw grounding).
Read-only.
"""
import db
from ingestion import INDEPENDENCE_MAX_OVERLAP, _jaccard

c = db.client()
infs = (
    c.table("nodes").select("id,content,confidence,depth")
    .eq("node_category", "inference").execute().data or []
)

# 1. depth distribution -------------------------------------------------------
dist: dict[int, int] = {}
for n in infs:
    d = int(n.get("depth") or 0)
    dist[d] = dist.get(d, 0) + 1
print("=== INFERENCE DEPTH DISTRIBUTION ===")
for d in sorted(dist):
    print(f"  depth {d}: {dist[d]}")

# 2 + 3. multi-hop inferences + confidence monotonicity -----------------------
print("\n=== MULTI-HOP INFERENCES (depth >= 2) + their premises ===")
multi = [n for n in infs if int(n.get("depth") or 0) >= 2]
violations = []
if not multi:
    print("  (none — no derived node was built on another inference yet)")
for n in multi:
    print(f"  [d{n['depth']} conf={n['confidence']:.2f}] {n['content'][:84]}")
    premise_ids = db.derives_from_targets(n["id"])
    prem = {r["id"]: r for r in db.nodes_by_ids(premise_ids)}
    for pid in premise_ids:
        p = prem.get(pid, {})
        cat = p.get("node_category", "?")
        pc = p.get("confidence")
        tag = "raw" if cat == "raw_input" else f"INFERENCE d{p.get('depth', 0)}"
        pcs = f"{pc:.2f}" if isinstance(pc, (int, float)) else "?"
        print(f"      <- [{tag} conf={pcs}] {p.get('content', '')[:70]}")
        if cat == "inference" and isinstance(pc, (int, float)) and n["confidence"] > pc + 1e-9:
            violations.append((n, p))

print("\n=== CONFIDENCE MONOTONICITY (derived premise must cap the child) ===")
if not violations:
    print("  OK — no derived child exceeds a derived premise's confidence")
for child, p in violations:
    print(f"  VIOLATION: child {child['confidence']:.2f} > premise {p['confidence']:.2f}")
    print(f"    child:   {child['content'][:70]}")
    print(f"    premise: {p['content'][:70]}")

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
