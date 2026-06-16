"""Quick inspection of what the seed produced."""
import db

print("=== TOPICS ===")
topics = db.client().table("topics").select("name,aliases").order("name").execute().data
for t in topics:
    aliases = [a for a in (t.get("aliases") or []) if a != t["name"]]
    suffix = f"  aliases: {aliases}" if aliases else ""
    print(f"  {t['name']}{suffix}")

print("\n=== ENTITIES ===")
entities = db.client().table("entities").select("name,aliases").order("name").execute().data
for e in entities:
    aliases = [a for a in (e.get("aliases") or []) if a != e["name"]]
    suffix = f"  aliases: {aliases}" if aliases else ""
    print(f"  {e['name']}{suffix}")

print("\n=== INFERENCE NODES (top 20 by confidence) ===")
inferences = (
    db.client()
    .table("nodes")
    .select("id,confidence,content")
    .eq("node_category", "inference")
    .order("confidence", desc=True)
    .limit(80)
    .execute()
    .data
)
meta_rows = db.client().table("inference_meta").select("*").execute().data
meta = {m["node_id"]: m for m in meta_rows}

status_counts: dict[str, int] = {}
for n in inferences:
    m = meta.get(n["id"], {})
    status = m.get("status", "?")
    status_counts[status] = status_counts.get(status, 0) + 1
    cov = m.get("coverage")
    base = m.get("base_confidence")
    sup = len(m.get("support_node_ids") or [])
    defs = len(m.get("defeater_node_ids") or [])
    conv = len(m.get("converged_with") or [])
    cov_s = f"{cov:.2f}" if isinstance(cov, (int, float)) else "?"
    base_s = f"{base:.2f}" if isinstance(base, (int, float)) else "?"
    flags = f"sup={sup} def={defs}" + (f" conv={conv}" if conv else "")
    print(
        f"  [{status:11}] cov={cov_s} base={base_s} -> conf={n['confidence']:.2f} "
        f"| {flags}\n      {n['content'][:88]}"
    )
print(f"\n  status tally: {status_counts}")

print("\n=== CONVERGENCE (independently re-derived inferences) ===")
conv_edges = (
    db.client().table("edges").select("source_id,target_id")
    .eq("edge_type", "converges_with").execute().data or []
)
content_by_id = {n["id"]: n["content"] for n in inferences}
if not conv_edges:
    print("  (none)")
for e in conv_edges:
    a = content_by_id.get(e["source_id"], "?")
    b = content_by_id.get(e["target_id"], "?")
    print(f"  - {a[:72]}\n    <=> {b[:72]}")

print("\n=== STREAM: United States <-> Iran ===")
rows = db.stream_between_names("United States", "Iran")
print(f"  {len(rows)} nodes")
for r in rows[:8]:
    print(f"  [{r['node_kind']}] {r['content'][:80]}")

print("\n=== STREAM: Russia <-> European Union ===")
rows = db.stream_between_names("Russia", "European Union")
print(f"  {len(rows)} nodes")
for r in rows[:8]:
    print(f"  [{r['node_kind']}] {r['content'][:80]}")

print("\n=== CROSS-DOMAIN CHECK: nodes under BOTH 'energy' and 'security' (DAG rollup) ===")
# Uses nodes_under_topic (root + all descendants), so e.g. 'security' aggregates
# 'military conflict', 'regional security', etc. — the payoff of the topic DAG.
energy_topic = db.find_topic("energy") or db.find_topic("oil exports")
sec_topic    = db.find_topic("security") or db.find_topic("military conflict")
if energy_topic and sec_topic:
    energy_ids = {n["id"] for n in db.nodes_under_topic(energy_topic["id"], 500)}
    sec_ids    = {n["id"] for n in db.nodes_under_topic(sec_topic["id"], 500)}
    overlap = energy_ids & sec_ids
    print(f"  energy rollup: {energy_topic['name']} ({len(energy_ids)} nodes)")
    print(f"  security rollup: {sec_topic['name']} ({len(sec_ids)} nodes)")
    print(f"  overlap: {len(overlap)} nodes under both")
    if overlap:
        rows = db.client().table("nodes").select("content").in_("id", list(overlap)).execute().data
        for r in rows:
            print(f"    > {r['content'][:90]}")
else:
    print("  Could not find expected topic names — listing all topics above to cross-check")
