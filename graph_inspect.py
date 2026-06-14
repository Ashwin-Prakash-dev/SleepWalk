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
    .select("node_kind,actor,subject,confidence,content")
    .eq("node_category", "inference")
    .order("confidence", desc=True)
    .limit(20)
    .execute()
    .data
)
for n in inferences:
    print(f"  [{n['node_kind']}] conf={n['confidence']:.2f}  {n['content'][:90]}")

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

print("\n=== CROSS-DOMAIN CHECK: nodes in BOTH 'energy' and 'military' topics ===")
energy_topic = db.find_topic("energy") or db.find_topic("energy exports") or db.find_topic("oil exports")
mil_topic    = db.find_topic("military") or db.find_topic("military conflict") or db.find_topic("military action")
if energy_topic and mil_topic:
    energy_ids = {
        r["node_id"]
        for r in db.client().table("node_topics").select("node_id").eq("topic_id", energy_topic["id"]).execute().data
    }
    mil_ids = {
        r["node_id"]
        for r in db.client().table("node_topics").select("node_id").eq("topic_id", mil_topic["id"]).execute().data
    }
    overlap = energy_ids & mil_ids
    print(f"  energy topic: {energy_topic['name']} ({len(energy_ids)} nodes)")
    print(f"  military topic: {mil_topic['name']} ({len(mil_ids)} nodes)")
    print(f"  overlap: {len(overlap)} nodes in both")
    if overlap:
        rows = db.client().table("nodes").select("content").in_("id", list(overlap)).execute().data
        for r in rows:
            print(f"    > {r['content'][:90]}")
else:
    print("  Could not find expected topic names — listing all topics above to cross-check")
