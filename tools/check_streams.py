import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on sys.path

import db

for a, b in [("United States", "Iran"), ("Russia", "Ukraine")]:
    rows = db.stream_between_names(a, b)
    print(f"\n=== {a}  <->  {b}   ({len(rows)} nodes) ===")
    for r in rows:
        print(f"  [{r.get('node_kind','?')}] {r.get('content','')}")

# if a pair is empty, it's likely a name mismatch — see what's actually stored:
print("\nentities in db:")
for e in db.client().table("entities").select("name").order("name").execute().data:
    print("  ", e["name"]) 