"""End-to-end smoke test for the Enceladus schema.

Prerequisites:
  1. schema.sql has been applied to your Supabase project.
  2. .env contains SUPABASE_URL, SUPABASE_KEY, and OPENAI_API_KEY.

Run:  python example.py
"""
from __future__ import annotations

import db
from embeddings import embed


def main() -> None:
    # 1. An entity the nodes can reference.
    entity = db.find_or_create_entity(
        "Acme Corp", aliases=["Acme", "Acme Corporation"]
    )
    print(f"entity: {entity['name']} ({entity['id']})")

    # 2. A couple of raw-input nodes with embeddings.
    claim_text = "Acme Corp announced a new battery factory in Nevada."
    claim = db.insert_node(
        node_category="raw_input",
        node_kind="event_announcement",
        content=claim_text,
        actor="Acme Corp",
        entity_id=entity["id"],
        subject="battery factory",
        confidence=0.9,
        source_url="https://example.com/acme-nevada",
        embedding=embed(claim_text),
    )
    print(f"node:   {claim['id']}  {claim['node_kind']}")

    denial_text = "A rival firm denied plans to build any new factory."
    denial = db.insert_node(
        node_category="raw_input",
        node_kind="denial",
        content=denial_text,
        actor="Rival Inc",
        subject="battery factory",
        embedding=embed(denial_text),
    )

    # 3. Link them — they share a subject.
    edge = db.insert_edge(claim["id"], denial["id"], "same_subject", weight=0.8)
    print(f"edge:   {edge['edge_type']}  {edge['source_id']} -> {edge['target_id']}")

    # 4. Semantic search.
    print("\nmatch_nodes('new manufacturing plant'):")
    for row in db.match_nodes(embed("new manufacturing plant"), match_threshold=0.2):
        print(f"  {row['similarity']:.3f}  [{row['node_kind']}] {row['content']}")


if __name__ == "__main__":
    main()
