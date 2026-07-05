# Enceladus

A Supabase-backed knowledge graph with semantic (pgvector) search. Nodes hold
facts/claims/predictions, edges relate them, and `match_nodes` does cosine
similarity search over 1536-dim embeddings.

## Layout

```
db.py ingestion.py llm_service.py    core library — data access, the ingestion +
embeddings.py rerank.py nli.py       adversarial-inference engine, LLM router,
sources.py                           and local models
schema.sql                           exact DDL: tables, indexes, RPCs
main.py  dashboard.html              FastAPI app + single-page dashboard
seeds/                               corpus seeders: seed_large, seed_news,
                                     seed_eval, seed_roots, fetch_news_snapshot
seed_data/                           frozen real-news snapshot (committed)
eval/                                labelling + run_eval measurement harness
tools/                               inspection/diagnostics: graph_inspect,
                                     gate_check, topics_tree, example, ...
tests/                               pytest suite (root conftest.py sets the path)
docs/                                EVAL_SEED_NOTES.md and other docs
.env.example                         credential template — copy to `.env`
```

Scripts under `seeds/`, `tools/`, `tests/`, and `eval/` add the repo root to
`sys.path`, so run them from the repo root — e.g. `python seeds/seed_eval.py --reset`.

## Setup

### 1. Apply the schema

Either paste `schema.sql` into the **Supabase SQL editor** and run it, or use
`psql` with your direct connection string:

```bash
psql "$SUPABASE_DB_URL" -f schema.sql
```

`SUPABASE_DB_URL` comes from *Project Settings → Database → Connection string*.

### 2. Configure credentials

```bash
cp .env.example .env   # then edit .env
```

- `SUPABASE_URL` / `SUPABASE_KEY` — *Project Settings → API*. Use the service-role
  key for writes from a trusted backend; the anon key only works if RLS allows it.
- `OPENAI_API_KEY` — needed by `embeddings.py`.

### 3. Install and run

```bash
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt
python tools/example.py
```

## Usage

```python
import db
from embeddings import embed

entity = db.find_or_create_entity("Acme Corp", aliases=["Acme"])

node = db.insert_node(
    node_category="raw_input",          # raw_input | inference
    node_kind="event_announcement",     # fact, claim, prediction, denial, ...
    content="Acme announced a Nevada factory.",
    actor="Acme Corp",
    entity_id=entity["id"],
    subject="battery factory",
    confidence=0.9,
    embedding=embed("Acme announced a Nevada factory."),
)

db.insert_edge(node["id"], other_id, "same_subject", weight=0.8)

hits = db.match_nodes(embed("new manufacturing plant"), match_threshold=0.75)
```

## Notes

- Embeddings are passed as plain Python lists; the Supabase client serialises
  them to the `vector` columns and to the `match_nodes` RPC parameter.
- Allowed `node_category`, `node_kind`, and `edge_type` values are enforced both
  by the SQL `CHECK` constraints and by validation in `db.py`.
- The `ivfflat` index uses `lists = 100`; for best recall, build the index after
  the table has a representative amount of data.
