# SleepWalk

**An adversarial verification engine for claims extracted from live news.**

Raw claims are pulled from news, embedded into a knowledge graph, paired against
related evidence, and every derived inference is put through a structured
**debate** (objection → rebuttal → adjudication) across multiple LLMs before it's
allowed to carry a confidence score.

## Why this exists

LLM-extracted claims are cheap to generate and easy to get wrong: two nodes that
merely *co-occur* topically look, to a naive similarity search, identical to two
nodes where one actually causes the other. SleepWalk is built around that failure
mode specifically. The diagnostic eval suite (`eval/labels.jsonl`,
`docs/EVAL_SEED_NOTES.md`) includes hand-constructed scenarios designed to catch
exactly this — e.g. two markets moving together on the same day with a shared
confound, where the correct verdict is "unverified," not "corroborated." A
system that can't tell the difference has a real, measurable precision problem;
this one is built to expose that problem rather than hide it.

## How it works

```
raw text -> extract_node (LLM) -> embed (OpenAI 1536-d) -> entity resolution
    -> insert into graph (Postgres/pgvector, Supabase)
    -> pair with similar/contradicting nodes (cross-encoder rerank + NLI)
    -> adversarial verification (debate loop) -> confidence-scored inference
```

**Retrieval & scoring**
- Two-stage entity resolution: a wide-net cosine pass surfaces candidates
  (`ENTITY_CANDIDATE_THRESHOLD`), then an LLM coreference call disambiguates
  them — so "Russian forces" merges into "Russia" without over-merging distinct
  topics.
- Evidence retrieval is reranked with a local cross-encoder
  (`ms-marco-MiniLM-L6-v2`) rather than trusting raw cosine similarity.
- A local NLI model (`nli-deberta-v3-base`) scores entailment/contradiction
  between each candidate inference and its evidence, independent of (or
  alongside) the LLM classifier — `nli.py`.

**Adversarial verification (`ingestion.run_inference_batch`, `llm_service.py`)**
- Pass 1 (`reason_pair`) proposes a candidate inference from two related nodes,
  with `enumerate_alternatives` generating competing explanations for the same
  evidence — so "these moved together" doesn't automatically become "one caused
  the other."
- Pass 2 (`classify_evidence`) retrieves the strongest supporting/contradicting
  nodes and classifies each as support, defeater (`supports_alternative`), or
  irrelevant.
- Pass 3 is a genuine debate: `debate_object` raises the strongest objection
  from the defeater evidence, `debate_rebut` responds, `debate_adjudicate`
  renders a verdict on the exchange.
- Confidence isn't a raw LLM number — it's computed from coverage (how much of
  the topic's corpus was actually checked), a corroboration bonus scaled by
  coverage × reportability, a defeater penalty when contested, and an
  **independence-checked convergence bonus**: if the same inference is
  independently re-derived from a disjoint source set (Jaccard overlap below
  `INDEPENDENCE_MAX_OVERLAP`), that's treated as real corroboration; if it's
  re-derived from the same sources, it isn't.

**Frontier maps (`frontier.py`, read-time only)**
- `contested_clusters()` — where the evidence actively disagrees, grouped by
  topic.
- `coverage_gaps()` — topics that are thin or under-corroborated: what the graph
  doesn't know yet.
- `underived_links()` — strongly related raw pairs that were never reasoned
  about: the engine's own backlog.

**Evaluation harness (`eval/run_eval.py`)**
Re-runs verification over a fixed, human-labeled set and reports:
- precision / recall / F1 of `status == corroborated` against `truth == true`
- a confidence-decile calibration table + Expected Calibration Error
- a 3×3 confusion matrix (predicted status × truth)
- `--compare` mode: an ablation grid isolating specific scoring changes
  (legacy vs. fixed convergence bonus, strict vs. weighted defeater policy)
  so a regression or improvement can be attributed to a specific change, not
  vibes.

Because the labeled set and LLM calls run at temperature 0, two runs of the same
config are directly comparable — this is a real regression harness, not a
one-off notebook metric.

## Layout

```
db.py ingestion.py llm_service.py    core library — data access, the ingestion +
embeddings.py rerank.py nli.py       adversarial-inference engine, LLM router,
sources.py                           and local cross-encoder / NLI models
schema.sql                           exact DDL: tables, indexes, RPCs
frontier.py                          read-time contested/gap/backlog maps
main.py  dashboard.html              FastAPI app + single-page dashboard
seeds/                               corpus seeders: seed_large, seed_news,
                                      seed_eval (diagnostic benchmark), seed_roots
seed_data/                           frozen real-news snapshot (committed)
eval/                                labels.jsonl + run_eval measurement harness
tools/                               inspection/diagnostics: graph_inspect,
                                      gate_check, topics_tree, example, ...
tests/                               pytest suite (root conftest.py sets the path)
docs/                                EVAL_SEED_NOTES.md — what each diagnostic
                                      scenario tests and the expected verdict
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
- `GROQ_API_KEY` / `GEMINI_API_KEY` — the two-provider LLM router
  (`llm_service.py`); Gemini is secondary failover for when Groq's daily token
  cap is hit.

### 3. Install and run

```bash
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt
python tools/example.py
```

### 4. Run the diagnostic eval

```bash
python seeds/seed_eval.py --reset
python eval/label_dump.py           # or open /dashboard to review verdicts
python eval/run_eval.py             # precision/recall/F1 + calibration
python eval/run_eval.py --compare   # ablation grid across scoring changes
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

Run the adversarial inference pass over newly ingested nodes:

```python
import ingestion
ingestion.run_inference_batch()
```

## API (FastAPI, `main.py`)

| Route | Purpose |
|---|---|
| `POST /ingest`, `/ingest/news`, `/ingest/poll` | feed raw text / NewsAPI articles into the graph |
| `POST /infer/run` | trigger a batched adversarial-inference pass |
| `GET /contested` | contested-cluster frontier view |
| `GET /frontier` | full frontier map (contested + coverage gaps + backlog) |
| `GET /domain/{topic_name}` | one topic's rollup + graded inferences |
| `GET /nodes`, `/nodes/{id}/graph`, `/inferences`, `/entities` | graph inspection |
| `GET /dashboard` | single-page dashboard over the above |

## Notes

- Embeddings are passed as plain Python lists; the Supabase client serialises
  them to the `vector` columns and to the `match_nodes` RPC parameter.
- Allowed `node_category`, `node_kind`, and `edge_type` values are enforced both
  by the SQL `CHECK` constraints and by validation in `db.py`.
- The `ivfflat` index uses `lists = 100`; for best recall, build the index after
  the table has a representative amount of data.
- `ENCELADUS_CONVERGENCE_LEGACY=1` restores the old (buggy) unconditional
  convergence bonus — kept only for the eval harness's `--compare` ablation, not
  for normal use.