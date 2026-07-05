-- Enceladus knowledge-graph schema
-- Run this in the Supabase SQL editor, or via psql:
--   psql "$SUPABASE_DB_URL" -f schema.sql

-- 1. Extensions -------------------------------------------------------------
create extension if not exists vector;

-- 2. Tables -----------------------------------------------------------------
create table if not exists entities (
  id         uuid primary key default gen_random_uuid(),
  name       text not null unique,
  aliases    text[] default '{}',
  embedding  vector(1536),
  created_at timestamptz default now()
);

create table if not exists nodes (
  id            uuid primary key default gen_random_uuid(),
  node_category text not null check (node_category in ('raw_input', 'inference')),
  node_kind     text not null check (node_kind in (
                  'fact', 'claim', 'position', 'event_announcement',
                  'prediction', 'denial', 'agreement', 'contradiction', 'derived')),
  content       text not null,
  actor         text,
  entity_id     uuid references entities(id),
  subject       text,
  confidence    float default 0.8,
  source_url    text,
  event_date    timestamptz,
  expires_at    timestamptz,
  embedding     vector(1536),
  -- false until the batched inference engine has paired this raw_input node.
  -- See section 9 and ingestion.run_inference_batch.
  inference_processed boolean default false,
  created_at    timestamptz default now()
);
-- Additive for DBs created before the batched inference engine existed.
alter table nodes add column if not exists inference_processed boolean default false;
-- Derivation depth: raw_input nodes are 0; a derived inference is 1 + max(premise
-- depths). Lets the forward-chaining engine layer by epistemic depth and cap it.
alter table nodes add column if not exists depth int default 0;
-- Source credibility weight in [0,1] (default 1.0). Written only when source
-- weighting is enabled (see ingestion.USE_SOURCE_WEIGHTS); coverage/corroboration
-- weight evidence by it so reputable sources count for more.
alter table nodes add column if not exists source_weight float default 1.0;

create table if not exists edges (
  id         uuid primary key default gen_random_uuid(),
  source_id  uuid references nodes(id) on delete cascade,
  target_id  uuid references nodes(id) on delete cascade,
  edge_type  text not null check (edge_type in (
               'same_subject', 'same_actor', 'semantically_similar',
               'derives_from', 'contradicts',
               'corroborated_by', 'converges_with', 'supersedes')),
  weight     float default 1.0,
  created_at timestamptz default now()
);
-- Additive for DBs whose edges CHECK predates the inference edge types. The
-- constraint name is Postgres' default for the inline CHECK above.
alter table edges drop constraint if exists edges_edge_type_check;
alter table edges add constraint edges_edge_type_check check (edge_type in (
  'same_subject', 'same_actor', 'semantically_similar',
  'derives_from', 'contradicts', 'corroborated_by', 'converges_with',
  'supersedes'));  -- supersedes: a newer report overtakes an older one (belief revision)

-- node_entities: many-to-many between nodes and the entities they involve, with
-- a role per link. Additive alongside nodes.entity_id (which stays the single
-- primary-actor reference); this table records every participating entity.
create table if not exists node_entities (
  node_id   uuid references nodes(id) on delete cascade,
  entity_id uuid references entities(id) on delete cascade,
  role      text,
  primary key (node_id, entity_id)
);

-- 3. Indexes ----------------------------------------------------------------
create index if not exists nodes_embedding_idx
  on nodes using ivfflat (embedding vector_cosine_ops) with (lists = 100);
create index if not exists entities_embedding_idx
  on entities using ivfflat (embedding vector_cosine_ops) with (lists = 100);
create index if not exists nodes_actor_idx     on nodes (actor);
create index if not exists nodes_subject_idx   on nodes (subject);
create index if not exists nodes_node_kind_idx on nodes (node_kind);
create index if not exists edges_source_id_idx on edges (source_id);
create index if not exists edges_target_id_idx on edges (target_id);
create index if not exists node_entities_entity_id_idx on node_entities (entity_id);
create index if not exists node_entities_node_id_idx   on node_entities (node_id);

-- 4. Similarity search function --------------------------------------------
create or replace function match_nodes(
  query_embedding vector(1536),
  match_threshold float default 0.75,
  match_count int default 15
)
returns table (
  id uuid,
  content text,
  node_kind text,
  actor text,
  subject text,
  confidence float,
  similarity float
)
language sql stable as $$
  select id, content, node_kind, actor, subject, confidence,
    1 - (embedding <=> query_embedding) as similarity
  from nodes
  where 1 - (embedding <=> query_embedding) > match_threshold
    and embedding is not null
  order by embedding <=> query_embedding
  limit match_count;
$$;

-- Entity resolution by embedding similarity (mirrors match_nodes). Used to
-- collapse sibling surface forms of one real-world actor ("Iran" / "Tehran" /
-- "Iran's foreign minister") onto a single entity. Threshold is intentionally
-- high: a false merge of two distinct actors corrupts the coverage/independence
-- signals more than residual fragmentation, so callers pass a conservative value.
create or replace function match_entities(
  query_embedding vector(1536),
  match_threshold float default 0.85,
  match_count int default 5
)
returns table (
  id uuid,
  name text,
  aliases text[],
  similarity float
)
language sql stable as $$
  select id, name, aliases,
    1 - (embedding <=> query_embedding) as similarity
  from entities
  where embedding is not null
    and 1 - (embedding <=> query_embedding) > match_threshold
  order by embedding <=> query_embedding
  limit match_count;
$$;

-- 5. Stream query (read-time prototype) -------------------------------------
-- Nodes that involve BOTH entities (the relationship "channel" between a pair),
-- newest first. Read-time only: there is no materialized streams table yet.
create or replace function stream_between(
  a uuid,
  b uuid,
  max_count int default 100
)
returns table (
  id uuid,
  node_category text,
  node_kind text,
  content text,
  actor text,
  subject text,
  confidence float,
  source_url text,
  event_date timestamptz,
  created_at timestamptz
)
language sql stable as $$
  select n.id, n.node_category, n.node_kind, n.content, n.actor, n.subject,
         n.confidence, n.source_url, n.event_date, n.created_at
  from nodes n
  where exists (
          select 1 from node_entities ne
          where ne.node_id = n.id and ne.entity_id = a)
    and exists (
          select 1 from node_entities ne
          where ne.node_id = n.id and ne.entity_id = b)
  order by coalesce(n.event_date, n.created_at) desc
  limit max_count;
$$;

-- 7. Topics (thematic/domain axis, mirrors entities) ------------------------
create table if not exists topics (
  id         uuid primary key default gen_random_uuid(),
  name       text not null unique,
  aliases    text[] default '{}',
  embedding  vector(1536),
  created_at timestamptz default now()
);

-- node_topics: many-to-many between nodes and their thematic domains.
-- A node can belong to multiple domains (e.g. "energy" + "military conflict").
create table if not exists node_topics (
  node_id  uuid references nodes(id) on delete cascade,
  topic_id uuid references topics(id) on delete cascade,
  primary key (node_id, topic_id)
);

create index if not exists topics_embedding_idx
  on topics using ivfflat (embedding vector_cosine_ops) with (lists = 100);
create index if not exists node_topics_topic_id_idx on node_topics (topic_id);
create index if not exists node_topics_node_id_idx  on node_topics (node_id);

-- Cosine KNN over topic embeddings. Default threshold is lower than
-- match_entities (0.85) because topic surface forms cluster more broadly
-- ("nuclear talks" / "nuclear negotiations" should merge).
create or replace function match_topics(
  query_embedding vector(1536),
  match_threshold float default 0.80,
  match_count int default 5
)
returns table (
  id uuid,
  name text,
  aliases text[],
  similarity float
)
language sql stable as $$
  select id, name, aliases,
    1 - (embedding <=> query_embedding) as similarity
  from topics
  where embedding is not null
    and 1 - (embedding <=> query_embedding) > match_threshold
  order by embedding <=> query_embedding
  limit match_count;
$$;

-- 7b. Topic hierarchy (overlapping DAG over the flat topics) -----------------
-- The flat `topics` rows are the leaves the LLM actually tags. This layer adds
-- IS-A relations *above* them, so e.g. 'chip exports' -> 'semiconductors' ->
-- 'economics'. It is a DAG, not a tree: a topic may have several parents
-- ('chip export controls' under both 'semiconductors' and 'trade policy'), which
-- is why parents live in an edge table rather than a parent_id column.
--
-- Rollups ("all economics news") are computed at READ TIME via the recursive
-- functions below — we deliberately do NOT materialise a node->ancestor closure
-- (same philosophy as streams: store the small, stable structure; derive the
-- aggregate on query).
alter table topics add column if not exists is_root boolean default false;

create table if not exists topic_relations (
  child_id   uuid references topics(id) on delete cascade,
  parent_id  uuid references topics(id) on delete cascade,
  created_at timestamptz default now(),
  primary key (child_id, parent_id),
  check (child_id <> parent_id)        -- no self-loops; cycles guarded in SQL below
);

create index if not exists topic_relations_child_idx  on topic_relations (child_id);
create index if not exists topic_relations_parent_idx on topic_relations (parent_id);

-- All topics at or below `root` (root included), walking child edges downward.
-- The path array is a cycle guard: a child already on the path is never re-expanded.
create or replace function topic_descendants(root uuid)
returns table (id uuid)
language sql stable as $$
  with recursive d(id, path) as (
    select root, array[root]
    union all
    select tr.child_id, d.path || tr.child_id
    from topic_relations tr
    join d on tr.parent_id = d.id
    where not (tr.child_id = any(d.path))
  )
  select distinct id from d;
$$;

-- All topics at or above `start_id` (start included), walking parent edges upward.
create or replace function topic_ancestors(start_id uuid)
returns table (id uuid)
language sql stable as $$
  with recursive a(id, path) as (
    select start_id, array[start_id]
    union all
    select tr.parent_id, a.path || tr.parent_id
    from topic_relations tr
    join a on tr.child_id = a.id
    where not (tr.parent_id = any(a.path))
  )
  select distinct id from a;
$$;

-- Read-time rollup: recent raw_input nodes tagged with `root` or any descendant
-- of it. EXISTS (not a join) so a node tagged with several descendants appears once.
create or replace function nodes_under_topic(root uuid, match_count int default 50)
returns table (
  id uuid,
  node_kind text,
  actor text,
  subject text,
  confidence float,
  content text
)
language sql stable as $$
  select n.id, n.node_kind, n.actor, n.subject, n.confidence, n.content
  from nodes n
  where n.node_category = 'raw_input'
    and exists (
      select 1 from node_topics nt
      where nt.node_id = n.id
        and nt.topic_id in (select id from topic_descendants(root))
    )
  order by coalesce(n.event_date, n.created_at) desc
  limit match_count;
$$;

-- 8. Backfill node_entities from the existing primary-actor reference --------
-- Idempotent: safe to re-run alongside the rest of this file.
insert into node_entities (node_id, entity_id, role)
select id, entity_id, 'actor'
from nodes
where entity_id is not null
on conflict (node_id, entity_id) do nothing;

-- 9. Batched, adversarially-verified inference ------------------------------
-- The inference *content* lives as a node (node_category='inference'); this
-- companion table holds the verification verdict and provenance for that node.
create table if not exists inference_meta (
  node_id           uuid primary key references nodes(id) on delete cascade,
  status            text not null check (status in ('contested', 'corroborated', 'unverified')),
  base_confidence   float,          -- Pass 1 confidence, before coverage gating
  coverage          float,          -- [0,1] corpus density around actor/subject/time
  support_node_ids  uuid[] default '{}',
  defeater_node_ids uuid[] default '{}',
  alternatives      jsonb  default '[]',  -- [{explanation, evidence_signature, reportability, retrieved_ids, status}]
  converged_with    uuid[] default '{}',  -- other inference node_ids that independently re-derive this one
  created_at        timestamptz default now()
);
create index if not exists inference_meta_status_idx on inference_meta (status);

-- Indexes for coverage estimation + structural evidence pulls (time windows).
create index if not exists nodes_event_date_idx on nodes (event_date);
create index if not exists nodes_created_at_idx on nodes (created_at);

-- match_inferences: cosine KNN restricted to inference nodes. Used by Pass 3 to
-- find semantically-equivalent existing inferences (convergence detection).
create or replace function match_inferences(
  query_embedding vector(1536),
  match_threshold float default 0.90,
  match_count int default 10
)
returns table (
  id uuid,
  content text,
  confidence float,
  similarity float
)
language sql stable as $$
  select id, content, confidence,
    1 - (embedding <=> query_embedding) as similarity
  from nodes
  where node_category = 'inference'
    and embedding is not null
    and 1 - (embedding <=> query_embedding) > match_threshold
  order by embedding <=> query_embedding
  limit match_count;
$$;

-- derivation_roots: the transitive raw_input ancestors of a node, walking
-- derives_from edges down to the leaves (a raw_input node returns just itself).
-- This is the provenance set the independence/convergence guard compares on:
-- two inferences only count as independent re-derivations when their RAW
-- grounding is largely disjoint, so a conclusion derived from its own
-- descendants can never masquerade as independent corroboration. The path array
-- guards against cycles in the derives_from graph.
create or replace function derivation_roots(start_id uuid)
returns table (id uuid)
language sql stable as $$
  with recursive chain(id, path) as (
    select start_id, array[start_id]
    union all
    select e.target_id, chain.path || e.target_id
    from edges e
    join chain on e.source_id = chain.id
    where e.edge_type = 'derives_from'
      and not (e.target_id = any(chain.path))
  )
  select distinct n.id
  from chain
  join nodes n on n.id = chain.id
  where n.node_category = 'raw_input';
$$;

-- 11. Belief revision (living model) -----------------------------------------
-- revised_at: when the engine last RE-verified this inference against newly
-- arrived evidence (null = never revised since creation). Supersession itself is
-- an edge ('supersedes', newer -> older) plus the CHECK addition in section 2.
alter table inference_meta add column if not exists revised_at timestamptz;
