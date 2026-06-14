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
  created_at    timestamptz default now()
);

create table if not exists edges (
  id         uuid primary key default gen_random_uuid(),
  source_id  uuid references nodes(id) on delete cascade,
  target_id  uuid references nodes(id) on delete cascade,
  edge_type  text not null check (edge_type in (
               'same_subject', 'same_actor', 'semantically_similar',
               'derives_from', 'contradicts')),
  weight     float default 1.0,
  created_at timestamptz default now()
);

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

-- 6. Backfill node_entities from the existing primary-actor reference --------
-- Idempotent: safe to re-run alongside the rest of this file.
insert into node_entities (node_id, entity_id, role)
select id, entity_id, 'actor'
from nodes
where entity_id is not null
on conflict (node_id, entity_id) do nothing;
