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

-- 3. Indexes ----------------------------------------------------------------
create index if not exists nodes_embedding_idx
  on nodes using ivfflat (embedding vector_cosine_ops) with (lists = 100);
create index if not exists nodes_actor_idx     on nodes (actor);
create index if not exists nodes_subject_idx   on nodes (subject);
create index if not exists nodes_node_kind_idx on nodes (node_kind);
create index if not exists edges_source_id_idx on edges (source_id);
create index if not exists edges_target_id_idx on edges (target_id);

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
