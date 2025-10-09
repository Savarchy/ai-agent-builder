create extension if not exists vector;
create extension if not exists pgcrypto;

create table if not exists documents (
  id uuid primary key default gen_random_uuid(),
  title text,
  url text,
  content text
);

create table if not exists chunks (
  id uuid primary key default gen_random_uuid(),
  document_id uuid references documents(id) on delete cascade,
  ord int,
  text text,
  embedding vector(1536),
  created_at timestamptz not null default now()
);

create index if not exists ix_chunks_embedding_cosine
  on chunks using ivfflat (embedding vector_cosine_ops) with (lists = 100);
