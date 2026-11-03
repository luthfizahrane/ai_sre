-- Phase 1 canonical schema for a NEW Supabase PostgreSQL database.
-- This script creates missing tables; it DOES NOT change existing table layouts.
-- If tables already exist, inspect their columns/constraints before using it.
-- BACK UP ALL INSTITUTIONAL RESEARCH DATA before any migration.

CREATE EXTENSION IF NOT EXISTS vector;

-- Metadata + vectorized chunks. Real page numbers are unknown with the current
-- text-only MarkItDown pipeline: page_number is therefore nullable.
CREATE TABLE IF NOT EXISTS public.document_chunks (
    id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    chunk_text TEXT NOT NULL,
    page_number INTEGER NULL,
    embedding VECTOR(768) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_document_chunks_doc_id
    ON public.document_chunks (doc_id);

-- One node per article. The SQL columns match INSERT/SELECT in kg_processor.py.
CREATE TABLE IF NOT EXISTS public.nodes_kg (
    id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    title TEXT,
    att_goal TEXT,
    att_method TEXT,
    att_background TEXT,
    att_future TEXT,
    att_gaps TEXT,
    type TEXT NOT NULL DEFAULT 'article',
    content TEXT NOT NULL,
    embedding VECTOR(768),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.edges_kg (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    "fromId" TEXT NOT NULL REFERENCES public.nodes_kg(id) ON DELETE CASCADE,
    "toId" TEXT NOT NULL REFERENCES public.nodes_kg(id) ON DELETE CASCADE,
    relation TEXT NOT NULL,
    label TEXT,
    color TEXT,
    context TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_edges_kg_from_to_relation UNIQUE ("fromId", "toId", relation),
    CONSTRAINT chk_edges_kg_not_self CHECK ("fromId" <> "toId")
);

-- A rerun with the same node pair and relation uses ON CONFLICT ... DO UPDATE.
-- If an old edges_kg already exists without the unique constraint, add one
-- AFTER verifying no duplicates exist, and reviewing any required data changes:
-- ALTER TABLE public.edges_kg ADD CONSTRAINT uq_edges_kg_from_to_relation
--   UNIQUE ("fromId", "toId", relation);
