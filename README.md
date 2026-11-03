# Academic Paper Knowledge Graph — Phase 2 refactoring

RAG-driven academic paper extraction and knowledge graph backend; based on the authorized Phase 1 working copy. **Institutional research ownership applies:** do not publish source code, research data, prompts, or diagrams without the owner's approval.

## Runtime flow

`POST /ingest_pdf` → PDF parser → chunker → embedding → pgvector chunks → retrieval → five-aspect LLM extraction → relation verification → graph DB.

**Unchanged public API:** `POST /ingest_pdf` (multipart `file`), `POST /ask_gpt` (form `question`), `GET /get_graph_data` (nodes/edges). **New:** `GET /ingestion/{document_id}` reports per-process, in-memory job status (not durable; requests currently run synchronously).

## Getting started (authorized test environment only)

```bash
python -m pip install -e '.[dev]'
cp .env.example .env  # set your OWN dev credentials, never commit .env
python -m pytest -q
uvicorn main:app --reload --port 8090
```

Before live use, inspect the existing database with `migrations/schema_inspection.sql`. `migrations/001_canonical_schema.sql` describes a **new/empty database only**; it never safely migrates a populated database. Configure 768-dimension vectors and re-embed legacy vectors when switching embedding models. Use the institutional owner's approval for any data migration.

## Design and limitations

See [docs/architecture.md](docs/architecture.md), [docs/migration.md](docs/migration.md) and [docs/status.md](docs/status.md). Offline tests use mock AI and DB. No live Gemini/Supabase integration has been verified. An indexed document may leave chunk rows behind if graph extraction later fails, matching Phase 1 behavior. Further RAG quality optimization and evaluation are Phase 3+.
