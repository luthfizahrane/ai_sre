# Moving from Phase 1

1. Backup your previous code and institutional PostgreSQL database.
2. Never overwrite institutional data or deploy without owner permission.
3. Use this new source-package layout instead of importing `embedding_process.py`, `kg_processor.py` directly. The `main.py` root-level Uvicorn entrypoint remains.
4. Create `.env` from `.env.example`; run offline tests before any live integration.
5. Verify canonical schema using `migrations/schema_inspection.sql` first. `001_canonical_schema.sql` is only for new databases and uses VECTOR(768). It does not upgrade existing schemas or constraints.
6. Test using synthetic or explicitly approved PDFs in a separate database.
7. The embedding model changed in Phase 1; embeddings from the prior model cannot be mixed safely even with the same vector dimension.
8. A failure after chunk insertion can leave orphan `document_chunks` rows; no automatic cleanup of institutional database rows is attempted.

## Change log

Phase 2 isolates HTTP/API, config, parsing, indexing, retrieval, extraction, relationship evaluation and SQL. Adds optional worker-local ingestion status endpoint and upload size validation; removes module-level DB/LLM construction. No automatic database changes.
