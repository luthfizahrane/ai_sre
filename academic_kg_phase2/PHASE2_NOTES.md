# Phase 2 — Refactoring completion notes

## Phase 1 vs. Phase 2

Phase 1 fixed initialization, relation prompt formatting, SQL mismatch, JSON parsing/null, credential logging, and relation UPSERT; it **did not** modularize the architecture or formalize job status. Phase 2 has now moved its logic into a `src/academic_kg` installable package while preserving the original endpoints and database contract.

| Requirement | Implemented location |
|---|---|
| Separate API, PDF, embeddings, retrieval, KG and database | `src/academic_kg/{api,ingestion,retrieval,extraction,graph,db}` |
| Central prompts and Pydantic models | `extraction/prompts.py`, `extraction/schemas.py` |
| Environment configuration and dependencies | `core/config.py`, `.env.example`, `pyproject.toml` |
| Pipeline stages, failure handling and temp-file cleanup | `services/pipeline.py`, `services/status.py`, `api/routes.py` |
| Cleanup of unused imports, comments, prints and duplicate conversion | Modules under `src/academic_kg` |

## Tests

Offline tests cover prompt formatting, model response parsing, schema validation, SQL contracts, transaction boundaries, API success/failure/cleanup, status tracking, and one synthetic in-memory end-to-end workflow. Run:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

These tests **do not** verify Google Gemini API, PDF parsing quality on real papers, or PostgreSQL schema compatibility with existing institutional data. No live model or production database credentials were used.

## Breaking/behavior changes to review

- Old imports `from kg_processor import KnowledgeGraphProcessor` and `from embedding_process import DocumentPreprocessor` are replaced with package imports; do not mix old and new source files. `uvicorn main:app` still works after package installation.
- In addition to the original three routes, `GET /ingestion/{document_id}` exposes worker-local, transient status. This is not a task queue, job database or durable recovery mechanism.
- HTTP 413 is returned if the PDF exceeds `MAX_PDF_BYTES` (default 20 MB).
- HTTP 500/502/503 errors do not expose provider or database error messages.
- The model provider is still the existing Google Gemini integration, but prompt formatting is now rendered directly via Python strings before calling `.ainvoke()` instead of LangChain `PromptTemplate` chains.
- The original PDF chunks table is committed before KG computation. On a downstream failure, those chunks can remain and need a controlled recovery policy.
- Previous prompts and graph relation definitions are preserved; **no claims about improved accuracy** should be made without Phase 3 evaluation.

## Dataset and ownership

The research organization's ownership and data handling policies apply. Do not publish protected source, files, real research papers or database data to a public GitHub repository without explicit permission. The zip is a local working artifact, not authorization to publish it.
