# Module architecture

- `api/`: HTTP-only validation, response, cleanup and application factory.
- `core/`: centralized validated configuration, logger, response parsing utilities.
- `db/`: single SQL ownership (`DocumentRepository`, `GraphRepository`) and pgvector connection.
- `ingestion/`: MarkItDown PDF conversion, chunking, embedding adapter, indexing service.
- `retrieval/`: document-scoped pgvector nearest-neighbor retrieval, 2000-character fallback.
- `extraction/`: versionable prompts and Pydantic schemas, academic five-aspect extraction.
- `graph/`: node/domain mappings and five relation types with LLM verification.
- `services/`: dependency assembly, synchronous pipeline orchestration, per-worker status tracker and standalone chat.

## Compatibility

Preserves Phase 1 chunk size 800, overlap 100, retrieval top-k 20, LLM temperatures, five relation labels, PostgreSQL columns and three existing HTTP endpoint paths. Moves prompt formatting into plain Python strings sent to the same LangChain Google chat integration; not a model/prompt redesign. Graph response preserves `nodes[].label`, `details`, `edges[].source_id`, `target_id`, `type` aliases.
