# Processing status and error semantics

`GET /ingestion/{document_id}` returns `{document_id, stage, error_code, updated_at}`.
Stages: PENDING, PARSING, INDEXING, RETRIEVING, EXTRACTING, LINKING, SAVING, COMPLETED, FAILED.

The existing `POST /ingest_pdf` **awaits** completion before sending success; it is not a background job. For failures after an ID exists, `X-Document-ID` exposes the ID for status inspection. The status table exists only within this application worker's memory (bounded at 500 jobs), so workers/restarts do not share or retain status. Never use it as a production audit log.

- Invalid PDF type or bytes: 400
- PDF over upload limit: 413
- No extractable text: 422
- Downstream processing failure: 500
- Chat provider failure: 502
- Graph fetch failure: 503

All temporary PDF files are deleted within `finally`, including processing errors. No original fulltext or credentials are intentionally written to app logs. Ingestion is not fully atomic: chunks are committed before graph extraction. Existing failed-document chunks require an authorized cleanup/recovery policy before production rollout.
- Request cancellation: `CANCELLED` job status if the worker can update its in-memory status, with temporary PDF cleanup in `finally`.
