"""HTTP boundary: validation, safe errors, cleanup and response contracts."""
import asyncio
import logging
import uuid
from pathlib import Path
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from academic_kg.services.status import Stage

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/")
async def root():
    return {"message": "Selamat datang di Knowledge Graph API!"}


@router.post("/ingest_pdf")
async def ingest_pdf_endpoint(request: Request, file: UploadFile = File(...)):
    """Preserves Phase 1 success JSON; errors use minimal generic messages."""
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        await file.close()
        raise HTTPException(status_code=400, detail="Hanya file PDF yang diizinkan.")
    doc_id = str(uuid.uuid4())
    settings = request.app.state.settings
    store = request.app.state.status_store
    directory = Path(settings.upload_temp_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{doc_id}.pdf"
    await store.update(doc_id, Stage.PENDING)
    try:
        # Read upload incrementally; do not load arbitrarily large PDFs in memory.
        total = 0
        header = b""
        with target.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > settings.max_pdf_bytes:
                    raise HTTPException(status_code=413, detail="Ukuran PDF melebihi batas.")
                if len(header) < 5:
                    header += chunk[:5 - len(header)]
                output.write(chunk)
        if header != b"%PDF-":
            raise HTTPException(status_code=400, detail="File bukan PDF yang valid.")

        async def report(stage: Stage):
            await store.update(doc_id, stage)

        await request.app.state.pipeline.run(target, doc_id, filename, report)
        await store.update(doc_id, Stage.COMPLETED)
        return {"message": "PDF berhasil diproses untuk Knowledge Graph.", "document_id": doc_id}
    except asyncio.CancelledError:
        await store.update(doc_id, Stage.FAILED, error_code="CANCELLED")
        raise
    except HTTPException as exc:
        await store.update(doc_id, Stage.FAILED, error_code=f"HTTP_{exc.status_code}")
        exc.headers = {**(exc.headers or {}), "X-Document-ID": doc_id}
        raise
    except ValueError as exc:
        if str(exc) == "PDF_EMPTY_TEXT":
            await store.update(doc_id, Stage.FAILED, error_code="PDF_EMPTY_TEXT")
            raise HTTPException(status_code=422,
                                detail="Teks PDF tidak dapat diekstraksi.",
                                headers={"X-Document-ID": doc_id}) from None
        await store.update(doc_id, Stage.FAILED, error_code="PROCESSING_ERROR")
        logger.error("PDF processing failed doc_id=%s (%s)", doc_id, type(exc).__name__)
        raise HTTPException(status_code=500, detail="Gagal memproses PDF.",
                            headers={"X-Document-ID": doc_id}) from None
    except Exception as exc:
        await store.update(doc_id, Stage.FAILED, error_code="PROCESSING_ERROR")
        logger.error("PDF processing failed doc_id=%s (%s)", doc_id, type(exc).__name__)
        raise HTTPException(status_code=500, detail="Gagal memproses PDF.",
                            headers={"X-Document-ID": doc_id}) from None
    finally:
        await file.close()
        target.unlink(missing_ok=True)


@router.get("/ingestion/{document_id}")
async def ingestion_status(request: Request, document_id: str):
    status = await request.app.state.status_store.get(document_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Status tidak ditemukan di worker ini.")
    return status


@router.post("/ask_gpt")
async def ask_gpt_endpoint(request: Request, question: str = Form(...)):
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question is required")
    try:
        answer = await request.app.state.chat.ask(question)
        return {"answer": answer}
    except Exception as exc:
        logger.warning("LLM unavailable for /ask_gpt (%s)", type(exc).__name__)
        raise HTTPException(status_code=502, detail="LLM service tidak tersedia.") from None


@router.get("/get_graph_data")
async def get_graph_data_endpoint(request: Request):
    try:
        return await request.app.state.graph_repo.graph_data()
    except Exception as exc:
        logger.warning("Failed graph query (%s)", type(exc).__name__)
        raise HTTPException(status_code=503, detail="Graph data tidak tersedia.") from None
