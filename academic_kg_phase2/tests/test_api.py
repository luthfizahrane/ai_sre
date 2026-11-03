from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock
from fastapi.testclient import TestClient
from academic_kg.api.main import create_app
from academic_kg.core.config import Settings


def make_client(tmp):
    settings=Settings(google_api_key="fake",db_host="dev",db_port=5432,db_user="dev",
                     db_password="private",db_name="dev",upload_temp_dir=str(tmp),max_pdf_bytes=50)
    pipeline=SimpleNamespace(run=AsyncMock())
    graph_repo=SimpleNamespace(graph_data=AsyncMock(return_value={"nodes":[],"edges":[]}))
    chat=SimpleNamespace(ask=AsyncMock(return_value="answer"))
    deps=SimpleNamespace(pipeline=pipeline,graph_repo=graph_repo,chat=chat)
    return create_app(settings=settings,container=deps),deps


def test_api_success_preserves_endpoints_and_cleanup():
    with TemporaryDirectory() as directory:
        app,deps=make_client(Path(directory))
        with TestClient(app) as client:
            resp=client.post('/ingest_pdf',files={"file":("paper.pdf",b"%PDF-1.4 sample","application/pdf")})
            assert resp.status_code==200
            doc_id=resp.json()["document_id"]
            assert resp.json()["message"]=="PDF berhasil diproses untuk Knowledge Graph."
            assert client.get('/ingestion/'+doc_id).json()["stage"]=="COMPLETED"
            assert client.get('/get_graph_data').json()=={"nodes":[],"edges":[]}
            assert client.post('/ask_gpt',data={"question":"hello"}).json()=={"answer":"answer"}
            assert list(Path(directory).iterdir())==[]
            assert deps.pipeline.run.await_count==1


def test_bad_content_and_limits_are_cleaned_up():
    with TemporaryDirectory() as directory:
        app,_=make_client(Path(directory))
        with TestClient(app) as client:
            resp=client.post('/ingest_pdf',files={"file":("paper.pdf",b"not actually pdf","application/pdf")})
            assert resp.status_code==400
            assert client.get('/ingestion/'+resp.headers['X-Document-ID']).json()['stage']=='FAILED'
            assert client.post('/ingest_pdf',files={"file":("paper.pdf",b"%PDF-"+b"x"*60,"application/pdf")}).status_code==413
        assert list(Path(directory).iterdir())==[]


def test_processing_failure_is_reported_without_sensitive_body():
    with TemporaryDirectory() as directory:
        app,deps=make_client(Path(directory))
        deps.pipeline.run.side_effect=RuntimeError("Secret DB error")
        with TestClient(app) as client:
            resp=client.post('/ingest_pdf',files={"file":("p.pdf",b"%PDF-1.4 sample","application/pdf")})
            assert resp.status_code==500
            assert "Secret" not in resp.text
            assert client.get('/ingestion/'+resp.headers['X-Document-ID']).json()['error_code']=='PROCESSING_ERROR'
        assert list(Path(directory).iterdir())==[]


def test_pdf_with_no_text_returns_422_and_cleans():
    with TemporaryDirectory() as directory:
        app, deps = make_client(Path(directory))
        deps.pipeline.run.side_effect = ValueError("PDF_EMPTY_TEXT")
        with TestClient(app) as client:
            resp = client.post('/ingest_pdf', files={"file": ("p.pdf", b"%PDF-1.4 sample", "application/pdf")})
            assert resp.status_code == 422
            assert client.get('/ingestion/' + resp.headers['X-Document-ID']).json()['error_code'] == 'PDF_EMPTY_TEXT'
        assert list(Path(directory).iterdir()) == []
