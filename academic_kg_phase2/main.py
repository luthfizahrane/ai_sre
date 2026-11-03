"""Backwards compatible entrypoint: uvicorn main:app --reload --port 8090."""
from academic_kg.api.main import app, create_app

__all__ = ["app", "create_app"]
