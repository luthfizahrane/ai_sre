"""Application factory: inject services in tests; lazily load providers in lifespan."""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from academic_kg.api.routes import router
from academic_kg.core.config import Settings
from academic_kg.core.logging import configure_logging
from academic_kg.services.status import StatusStore


def create_app(settings: Settings | None = None, container=None, status_store=None) -> FastAPI:
    # Local tests can supply fake settings+container without Google/Supabase.
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        from academic_kg.services.container import build_container
        app.state.settings = settings or Settings.from_env()
        deps = container if container is not None else build_container(app.state.settings)
        app.state.pipeline = deps.pipeline
        app.state.graph_repo = deps.graph_repo
        app.state.chat = deps.chat
        app.state.status_store = status_store if status_store is not None else StatusStore()
        yield

    configure_logging()
    app = FastAPI(title="Academic Paper Knowledge Graph API", lifespan=lifespan)
    # This default is overridden by Settings at app creation only if supplied.
    # For env-based settings, resolve only the safe CORS value before startup.
    import os
    from dotenv import load_dotenv
    load_dotenv()
    origins = settings.cors_origins if settings else tuple(
        item.strip() for item in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") if item.strip()
    )
    if "*" in origins:
        raise ValueError("CORS_ORIGINS must be explicit")
    app.add_middleware(
        CORSMiddleware, allow_origins=list(origins), allow_credentials=True,
        allow_methods=["GET", "POST"], allow_headers=["*"],
        expose_headers=["X-Document-ID"],
    )
    app.include_router(router)
    return app


app = create_app()
