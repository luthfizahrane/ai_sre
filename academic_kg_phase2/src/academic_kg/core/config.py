"""Validated, centralized environment-based configuration."""
from dataclasses import dataclass, field
import os


@dataclass(frozen=True)
class Settings:
    google_api_key: str = field(repr=False)
    db_host: str
    db_port: int
    db_user: str
    db_password: str = field(repr=False)
    db_name: str
    embedding_model: str = "models/gemini-embedding-001"
    embedding_dim: int = 768
    extraction_model: str = "gemini-2.5-pro"
    chat_model: str = "gemini-2.5-flash"
    chunk_size: int = 800
    chunk_overlap: int = 100
    retrieval_top_k: int = 20
    relation_timeout_seconds: int = 30
    upload_temp_dir: str = "temp_uploads"
    max_pdf_bytes: int = 20 * 1024 * 1024
    cors_origins: tuple[str, ...] = ("http://localhost:3000",)

    def __post_init__(self):
        if not self.google_api_key or not all((self.db_host, self.db_user, self.db_password, self.db_name)):
            raise ValueError("Missing GOOGLE_API_KEY or Supabase database configuration")
        if not 1 <= self.db_port <= 65535:
            raise ValueError("Invalid database port")
        if self.embedding_dim != 768:
            raise ValueError("Current SQL migration requires EMBEDDING_DIM=768")
        if self.chunk_size <= 0 or not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("CHUNK_SIZE/CHUNK_OVERLAP invalid")
        if self.retrieval_top_k <= 0 or self.max_pdf_bytes <= 0 or self.relation_timeout_seconds <= 0:
            raise ValueError("Retrieval limit, PDF size and timeout must be positive")
        if "*" in self.cors_origins:
            raise ValueError("Configure explicit CORS_ORIGINS, not wildcard origins")

    @classmethod
    def from_env(cls) -> "Settings":
        from dotenv import load_dotenv
        load_dotenv()
        def required(key: str) -> str:
            value = os.getenv(key, "").strip()
            if not value:
                raise ValueError(f"Missing required configuration: {key}")
            return value
        return cls(
            google_api_key=required("GOOGLE_API_KEY"),
            db_host=required("SUPABASE_DB_HOST"),
            db_port=int(os.getenv("SUPABASE_DB_PORT", "5432")),
            db_user=required("SUPABASE_DB_USER"),
            db_password=required("SUPABASE_DB_PASSWORD"),
            db_name=required("SUPABASE_DB_NAME"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001"),
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "768")),
            extraction_model=os.getenv("EXTRACTION_MODEL", "gemini-2.5-pro"),
            chat_model=os.getenv("CHAT_MODEL", "gemini-2.5-flash"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "100")),
            retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", "20")),
            relation_timeout_seconds=int(os.getenv("RELATION_TIMEOUT_SECONDS", "30")),
            upload_temp_dir=os.getenv("UPLOAD_TEMP_DIR", "temp_uploads"),
            max_pdf_bytes=int(os.getenv("MAX_PDF_BYTES", str(20 * 1024 * 1024))),
            cors_origins=tuple(x.strip() for x in os.getenv(
                "CORS_ORIGINS", "http://localhost:3000"
            ).split(",") if x.strip()),
        )
