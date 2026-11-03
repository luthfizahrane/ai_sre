"""Per-worker, in-memory ingestion status registry (not durable job processing)."""
import asyncio
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum


class Stage(str, Enum):
    PENDING = "PENDING"
    PARSING = "PARSING"
    INDEXING = "INDEXING"
    RETRIEVING = "RETRIEVING"
    EXTRACTING = "EXTRACTING"
    LINKING = "LINKING"
    SAVING = "SAVING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class Job:
    document_id: str
    stage: Stage = Stage.PENDING
    error_code: str | None = None
    updated_at: str = ""

    def as_dict(self):
        result = asdict(self)
        result["stage"] = self.stage.value
        return result


class StatusStore:
    def __init__(self, max_items: int = 500):
        self.jobs: dict[str, Job] = {}
        self.lock = asyncio.Lock()
        self.max_items = max_items

    async def update(self, document_id: str, stage: Stage, error_code: str | None = None):
        async with self.lock:
            self.jobs[document_id] = Job(
                document_id, stage, error_code,
                datetime.now(timezone.utc).isoformat(),
            )
            # Bound memory usage. This is not a durable status store.
            while len(self.jobs) > self.max_items:
                self.jobs.pop(next(iter(self.jobs)))

    async def get(self, document_id: str):
        async with self.lock:
            job = self.jobs.get(document_id)
            return job.as_dict() if job else None
