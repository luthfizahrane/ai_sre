"""Document ingestion: parse, chunk, embed and persist chunks."""
import logging

logger = logging.getLogger(__name__)


class DocumentProcessor:
    def __init__(self, parser, chunker, embeddings, repository):
        self.parser = parser
        self.chunker = chunker
        self.embeddings = embeddings
        self.repository = repository

    async def parse(self, file_path: str) -> str:
        return await self.parser.extract_text(file_path)

    async def index(self, text: str, document_id: str) -> int:
        chunks = self.chunker.split(text)
        rows = []
        for index, chunk in enumerate(chunks):
            vec = await self.embeddings.embed(chunk)
            rows.append((f"{document_id}-md-{index}", document_id, chunk, None, vec))
        await self.repository.save_chunks(rows)
        logger.info("Indexed %d chunks for doc_id=%s", len(rows), document_id)
        return len(rows)
