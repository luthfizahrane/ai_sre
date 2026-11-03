"""RAG retrieval stage (with explicit Phase 1 fallback)."""
import logging

logger = logging.getLogger(__name__)


class ArticleRetriever:
    def __init__(self, repository, embeddings, top_k: int = 20):
        self.repository = repository
        self.embeddings = embeddings
        self.top_k = top_k

    async def retrieve(self, full_text: str, doc_id: str, title: str) -> str:
        try:
            query = f"Judul: {title}. Ringkasan awal: {full_text[:500]}..."
            embedding = await self.embeddings.embed(query)
            chunks = await self.repository.find_similar_chunks(doc_id, embedding, self.top_k)
            if chunks:
                logger.info("Retrieved %d chunks for doc_id=%s", len(chunks), doc_id)
                return "\n\n".join(chunks)
        except Exception as exc:
            logger.warning("Retriever fallback for doc_id=%s (%s)", doc_id, type(exc).__name__)
        return full_text[:2000]
