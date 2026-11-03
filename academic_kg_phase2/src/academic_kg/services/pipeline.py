"""End-to-end orchestration of document and graph operations."""
import logging
from pathlib import Path
from academic_kg.services.status import Stage

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, processor, retriever, extractor, relations, graph_repo, embeddings):
        self.processor = processor
        self.retriever = retriever
        self.extractor = extractor
        self.relations = relations
        self.graph_repo = graph_repo
        self.embeddings = embeddings

    async def run(self, file_path: Path, doc_id: str, title: str, report):
        """Report receives stage changes; any exception propagates to API.

        Chunk indexing is committed separately from graph persistence (same as Phase 1).
        A failed graph stage can leave indexed chunks; no automatic deletion is attempted.
        """
        await report(Stage.PARSING)
        text = await self.processor.parse(str(file_path))
        if not text:
            raise ValueError("PDF_EMPTY_TEXT")
        await report(Stage.INDEXING)
        await self.processor.index(text, doc_id)
        await report(Stage.RETRIEVING)
        context = await self.retriever.retrieve(text, doc_id, title)
        existing = await self.graph_repo.list_existing_nodes()
        await report(Stage.EXTRACTING)
        node = await self.extractor.extract(context, doc_id, title, existing)
        await report(Stage.LINKING)
        edges = await self.relations.discover(node, existing)
        await report(Stage.SAVING)
        node_vector = await self.embeddings.embed(node.content or node.title or "")
        await self.graph_repo.save_graph(node, node_vector, edges)
        logger.info("Pipeline complete for doc_id=%s, edges=%d", doc_id, len(edges))
        return {"document_id": doc_id, "edge_count": len(edges)}
