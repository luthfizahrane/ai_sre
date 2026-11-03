"""Synthetic end-to-end pipeline test with in-memory storage and fake LLM (no network)."""
import asyncio
from pathlib import Path
from types import SimpleNamespace

from academic_kg.ingestion.processor import DocumentProcessor
from academic_kg.retrieval.retriever import ArticleRetriever
from academic_kg.extraction.extractor import ArticleExtractor
from academic_kg.graph.relation_builder import RelationBuilder
from academic_kg.services.pipeline import Pipeline
from academic_kg.services.status import Stage


class Parser:
    async def extract_text(self, path):
        return "A paper about neural learning techniques and evaluation methods."


class Chunker:
    def split(self, text):
        return [text[:30], text[30:]]


class Embeddings:
    async def embed(self, text):
        return [0.1] * 768


class Store:
    def __init__(self):
        self.chunks = []
        self.node = None
        self.edges = []

    async def save_chunks(self, rows):
        self.chunks = rows

    async def find_similar_chunks(self, doc_id, vector, limit):
        assert limit == 20
        return [r[2] for r in self.chunks]

    async def list_existing_nodes(self):
        return [{"id": "doc_earlier", "att_background": "Neural learning",
                 "att_goal": None, "att_method": None, "att_future": None,
                 "att_gaps": None}]

    async def save_graph(self, node, embedding, edges):
        self.node = node
        self.edges = edges


class LLM:
    def __init__(self):
        self.calls = []

    async def ainvoke(self, prompt):
        self.calls.append(prompt)
        if 'KONTEKS ARTIKEL BARU:' in prompt:
            return SimpleNamespace(content='''{"article_node":{
                "id":"hijacked", "label":"different", "title":"wrong", "type":"wrong",
                "content":"Neural research summary", "att_background":"Neural learning",
                "att_goal":"A", "att_method":"B", "att_future":"C", "att_gaps":"D"
            }}''')
        if 'relasi knowledge graph akademik' in prompt:
            return SimpleNamespace(content='''{"type":"SERUPA_LATAR_BELAKANG", "context":"Shared background"}''')
        raise AssertionError('Unexpected prompt')


def test_end_to_end_synthetic_pipeline_without_external_services():
    async def run():
        storage = Store()
        llm = LLM()
        embeddings = Embeddings()
        pipeline = Pipeline(
            DocumentProcessor(Parser(), Chunker(), embeddings, storage),
            ArticleRetriever(storage, embeddings),
            ArticleExtractor(llm), RelationBuilder(llm),
            storage, embeddings,
        )
        stages = []
        async def report(stage): stages.append(stage)
        result = await pipeline.run(Path('/unused/paper.pdf'), 'abc', 'Test Paper', report)
        assert result == {"document_id":"abc", "edge_count":1}
        assert storage.node.id == 'doc_abc'
        assert storage.node.title == 'Test Paper'
        assert storage.edges[0]['relation'] == 'SERUPA_LATAR_BELAKANG'
        assert len(storage.chunks) == 2
        assert any('Neural learning' in call for call in llm.calls)
        assert stages[-1] == Stage.SAVING
    asyncio.run(run())
