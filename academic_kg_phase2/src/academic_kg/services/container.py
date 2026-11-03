"""Composition root: instantiates one provider model and shared dependencies."""
from dataclasses import dataclass
from academic_kg.core.config import Settings


@dataclass
class Container:
    pipeline: object
    graph_repo: object
    chat: object


def build_container(settings: Settings) -> Container:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from academic_kg.db.connection import ConnectionFactory
    from academic_kg.db.repositories import DocumentRepository, GraphRepository
    from academic_kg.ingestion.pdf_parser import PdfParser
    from academic_kg.ingestion.chunker import TextChunker
    from academic_kg.ingestion.embeddings import Embeddings
    from academic_kg.ingestion.processor import DocumentProcessor
    from academic_kg.retrieval.retriever import ArticleRetriever
    from academic_kg.extraction.extractor import ArticleExtractor
    from academic_kg.graph.relation_builder import RelationBuilder
    from academic_kg.services.chat import ChatService
    from academic_kg.services.pipeline import Pipeline

    conn = ConnectionFactory(settings)
    documents = DocumentRepository(conn)
    graph = GraphRepository(conn)
    embedding = Embeddings.from_settings(settings)
    kg_model = ChatGoogleGenerativeAI(
        google_api_key=settings.google_api_key,
        model=settings.extraction_model, temperature=0.2,
    )
    chat_model = ChatGoogleGenerativeAI(
        google_api_key=settings.google_api_key,
        model=settings.chat_model, temperature=0.8,
    )
    processor = DocumentProcessor(
        PdfParser(), TextChunker(settings.chunk_size, settings.chunk_overlap),
        embedding, documents,
    )
    pipeline = Pipeline(
        processor, ArticleRetriever(documents, embedding, settings.retrieval_top_k),
        ArticleExtractor(kg_model), RelationBuilder(kg_model, settings.relation_timeout_seconds),
        graph, embedding,
    )
    return Container(pipeline=pipeline, graph_repo=graph, chat=ChatService(chat_model))
