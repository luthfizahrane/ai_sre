"""Google embedding adapter shared by indexing, retrieval and graph storage."""
from academic_kg.core.config import Settings


class Embeddings:
    def __init__(self, model, dimension: int):
        self.model = model
        self.dimension = dimension

    @classmethod
    def from_settings(cls, settings: Settings):
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return cls(GoogleGenerativeAIEmbeddings(
            google_api_key=settings.google_api_key,
            model=settings.embedding_model,
            output_dimensionality=settings.embedding_dim,
        ), settings.embedding_dim)

    async def embed(self, text: str) -> list[float]:
        result = await self.model.aembed_query(text)
        if len(result) != self.dimension:
            raise ValueError("Embedding dimension does not match database schema")
        return result
