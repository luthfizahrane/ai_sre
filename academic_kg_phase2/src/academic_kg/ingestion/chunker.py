"""Plain text chunking using original Phase 1 size and overlap."""


class TextChunker:
    def __init__(self, size: int = 800, overlap: int = 100):
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=size, chunk_overlap=overlap,
        )

    def split(self, text: str) -> list[str]:
        return self.splitter.split_text(text)
