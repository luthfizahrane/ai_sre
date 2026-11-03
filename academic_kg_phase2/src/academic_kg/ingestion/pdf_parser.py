"""PDF to text. No fake page numbers are claimed by the current parser."""
import asyncio


class PdfParser:
    async def extract_text(self, file_path: str) -> str:
        def _extract():
            from markitdown import MarkItDown
            return (MarkItDown(enable_plugins=False).convert(file_path).text_content or "").strip()
        return await asyncio.to_thread(_extract)
