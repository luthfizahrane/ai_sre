"""Standalone non-RAG chat endpoint service preserved from Phase 1."""
from academic_kg.extraction.prompts import CHAT_TEMPLATE
from academic_kg.core.response_utils import message_to_text


class ChatService:
    def __init__(self, llm):
        self.llm = llm

    async def ask(self, question: str) -> str:
        return message_to_text(await self.llm.ainvoke(CHAT_TEMPLATE.format(question=question)))
