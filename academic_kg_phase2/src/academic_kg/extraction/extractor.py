"""Research aspect extraction with schema validation and bounded retry."""
import json
import logging
from academic_kg.core.response_utils import parse_llm_json
from academic_kg.extraction.prompts import ARTICLE_TEMPLATE, RETRY_TEMPLATE
from academic_kg.extraction.schemas import KnowledgeGraphOutput
from academic_kg.graph.node_builder import normalize_node

logger = logging.getLogger(__name__)


class ArticleExtractor:
    def __init__(self, llm):
        self.llm = llm

    async def extract(self, context: str, doc_id: str, title: str, existing: list[dict]):
        prompt = ARTICLE_TEMPLATE.format(
            retrieved_context=context, document_id=doc_id,
            document_title=title, existing_nodes_json=json.dumps(existing, ensure_ascii=False),
            format_instruction=json.dumps(KnowledgeGraphOutput.model_json_schema(), ensure_ascii=False),
        )
        raw = await self.llm.ainvoke(prompt)
        node = normalize_node(
            KnowledgeGraphOutput.model_validate(parse_llm_json(raw)).article_node, doc_id, title,
        )
        missing = [key for key in (
            "att_goal", "att_method", "att_background", "att_future", "att_gaps"
        ) if not getattr(node, key)]
        if missing:
            try:
                retry = await self.llm.ainvoke(RETRY_TEMPLATE.format(
                    missing_fields=", ".join(missing), context=context,
                ))
                retry_data = parse_llm_json(retry)
                for field in missing:
                    value = retry_data.get(field)
                    if isinstance(value, str) and value.strip():
                        setattr(node, field, value.strip())
            except Exception as exc:
                logger.warning("Extraction retry invalid for doc_id=%s (%s)", doc_id, type(exc).__name__)
        return node
