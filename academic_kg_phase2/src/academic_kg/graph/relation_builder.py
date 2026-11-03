"""Pairwise, five-aspect LLM verification; original Phase 1 comparison semantics."""
import asyncio
import json
import logging
import uuid
from academic_kg.core.response_utils import parse_llm_json
from academic_kg.extraction.prompts import RELATION_ASPECTS, build_relation_prompt
from academic_kg.extraction.schemas import RelationDecision
from academic_kg.graph.node_builder import aspect_text

logger = logging.getLogger(__name__)
RELATION_COLORS = {
    "SERUPA_LATAR_BELAKANG": "#1f77b4",
    "SERUPA_TUJUAN": "#ff7f0e",
    "SERUPA_METODOLOGI": "#2ca02c",
    "SERUPA_PENELITIAN_LANJUT": "#d62728",
    "SERUPA_GAP_PENELITIAN": "#9467bd",
}


class RelationBuilder:
    def __init__(self, llm, timeout: int = 30):
        self.llm = llm
        self.timeout = timeout

    async def discover(self, node, existing: list[dict]) -> list[dict]:
        candidates = []
        schema = json.dumps(RelationDecision.model_json_schema(), ensure_ascii=False)
        for other in existing:
            if other["id"] == node.id:
                continue
            for relation, (aspect, description) in RELATION_ASPECTS.items():
                current = aspect_text(node, aspect)
                previous = aspect_text(other, aspect)
                if not current or not previous:
                    continue
                prompt = build_relation_prompt(relation, description).format(
                    article1_id=node.id, article1_excerpt=current,
                    article2_id=other["id"], article2_excerpt=previous,
                    format_instruction=schema,
                )
                try:
                    decision_raw = await asyncio.wait_for(self.llm.ainvoke(prompt), timeout=self.timeout)
                    decision = RelationDecision.model_validate(parse_llm_json(decision_raw))
                    if decision.type == relation:
                        candidates.append({
                            "id": uuid.uuid4(), "fromId": node.id, "toId": other["id"],
                            "relation": relation, "label": relation.replace("_", " ").title(),
                            "color": RELATION_COLORS[relation], "context": decision.context,
                        })
                except Exception as exc:
                    logger.warning("Relation verification skipped for node=%s, relation=%s (%s)",
                                   node.id, relation, type(exc).__name__)
        return candidates
