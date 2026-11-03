"""Domain mapping and system-controlled article node identity."""
from academic_kg.extraction.schemas import ArticleMainNode

ASPECT_FIELDS = {
    "background": "att_background",
    "purpose": "att_goal",
    "methodology": "att_method",
    "future_research": "att_future",
    "research_gap": "att_gaps",
}


def aspect_text(node: ArticleMainNode | dict, aspect: str) -> str:
    field = ASPECT_FIELDS[aspect]
    value = node.get(field) if isinstance(node, dict) else getattr(node, field)
    if not isinstance(value, str):
        return ""
    value = value.strip()
    return "" if value.casefold() == "tidak ditemukan" else value


def normalize_node(node: ArticleMainNode, doc_id: str, title: str) -> ArticleMainNode:
    """Reject generated IDs, labels and titles supplied by a model."""
    return node.model_copy(update={
        "id": f"doc_{doc_id}", "title": title, "type": "article", "label": "Artikel",
    })
