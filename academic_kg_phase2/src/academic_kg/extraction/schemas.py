"""Canonical LLM output schemas."""
from typing import Literal
from pydantic import BaseModel, Field


class ArticleMainNode(BaseModel):
    id: str
    label: str
    title: str | None = None
    att_goal: str | None = None
    att_method: str | None = None
    att_background: str | None = None
    att_future: str | None = None
    att_gaps: str | None = None
    type: str
    content: str


class KnowledgeGraphOutput(BaseModel):
    article_node: ArticleMainNode


AllowedRelation = Literal[
    "SERUPA_LATAR_BELAKANG", "SERUPA_TUJUAN", "SERUPA_METODOLOGI",
    "SERUPA_PENELITIAN_LANJUT", "SERUPA_GAP_PENELITIAN",
]


class RelationDecision(BaseModel):
    type: AllowedRelation | None
    context: str = Field(min_length=1)
