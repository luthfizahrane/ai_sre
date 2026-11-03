"""All SQL interactions live here; schema matches migrations/001_canonical_schema.sql."""
from __future__ import annotations
from typing import Any
from academic_kg.extraction.schemas import ArticleMainNode


class DocumentRepository:
    def __init__(self, connections):
        self.connections = connections

    async def save_chunks(self, rows: list[tuple]) -> None:
        if not rows:
            return
        conn = await self.connections.connect()
        try:
            async with conn.transaction():
                for row in rows:
                    await conn.execute("""
                        INSERT INTO document_chunks (id, doc_id, chunk_text, page_number, embedding)
                        VALUES ($1, $2, $3, $4, $5)
                        ON CONFLICT (id) DO UPDATE SET
                            chunk_text = EXCLUDED.chunk_text,
                            page_number = EXCLUDED.page_number,
                            embedding = EXCLUDED.embedding
                    """, *row)
        finally:
            await conn.close()

    async def find_similar_chunks(self, doc_id: str, vector: list[float], limit: int) -> list[str]:
        conn = await self.connections.connect()
        try:
            records = await conn.fetch("""
                SELECT chunk_text, embedding <-> $1 AS distance
                FROM document_chunks WHERE doc_id = $2
                ORDER BY distance LIMIT $3
            """, vector, doc_id, limit)
            return [r["chunk_text"] for r in records]
        finally:
            await conn.close()


class GraphRepository:
    def __init__(self, connections):
        self.connections = connections

    async def list_existing_nodes(self) -> list[dict]:
        conn = await self.connections.connect()
        try:
            rows = await conn.fetch("""
                SELECT id, title, att_goal, att_method, att_background, att_future, att_gaps
                FROM nodes_kg
            """)
            return [dict(row) for row in rows]
        finally:
            await conn.close()

    async def save_graph(self, node: ArticleMainNode, embedding: list[float], edges: list[dict]) -> None:
        conn = await self.connections.connect()
        try:
            async with conn.transaction():
                await conn.execute("""
                    INSERT INTO nodes_kg
                        (id, label, title, att_goal, att_method, att_background,
                         att_future, att_gaps, type, content, embedding)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    ON CONFLICT (id) DO UPDATE SET
                        label = EXCLUDED.label, title = EXCLUDED.title,
                        att_goal = EXCLUDED.att_goal, att_method = EXCLUDED.att_method,
                        att_background = EXCLUDED.att_background,
                        att_future = EXCLUDED.att_future, att_gaps = EXCLUDED.att_gaps,
                        type = EXCLUDED.type, content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding
                """, node.id, node.label, node.title, node.att_goal,
                     node.att_method, node.att_background, node.att_future,
                     node.att_gaps, node.type, node.content, embedding)
                for edge in edges:
                    await conn.execute("""
                        INSERT INTO edges_kg (id, "fromId", "toId", relation, label, color, context)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                        ON CONFLICT ("fromId", "toId", relation) DO UPDATE SET
                            label = EXCLUDED.label, color = EXCLUDED.color,
                            context = EXCLUDED.context
                    """, edge["id"], edge["fromId"], edge["toId"], edge["relation"],
                         edge["label"], edge["color"], edge["context"])
        finally:
            await conn.close()

    async def graph_data(self) -> dict[str, Any]:
        conn = await self.connections.connect()
        try:
            nodes = await conn.fetch("""
                SELECT id, label, title, att_goal, att_method, att_background,
                       att_future, att_gaps, type, content FROM nodes_kg ORDER BY id
            """)
            edges = await conn.fetch("""
                SELECT id, "fromId", "toId", relation, label, color, context
                FROM edges_kg ORDER BY id
            """)
            node_data = []
            for record in nodes:
                item = dict(record)
                item["node_type_label"] = item["label"]
                item["label"] = item["title"] or item["label"]
                item["details"] = {
                    "background": item["att_background"],
                    "methodology": item["att_method"],
                    "purpose": item["att_goal"],
                    "future_research": item["att_future"],
                    "research_gap": item["att_gaps"],
                }
                node_data.append(item)
            edge_data = []
            for record in edges:
                edge = dict(record)
                edge["id"] = str(edge["id"])
                edge["source_id"] = edge["fromId"]
                edge["target_id"] = edge["toId"]
                edge["type"] = edge["relation"]
                edge_data.append(edge)
            return {"nodes": node_data, "edges": edge_data}
        finally:
            await conn.close()
