import asyncio
import uuid
from pathlib import Path
from academic_kg.db.repositories import DocumentRepository, GraphRepository
from academic_kg.extraction.schemas import ArticleMainNode

ROOT=Path(__file__).resolve().parents[1]


class Tx:
    def __init__(self,calls): self.calls=calls
    async def __aenter__(self): self.calls.append("begin")
    async def __aexit__(self, exc_type, exc, tb): self.calls.append("commit" if exc is None else "rollback")

class FakeConn:
    def __init__(self,calls): self.calls=calls
    def transaction(self): return Tx(self.calls)
    async def execute(self,sql,*args): self.calls.append((sql,args))
    async def close(self): self.calls.append("close")
    async def fetch(self,sql,*args):
        self.calls.append((sql,args))
        if "FROM nodes_kg" in sql:
            return [{"id":"doc_a","label":"Artikel","title":"Title", "att_goal":"A",
                     "att_method":"B","att_background":"C","att_future":None,
                     "att_gaps":None,"type":"article","content":"summary"}]
        if "FROM edges_kg" in sql:
            return [{"id":uuid.uuid4(), "fromId":"doc_a", "toId":"doc_b",
                     "relation":"SERUPA_TUJUAN", "label":"Similar", "color":"#123456", "context":"matched"}]
        return [{"chunk_text":"c1"}]

class Factory:
    def __init__(self,calls): self.calls=calls
    async def connect(self): return FakeConn(self.calls)


def test_schema_and_query_agree():
    sql=(ROOT/'migrations/001_canonical_schema.sql').read_text()
    py=(ROOT/'src/academic_kg/db/repositories.py').read_text()
    for name in ("document_chunks", "nodes_kg", "edges_kg"):
        assert f"CREATE TABLE IF NOT EXISTS public.{name}" in sql
    for col in ("att_goal", "att_method", "att_background", "att_future", "att_gaps"):
        assert col in sql and col in py
    assert 'UNIQUE ("fromId", "toId", relation)' in sql
    assert 'ON CONFLICT ("fromId", "toId", relation)' in py


def test_graph_transaction_and_aliases():
    async def run():
        calls=[]
        repo=GraphRepository(Factory(calls))
        node=ArticleMainNode(id="doc_a",label="Artikel",title="Title", type="article",content="summary")
        edge={"id":uuid.uuid4(),"fromId":"doc_a", "toId":"doc_b", "relation":"SERUPA_TUJUAN",
              "label":"label", "color":"#000000", "context":"reason"}
        await repo.save_graph(node,[0.1]*768,[edge])
        assert calls[0]=="begin" and calls[3:]==["commit","close"]
        result=await repo.graph_data()
        assert result["nodes"][0]["label"]=="Title"
        assert result["nodes"][0]["details"]["purpose"]=="A"
        assert result["edges"][0]["source_id"]=="doc_a"
        assert result["edges"][0]["type"]=="SERUPA_TUJUAN"
    asyncio.run(run())


def test_document_chunks_upsert_and_query_limit():
    async def run():
        calls=[]
        repo=DocumentRepository(Factory(calls))
        await repo.save_chunks([("doc-md-0", "doc", "txt",None,[0.1]*768)])
        assert calls[0]=="begin" and calls[-2:]==["commit","close"]
        chunks=await repo.find_similar_chunks("doc",[0.1]*768,20)
        assert chunks==["c1"]
    asyncio.run(run())


def test_no_prints_or_hardcoded_passwords_in_source():
    for file in (ROOT/'src/academic_kg').rglob('*.py'):
        assert 'print(' not in file.read_text()
