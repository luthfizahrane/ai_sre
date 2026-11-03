import asyncio
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from academic_kg.extraction.schemas import ArticleMainNode
from academic_kg.graph.relation_builder import RelationBuilder
from academic_kg.graph.node_builder import aspect_text, normalize_node
from academic_kg.extraction.extractor import ArticleExtractor
from academic_kg.services.pipeline import Pipeline
from academic_kg.services.status import Stage, StatusStore


def node():
    return ArticleMainNode(id="bad", label="Wrong", title="Model Title", type="bad",
        content="Summary", att_background="Digital education", att_goal="Study techniques")


def test_normalize_model_ids():
    result=normalize_node(node(),"abc","Trusted Name")
    assert (result.id,result.label,result.title,result.type)==(
        "doc_abc","Artikel","Trusted Name","article")
    assert aspect_text(result,"background") == "Digital education"
    assert aspect_text({"att_background":"Tidak ditemukan"},"background") == ""


def test_extraction_validates_and_overrides_model_metadata():
    model=SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(
        content='{"article_node":{"id":"fake","label":"fake","title":"LLM",'
                '"type":"fake","content":"Summary","att_goal":"A",'
                '"att_method":"B","att_background":"C","att_future":"D",'
                '"att_gaps":"E"}}')))
    result=asyncio.run(ArticleExtractor(model).extract("context","abc","Trusted Name",[]))
    assert result.id == "doc_abc" and result.title == "Trusted Name"
    assert model.ainvoke.await_count == 1


def test_relation_null_and_match():
    response=SimpleNamespace(content='{"type":null,"context":"beda"}')
    llm=SimpleNamespace(ainvoke=AsyncMock(return_value=response))
    other=[{"id":"doc_other", "att_background":"Online classrooms", "att_goal":None,
            "att_method":None,"att_future":None,"att_gaps":None}]
    result=asyncio.run(RelationBuilder(llm).discover(node(),other))
    assert result==[] and llm.ainvoke.await_count==1
    response.content='{"type":"SERUPA_LATAR_BELAKANG","context":"similar"}'
    result=asyncio.run(RelationBuilder(llm).discover(node(),other))
    assert len(result)==1 and result[0]["relation"]=="SERUPA_LATAR_BELAKANG"
    prompt=llm.ainvoke.await_args.args[0]
    assert "Digital education" in prompt and "Online classrooms" in prompt


def test_pipeline_status_and_order():
    calls=[]
    class Processor:
        async def parse(self, path):
            calls.append("parse")
            return "paper text"
        async def index(self, text,doc_id):
            calls.append("index")
            return 1
    class Retriever:
        async def retrieve(self,text,doc_id,title):
            calls.append("retrieve")
            return "context"
    class Extractor:
        async def extract(self,context,doc_id,title,existing):
            calls.append("extract")
            return node()
    class Relations:
        async def discover(self,node,existing):
            calls.append("links")
            return []
    class Repo:
        async def list_existing_nodes(self):
            calls.append("existing")
            return []
        async def save_graph(self,node,vector,edges):
            calls.append("save")
    class Embed:
        async def embed(self,text):
            calls.append("embed")
            return [0.1]*768
    async def run():
        stages=[]
        async def report(stage): stages.append(stage)
        pipeline=Pipeline(Processor(),Retriever(),Extractor(),Relations(),Repo(),Embed())
        result=await pipeline.run("/tmp/f.pdf","test","paper.pdf",report)
        return result,stages
    result,stages=asyncio.run(run())
    assert result["document_id"]=="test"
    assert calls == ["parse","index","retrieve","existing","extract","links","embed","save"]
    assert stages==[Stage.PARSING,Stage.INDEXING,Stage.RETRIEVING,Stage.EXTRACTING,Stage.LINKING,Stage.SAVING]


def test_status_store_bounded_and_failure():
    async def run():
        store=StatusStore(max_items=2)
        await store.update("a",Stage.INDEXING)
        await store.update("b",Stage.COMPLETED)
        await store.update("c",Stage.FAILED,"PROCESSING_ERROR")
        assert await store.get("a") is None
        assert (await store.get("c"))["error_code"] == "PROCESSING_ERROR"
    asyncio.run(run())
