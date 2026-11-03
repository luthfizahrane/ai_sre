import json
import pytest
from pydantic import ValidationError
from academic_kg.core.config import Settings
from academic_kg.core.response_utils import parse_llm_json, message_to_text
from academic_kg.extraction.prompts import ARTICLE_TEMPLATE, RELATION_ASPECTS, build_relation_prompt
from academic_kg.extraction.schemas import RelationDecision


@pytest.fixture
def settings():
    return Settings(google_api_key="fake", db_host="dev", db_port=5432,
                    db_user="dev", db_password="private", db_name="dev")


def test_settings_validate_dimension_and_overlap(settings):
    assert settings.chunk_size == 800
    with pytest.raises(ValueError, match="768"):
        Settings(**{**settings.__dict__, "embedding_dim": 1024})
    with pytest.raises(ValueError, match="CHUNK_SIZE"):
        Settings(**{**settings.__dict__, "chunk_overlap": 800})


def test_models_are_not_initialized_by_settings(settings):
    assert settings.embedding_model == "models/gemini-embedding-001"


@pytest.mark.parametrize("raw", [
    '{"type": null, "context": "beda"}',
    '```json\n{"type": null, "context": "beda"}\n```',
    '```\n{"type": null, "context": "beda"}\n```',
])
def test_parse_json_null(raw):
    assert RelationDecision.model_validate(parse_llm_json(raw)).type is None


def test_invalid_type_rejected():
    with pytest.raises(ValidationError):
        RelationDecision.model_validate({"type":"DIFFERENT", "context":"no"})


def test_ai_message_blocks():
    class Message:
        content = [{"type":"text", "text":"alpha"}, {"type":"other"}, {"type":"text", "text":"beta"}]
    assert message_to_text(Message()) == "alpha\nbeta"


@pytest.mark.parametrize("relation,description", [(k,v[1]) for k,v in RELATION_ASPECTS.items()])
def test_relation_prompts_include_both_articles(relation,description):
    prompt = build_relation_prompt(relation,description).format(
        article1_id="a", article2_id="b", article1_excerpt="alpha",
        article2_excerpt="beta", format_instruction="schema")
    assert relation in prompt and "alpha" in prompt and "beta" in prompt
    assert "{article1_excerpt}" not in prompt


def test_article_prompt_preserves_original_domain_fields():
    prompt=ARTICLE_TEMPLATE.format(document_title="A", document_id="id", retrieved_context="context",
        existing_nodes_json="[]", format_instruction=json.dumps({"article_node":{}}))
    for term in ["att_goal", "att_method", "att_background", "att_future", "att_gaps"]:
        assert term in prompt
    assert "Konten: context" in prompt and "ID: 'doc_id'" in prompt


def test_secrets_are_not_exposed_by_settings_repr(settings):
    assert 'private' not in repr(settings)
    assert 'fake' not in repr(settings)
