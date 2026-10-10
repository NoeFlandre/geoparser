"""The registry's pins and recipes, checked against values read from metadata.

The expected revisions, prompts, poolings and dimensions below were copied from the
Hugging Face API and the sentence-transformers configuration files on 2026-10-09,
not computed from the registry, so a silent edit to the registry fails here.
"""

import dataclasses
import re

import pytest

from scripts.embedding_resolution.models import (
    BGE_M3,
    GEO_MINILM,
    HISTORICAL_SETTINGS,
    JINA_V5_TEXT_SMALL,
    MODELS,
    QWEN3_EMBEDDING_0_6B,
    QWEN3_EMBEDDING_4B,
    EmbeddingModel,
    get_model,
    historical_setting,
)

VERIFIED_REVISIONS = {
    "geo-minilm": "d678769f195c194ffdc6f3735a9360118a855b43",
    "jina-v5-text-small": "dd76d535f5447ca3897a9c893fb1e612ead98192",
    "qwen3-embedding-0.6b": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
    "qwen3-embedding-4b": "5cf2132abc99cad020ac570b19d031efec650f2b",
    "bge-m3": "5617a9f61b028005a4858fdac845db406aefb181",
}
QWEN_QUERY_PROMPT = (
    "Instruct: Given a web search query, retrieve relevant passages that answer "
    "the query\nQuery:"
)


def test_registry_holds_exactly_the_five_compared_models():
    assert {model.key: model.revision for model in MODELS} == VERIFIED_REVISIONS


def test_every_pin_is_a_full_commit_hash():
    for model in MODELS:
        assert re.fullmatch(r"[a-f0-9]{40}", model.revision), model.key


def test_retained_and_added_models_are_labelled_as_the_issue_states():
    statuses = {model.key: model.status for model in MODELS}
    assert statuses == {
        "geo-minilm": "retained",
        "jina-v5-text-small": "retained",
        "qwen3-embedding-0.6b": "added",
        "qwen3-embedding-4b": "added",
        "bge-m3": "added",
    }


def test_dimensions_and_poolings_match_the_pinned_module_configs():
    observed = {
        model.key: (model.dimension, model.pooling, model.normalize) for model in MODELS
    }
    assert observed == {
        "geo-minilm": (384, "mean", True),
        "jina-v5-text-small": (1024, "last_token", True),
        "qwen3-embedding-0.6b": (1024, "last_token", True),
        "qwen3-embedding-4b": (2560, "last_token", True),
        "bge-m3": (1024, "cls", True),
    }


def test_qwen_query_prompt_is_the_documented_instruction_and_documents_have_none():
    for model in (QWEN3_EMBEDDING_0_6B, QWEN3_EMBEDDING_4B):
        assert model.prompt("query") == QWEN_QUERY_PROMPT
        assert model.prompt("document") == ""
        assert model.task is None


def test_jina_uses_the_retrieval_task_with_its_documented_prompts():
    assert JINA_V5_TEXT_SMALL.task == "retrieval"
    assert JINA_V5_TEXT_SMALL.prompt("query") == "Query: "
    assert JINA_V5_TEXT_SMALL.prompt("document") == "Document: "


def test_models_without_documented_prompts_use_none_for_both_roles():
    for model in (GEO_MINILM, BGE_M3):
        assert model.prompt("query") == ""
        assert model.prompt("document") == ""


def test_only_jina_loads_custom_model_code():
    assert {model.key for model in MODELS if model.trust_remote_code} == {
        "jina-v5-text-small"
    }


def test_licences_are_recorded_and_jina_is_flagged_non_commercial():
    assert JINA_V5_TEXT_SMALL.license == "cc-by-nc-4.0"
    assert BGE_M3.license == "mit"
    assert GEO_MINILM.license == "not declared in model metadata"


def test_branch_names_are_refused_as_revisions():
    fields = dataclasses.asdict(GEO_MINILM)
    fields["revision"] = "main"
    with pytest.raises(ValueError, match="full 40-character commit hash"):
        EmbeddingModel(**fields)


def test_get_model_names_the_registered_keys_when_unknown():
    assert get_model("bge-m3") is BGE_M3
    with pytest.raises(KeyError, match="expected one of"):
        get_model("unregistered")


def test_prompt_returns_the_text_for_each_role():
    assert QWEN3_EMBEDDING_4B.prompt("query") == QWEN_QUERY_PROMPT
    assert JINA_V5_TEXT_SMALL.prompt("document") == "Document: "


def test_historical_zero_six_is_recorded_only_for_the_model_that_used_it():
    assert historical_setting("geo-minilm", "similarity", 0.6) is not None
    assert historical_setting("jina-v5-text-small", "similarity", 0.6) is None
    assert historical_setting("bge-m3", "similarity", 0.6) is None


def test_a_historical_value_is_refused_under_a_policy_that_never_used_it():
    assert historical_setting("geo-minilm", "population", 0.6) is None
    assert historical_setting("geo-minilm", "population", 0.0) is not None


def test_historical_settings_are_labelled_with_their_policy_and_origin():
    assert {
        (setting.model, setting.policy, setting.min_similarity)
        for setting in HISTORICAL_SETTINGS
    } == {
        ("geo-minilm", "similarity", 0.6),
        ("geo-minilm", "similarity", 0.0),
        ("geo-minilm", "population", 0.0),
    }
    assert all(setting.origin for setting in HISTORICAL_SETTINGS)


def test_an_uncalibrated_value_is_not_a_historical_setting():
    assert historical_setting("geo-minilm", "similarity", 0.5) is None
