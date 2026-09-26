"""Factory tests for benchmark pipeline composition."""

import sys
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from scripts.benchmark import pipelines
from scripts.benchmark.__main__ import build_parser


def _patch_module_classes(monkeypatch, **classes):
    """Patch lazy module imports without loading transformer models."""
    package = ModuleType("geoparser")
    package.__path__ = []
    modules = ModuleType("geoparser.modules")

    def get_module_class(name: str) -> Any:
        if name not in classes:
            raise AttributeError(name)
        return classes[name]

    modules.__dict__["__getattr__"] = get_module_class
    gazetteer = ModuleType("geoparser.gazetteer")
    gazetteer.__dict__["Gazetteer"] = Mock()
    package.__dict__["modules"] = modules
    package.__dict__["gazetteer"] = gazetteer
    monkeypatch.setitem(sys.modules, "geoparser", package)
    monkeypatch.setitem(sys.modules, "geoparser.modules", modules)
    monkeypatch.setitem(sys.modules, "geoparser.gazetteer", gazetteer)


def test_hybrid_is_a_distinct_benchmark_pipeline():
    assert pipelines.DEFAULT_PIPELINES == ("upstream", "swapped", "hybrid", "prior")


def test_cli_accepts_the_hybrid_pipeline():
    arguments = build_parser().parse_args(["--pipeline", pipelines.HYBRID])

    assert arguments.pipeline == [pipelines.HYBRID]


def test_hybrid_uses_gliner2_for_recognition(monkeypatch):
    recognizer = SimpleNamespace(model_name="fastino/gliner2.5-multi-v1", model=Mock())
    gliner_factory = Mock(return_value=recognizer)
    _patch_module_classes(monkeypatch, GLiNER2Recognizer=gliner_factory)

    result = pipelines.build_recognizer(pipelines.HYBRID, device="cuda")

    assert result is recognizer
    gliner_factory.assert_called_once_with()
    recognizer.model.to.assert_called_once_with("cuda")


@pytest.mark.parametrize("pipeline", [pipelines.UPSTREAM, "hybrid"])
def test_upstream_and_hybrid_use_the_same_minilm_resolver(monkeypatch, pipeline):
    transformer = Mock()
    resolver = SimpleNamespace(
        model_name="dguzh/geo-all-MiniLM-L6-v2",
        transformer=transformer,
        reranker=None,
    )
    resolver_factory = Mock(return_value=resolver)
    _patch_module_classes(monkeypatch, SentenceTransformerResolver=resolver_factory)
    monkeypatch.setattr(pipelines, "GAZETTEER_NAME", "geonames")

    result = pipelines.build_resolver(pipeline, device="cuda", min_similarity=0.0)

    assert result is resolver
    resolver_factory.assert_called_once_with(
        model_name=pipelines.UPSTREAM_RESOLVER_MODEL,
        gazetteer_name="geonames",
        min_similarity=0.0,
    )
    transformer.to.assert_called_once_with("cuda")


def test_swapped_keeps_jina_resolver(monkeypatch):
    resolver = SimpleNamespace(transformer=Mock(), reranker=Mock())
    jina_factory = Mock(return_value=resolver)
    _patch_module_classes(monkeypatch, JinaResolver=jina_factory)

    result = pipelines.build_resolver(
        pipelines.SWAPPED, device="cpu", min_similarity=0.0
    )

    assert result is resolver
    jina_factory.assert_called_once_with(gazetteer_name="geonames", min_similarity=0.0)
    resolver.transformer.to.assert_called_once_with("cpu")
    resolver.reranker.to.assert_called_once_with("cpu")


def test_unknown_pipeline_is_rejected():
    with pytest.raises(ValueError, match="Unknown benchmark pipeline"):
        pipelines.build_recognizer("unknown", device="cpu")


def test_prior_uses_gliner2_for_recognition(monkeypatch):
    recognizer = SimpleNamespace(model=Mock())
    gliner_factory = Mock(return_value=recognizer)
    _patch_module_classes(monkeypatch, GLiNER2Recognizer=gliner_factory)

    assert pipelines.build_recognizer(pipelines.PRIOR, device="cpu") is recognizer


def test_prior_uses_the_prior_resolver_on_the_upstream_model(monkeypatch):
    resolver = SimpleNamespace(transformer=Mock(), reranker=None)
    prior_factory = Mock(return_value=resolver)
    _patch_module_classes(monkeypatch, PriorResolver=prior_factory)
    monkeypatch.setattr(pipelines, "GAZETTEER_NAME", "geonames")

    result = pipelines.build_resolver(
        pipelines.PRIOR, device="cuda", min_similarity=0.0
    )

    assert result is resolver
    prior_factory.assert_called_once_with(
        model_name=pipelines.UPSTREAM_RESOLVER_MODEL,
        gazetteer_name="geonames",
        min_similarity=0.0,
        population_weight=0.3,
        inflection_fallback=False,
    )
    resolver.transformer.to.assert_called_once_with("cuda")


# One factor at a time: every ablation is hybrid with the settings listed.
ABLATIONS = {
    "trim": (0.0, True),
    "population": (0.1, False),
    "population-0.05": (0.05, False),
    "population-0.2": (0.2, False),
    "population-0.3": (0.3, False),
    "population-0.5": (0.5, False),
    "population-1.0": (1.0, False),
}


def test_ablations_are_offered_but_not_run_by_default():
    """The ablation pipelines are selectable and stay out of a default run."""
    assert set(ABLATIONS) <= set(pipelines.PIPELINES)
    assert not set(ABLATIONS) & set(pipelines.DEFAULT_PIPELINES)
    assert build_parser().parse_args(["--pipeline", "trim"]).pipeline == ["trim"]


@pytest.mark.parametrize(("pipeline", "settings"), sorted(ABLATIONS.items()))
def test_each_ablation_changes_one_setting(monkeypatch, pipeline, settings):
    """Each builds the prior resolver with exactly its settings."""
    weight, fallback = settings
    resolver = SimpleNamespace(transformer=Mock(), reranker=None)
    prior_factory = Mock(return_value=resolver)
    _patch_module_classes(monkeypatch, PriorResolver=prior_factory)
    monkeypatch.setattr(pipelines, "GAZETTEER_NAME", "geonames")

    pipelines.build_resolver(pipeline, device="cpu", min_similarity=0.0)

    prior_factory.assert_called_once_with(
        model_name=pipelines.UPSTREAM_RESOLVER_MODEL,
        gazetteer_name="geonames",
        min_similarity=0.0,
        population_weight=weight,
        inflection_fallback=fallback,
    )


@pytest.mark.parametrize("pipeline", sorted(ABLATIONS))
def test_ablations_share_hybrids_recognizer(monkeypatch, pipeline):
    """Only resolution varies, so recognition is GLiNER2 throughout."""
    recognizer = SimpleNamespace(model=Mock())
    gliner_factory = Mock(return_value=recognizer)
    _patch_module_classes(monkeypatch, GLiNER2Recognizer=gliner_factory)

    assert pipelines.build_recognizer(pipeline, device="cpu") is recognizer
