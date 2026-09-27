"""Executable acceptance coverage for the spaCy transformer prerequisite."""

from unittest.mock import Mock, patch

import catalogue
import pytest
import spacy
from confection import Config
from pytest_bdd import given, parsers, scenarios, then, when
from spacy.util import registry

from geoparser.modules.recognizers.spacy import SpacyRecognizer

pytestmark = pytest.mark.acceptance
scenarios("features/spacy_transformer_model.feature")

MISSING_FACTORY_ERROR = (
    "[E002] Can't find factory for 'curated_transformer' for language English "
    "(en). This usually happens when spaCy calls `nlp.create_pipe` with a "
    "custom component name that's not registered on the current language class."
)


@pytest.fixture
def model_state() -> dict[str, object]:
    """Hold the model setup and result for one scenario."""
    return {}


@given(
    parsers.parse('the spaCy model "{model_name}" needs the missing transformer plugin')
)
def model_needs_plugin(model_state: dict[str, object], model_name: str) -> None:
    """Configure spaCy to report its missing transformer factory."""
    error = ValueError(MISSING_FACTORY_ERROR)
    model_state["spacy_error"] = error
    model_state["load"] = Mock(side_effect=error)


@when(parsers.parse('I build a recognizer on "{model_name}"'))
def build_recognizer(model_state: dict[str, object], model_name: str) -> None:
    """Build the recognizer through the public constructor."""
    with patch("geoparser.modules.recognizers.spacy.spacy.load", model_state["load"]):
        try:
            model_state["recognizer"] = SpacyRecognizer(model_name=model_name)
        except ValueError as error:
            model_state["error"] = error


@then(parsers.parse('I am told to install "{plugin}"'))
def told_to_install(model_state: dict[str, object], plugin: str) -> None:
    """Assert that the message names the missing plugin."""
    error = model_state["error"]
    assert isinstance(error, ValueError)
    assert plugin in str(error)


@then("I am still shown the original spaCy error")
def original_error_kept(model_state: dict[str, object]) -> None:
    """Assert that the original exception remains available as the cause."""
    error = model_state["error"]
    assert isinstance(error, ValueError)
    assert "[E002]" in str(error)
    assert error.__cause__ is model_state["spacy_error"]


def test_real_spacy_loader_reaches_missing_factory_fallback(tmp_path, monkeypatch):
    """Exercise the fallback through spaCy's real local-model loader."""
    monkeypatch.setattr(registry._entry_point_factories, "get_all", dict)
    factory_key = (*registry.factories.namespace, "curated_transformer")
    monkeypatch.delitem(catalogue.REGISTRY, factory_key, raising=False)

    nlp = spacy.blank("en")
    model_path = tmp_path / "local_transformer_model"
    nlp.to_disk(model_path)
    config = Config().from_disk(model_path / "config.cfg")
    config["nlp"]["pipeline"] = ["curated_transformer"]
    config["components"]["curated_transformer"] = {"factory": "curated_transformer"}
    config.to_disk(model_path / "config.cfg")

    with pytest.raises(ValueError) as raised:
        SpacyRecognizer(model_name=str(model_path))

    assert "spacy-curated-transformers" in str(raised.value)
    assert "Can't find factory for 'curated_transformer'" in str(raised.value)
    assert raised.value.__cause__ is not None
