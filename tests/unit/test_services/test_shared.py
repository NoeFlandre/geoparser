"""Tests for shared service scaffolding."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from sqlmodel import select

from geoparser.db.crud import RecognizerRepository
from geoparser.db.db import get_session
from geoparser.db.models import Recognizer, RecognizerCreate
from geoparser.services._shared import ensure_module_record, require_fit


@pytest.mark.unit
def test_ensure_module_record_persists_and_reuses_module_configuration():
    """Both service types can share one idempotent module-row helper."""
    module = SimpleNamespace(
        id="test-recognizer",
        name="TestRecognizer",
        config={"model_name": "test/model"},
    )

    first_id = ensure_module_record(RecognizerRepository, RecognizerCreate, module)
    second_id = ensure_module_record(RecognizerRepository, RecognizerCreate, module)

    with get_session() as session:
        rows = session.exec(select(Recognizer)).all()
    assert first_id == second_id == "test-recognizer"
    assert [(row.name, row.config) for row in rows] == [
        ("TestRecognizer", {"model_name": "test/model"})
    ]


@pytest.mark.unit
def test_require_fit_returns_the_module_fit_callable():
    """The shared helper returns the original bound fit method."""
    fit = Mock()
    module = SimpleNamespace(name="TestRecognizer", fit=fit)

    assert require_fit(module, "Recognizer") is fit


@pytest.mark.unit
def test_require_fit_names_modules_without_fit_methods():
    """A missing fit method has the same clear error for both services."""
    module = SimpleNamespace(name="ManualResolver", fit=None)

    with pytest.raises(
        ValueError,
        match="Resolver 'ManualResolver' does not implement a fit method",
    ):
        require_fit(module, "Resolver")
