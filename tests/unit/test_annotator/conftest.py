"""Shared lightweight FastAPI and SQLite fixtures for annotator route tests."""

import sys
from collections.abc import Iterator
from importlib import import_module
from types import ModuleType

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine


def _hide_optional_spacy_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep route tests from initializing spaCy or its model packages."""
    spacy_package = ModuleType("spacy")
    spacy_package.__path__ = []
    spacy_util = ModuleType("spacy.util")
    spacy_util.__dict__["get_installed_models"] = list
    spacy_package.__dict__["util"] = spacy_util
    monkeypatch.setitem(sys.modules, "spacy", spacy_package)
    monkeypatch.setitem(sys.modules, "spacy.util", spacy_util)

    recognizer_module = ModuleType("geoparser.modules.recognizers.spacy")
    recognizer_module.__dict__["SpacyRecognizer"] = type("SpacyRecognizer", (), {})
    monkeypatch.setitem(
        sys.modules, "geoparser.modules.recognizers.spacy", recognizer_module
    )


@pytest.fixture
def follow_redirects(request: pytest.FixtureRequest) -> bool:
    """Default to TestClient's behavior unless a route test opts out."""
    return bool(getattr(request, "param", True))


@pytest.fixture
def annotator_app_context(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[ModuleType, Engine]]:
    """Build the app against an isolated database and restore its overrides."""
    _hide_optional_spacy_imports(monkeypatch)
    annotator_app = import_module("geoparser.annotator.app")
    get_db = import_module("geoparser.annotator.db.db").get_db
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)

    def override_get_db() -> Iterator[Session]:
        with Session(engine) as db:
            yield db

    annotator_app.app.dependency_overrides[get_db] = override_get_db
    try:
        yield annotator_app, engine
    finally:
        annotator_app.app.dependency_overrides.pop(get_db, None)
        engine.dispose()


@pytest.fixture
def annotator_client(
    annotator_app_context: tuple[ModuleType, Engine], follow_redirects: bool
) -> Iterator[tuple[TestClient, Engine, ModuleType]]:
    """Expose the app with per-test redirect behavior and shared teardown."""
    annotator_app, engine = annotator_app_context
    with TestClient(
        annotator_app.app,
        follow_redirects=follow_redirects,
        raise_server_exceptions=False,
    ) as client:
        yield client, engine, annotator_app
