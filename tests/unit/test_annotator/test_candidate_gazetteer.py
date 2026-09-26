"""Candidate lookup must use the gazetteer belonging to its URL session."""

import sys
from importlib import import_module
from types import ModuleType
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from geoparser.annotator.db.models.document import AnnotatorDocumentCreate


@pytest.fixture
def annotator_client(monkeypatch):
    """Build the app with an isolated in-memory database and no ML startup."""
    spacy_package = ModuleType("spacy")
    spacy_package.__path__ = []
    spacy_util = ModuleType("spacy.util")
    spacy_util.get_installed_models = lambda: []
    spacy_package.util = spacy_util
    monkeypatch.setitem(sys.modules, "spacy", spacy_package)
    monkeypatch.setitem(sys.modules, "spacy.util", spacy_util)

    recognizer_module = ModuleType("geoparser.modules.recognizers.spacy")
    recognizer_module.SpacyRecognizer = type("SpacyRecognizer", (), {})
    monkeypatch.setitem(
        sys.modules, "geoparser.modules.recognizers.spacy", recognizer_module
    )

    annotator_app = import_module("geoparser.annotator.app")
    if hasattr(annotator_app, "current_gazetteer_name"):
        annotator_app.current_gazetteer_name = None

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)

    def override_get_db():
        with Session(engine) as db:
            yield db

    annotator_app.app.dependency_overrides[annotator_app.get_db] = override_get_db
    try:
        with TestClient(annotator_app.app, raise_server_exceptions=False) as client:
            yield client, engine, annotator_app
    finally:
        annotator_app.app.dependency_overrides.pop(annotator_app.get_db, None)
        engine.dispose()


def _create_session_with_document(engine, annotator_app, gazetteer: str) -> UUID:
    """Add one session and document so the candidate route can resolve them."""
    with Session(engine) as db:
        session = annotator_app.SessionRepository.create(
            db, annotator_app.AnnotatorSessionCreate(gazetteer=gazetteer)
        )
        annotator_app.DocumentRepository.create(
            db,
            AnnotatorDocumentCreate(
                filename="place.txt",
                spacy_model="en_core_web_sm",
                text="Paris",
            ),
            additional={"session_id": session.id},
        )
        return session.id


@pytest.mark.unit
def test_candidate_lookup_uses_its_session_after_another_session_is_opened(
    annotator_client, monkeypatch
):
    """Opening B after A does not redirect A's candidate query to B's gazetteer."""
    client, engine, annotator_app = annotator_client
    session_a = _create_session_with_document(engine, annotator_app, "geonames")
    session_b = _create_session_with_document(engine, annotator_app, "swissnames3d")
    gazetteers = []

    def record_gazetteer(cls, document, gazetteer, request):
        gazetteers.append(gazetteer)
        return {"gazetteer": gazetteer}

    monkeypatch.setattr(
        annotator_app.ToponymRepository,
        "get_candidates",
        classmethod(record_gazetteer),
    )

    assert client.get(f"/session/{session_a}/document/0/annotate").status_code == 200
    assert client.get(f"/session/{session_b}/document/0/annotate").status_code == 200
    response = client.post(f"/session/{session_a}/document/0/get_candidates", json={})

    assert response.status_code == 200
    assert gazetteers == ["geonames"]
    assert response.json() == {"gazetteer": "geonames"}


@pytest.mark.unit
def test_candidate_lookup_survives_server_restart_without_a_page_load(
    annotator_client, monkeypatch
):
    """The URL session supplies its gazetteer even before any page has loaded."""
    client, engine, annotator_app = annotator_client
    session_id = _create_session_with_document(engine, annotator_app, "swissnames3d")
    gazetteers = []

    def record_gazetteer(cls, document, gazetteer, request):
        gazetteers.append(gazetteer)
        return {"gazetteer": gazetteer}

    monkeypatch.setattr(
        annotator_app.ToponymRepository,
        "get_candidates",
        classmethod(record_gazetteer),
    )
    if hasattr(annotator_app, "current_gazetteer_name"):
        annotator_app.current_gazetteer_name = None

    response = client.post(f"/session/{session_id}/document/0/get_candidates", json={})

    assert response.status_code == 200
    assert gazetteers == ["swissnames3d"]
    assert response.json() == {"gazetteer": "swissnames3d"}
