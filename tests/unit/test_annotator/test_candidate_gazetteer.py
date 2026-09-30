"""Candidate lookup must use the gazetteer belonging to its URL session."""

from uuid import UUID

import pytest
from sqlmodel import Session

from geoparser.annotator.db.crud import (
    DocumentRepository,
    SessionRepository,
    ToponymRepository,
)
from geoparser.annotator.db.models.document import AnnotatorDocumentCreate
from geoparser.annotator.db.models.session import AnnotatorSessionCreate


def _create_session_with_document(engine, annotator_app, gazetteer: str) -> UUID:
    """Add one session and document so the candidate route can resolve them."""
    with Session(engine) as db:
        session = SessionRepository.create(
            db, AnnotatorSessionCreate(gazetteer=gazetteer)
        )
        DocumentRepository.create(
            db,
            AnnotatorDocumentCreate(
                filename="place.txt",
                spacy_model="en_core_web_sm",
                text="Paris",
            ),
            additional={"session_id": session.id},
        )
        return session.id


@pytest.fixture
def candidate_lookup_after_open_sessions(annotator_client, monkeypatch):
    """Open two sessions, then fetch candidates through the first session URL."""
    client, engine, annotator_app = annotator_client
    session_a = _create_session_with_document(engine, annotator_app, "geonames")
    session_b = _create_session_with_document(engine, annotator_app, "swissnames3d")
    gazetteers = []

    def record_gazetteer(cls, document, gazetteer, request):
        gazetteers.append(gazetteer)
        return {"gazetteer": gazetteer}

    monkeypatch.setattr(
        ToponymRepository,
        "get_candidates",
        classmethod(record_gazetteer),
    )
    assert client.get(f"/session/{session_a}/document/0/annotate").status_code == 200
    assert client.get(f"/session/{session_b}/document/0/annotate").status_code == 200
    response = client.post(f"/session/{session_a}/document/0/get_candidates", json={})
    return response, gazetteers


@pytest.mark.unit
def test_candidate_lookup_uses_its_session_after_another_session_is_opened(
    candidate_lookup_after_open_sessions,
):
    """Opening B after A does not redirect A's candidate query to B's gazetteer."""
    response, gazetteers = candidate_lookup_after_open_sessions
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
        ToponymRepository,
        "get_candidates",
        classmethod(record_gazetteer),
    )
    assert not hasattr(annotator_app, "current_gazetteer_name")

    response = client.post(f"/session/{session_id}/document/0/get_candidates", json={})

    assert response.status_code == 200
    assert gazetteers == ["swissnames3d"]
    assert response.json() == {"gazetteer": "swissnames3d"}
