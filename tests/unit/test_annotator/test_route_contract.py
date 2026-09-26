"""TestClient coverage for the annotator's public routes and import/export path."""

import json
import sys
from importlib import import_module
from pathlib import Path
from types import ModuleType
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from geoparser.annotator.db.crud import DocumentRepository
from geoparser.annotator.db.db import get_db


@pytest.fixture
def annotator_client(monkeypatch):
    """Build the current app with an isolated database and no ML startup."""
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
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)

    def override_get_db():
        with Session(engine) as db:
            yield db

    annotator_app.app.dependency_overrides[get_db] = override_get_db
    try:
        with TestClient(
            annotator_app.app,
            follow_redirects=False,
            raise_server_exceptions=False,
        ) as client:
            yield client, engine, annotator_app
    finally:
        annotator_app.app.dependency_overrides.pop(get_db, None)
        engine.dispose()


EXPECTED_ROUTE_MAP = {
    ("/openapi.json", ("GET", "HEAD")),
    ("/docs", ("GET", "HEAD")),
    ("/docs/oauth2-redirect", ("GET", "HEAD")),
    ("/redoc", ("GET", "HEAD")),
    ("/static", ()),
    ("/", ("GET",)),
    ("/start_new_session", ("GET",)),
    ("/continue_session", ("GET",)),
    ("/session/{session_id}/document/{doc_index}/annotate", ("GET",)),
    ("/session", ("POST",)),
    ("/session/read/legacy-files", ("POST",)),
    ("/session/continue/cached", ("POST",)),
    ("/session/continue/file", ("POST",)),
    ("/session/{session_id}", ("DELETE",)),
    ("/session/{session_id}/documents", ("POST",)),
    ("/session/{session_id}/documents", ("GET",)),
    ("/session/{session_id}/document/{doc_index}/parse", ("POST",)),
    ("/session/{session_id}/document/{doc_index}/progress", ("GET",)),
    ("/session/{session_id}/document/{doc_index}/text", ("GET",)),
    ("/session/{session_id}/document/{doc_index}", ("DELETE",)),
    ("/session/{session_id}/document/{doc_index}/get_candidates", ("POST",)),
    ("/session/{session_id}/document/{doc_index}/annotation", ("POST",)),
    ("/session/{session_id}/annotations/download", ("GET",)),
    ("/session/{session_id}/document/{doc_index}/annotation", ("PUT",)),
    ("/session/{session_id}/document/{doc_index}/annotation", ("PATCH",)),
    ("/session/{session_id}/document/{doc_index}/annotation", ("DELETE",)),
    ("/session/{session_id}/settings", ("GET",)),
    ("/session/{session_id}/settings", ("PUT",)),
}


def test_app_route_map_is_pinned(annotator_client):
    """The refactor preserves every registered method and URL path."""
    _, _, annotator_app = annotator_client
    actual = {
        (
            route.path,
            tuple(sorted(getattr(route, "methods", None) or ())),
        )
        for route in annotator_app.app.routes
    }

    assert actual == EXPECTED_ROUTE_MAP


def test_annotator_api_round_trip_and_route_statuses(annotator_client, monkeypatch):
    """Exercise session, document, annotation and settings routes end to end."""
    client, _engine, _annotator_app = annotator_client

    assert client.get("/").status_code == 200
    assert client.get("/start_new_session").status_code == 200
    assert client.get("/continue_session").status_code == 200
    missing_session = UUID("00000000-0000-0000-0000-000000000000")
    assert client.get(f"/session/{missing_session}/documents").status_code == 404
    assert (
        client.get(f"/session/{missing_session}/annotations/download").status_code
        == 404
    )
    assert client.get(f"/session/{missing_session}/settings").status_code == 404
    assert (
        client.put(
            f"/session/{missing_session}/settings",
            json={
                "auto_close_annotation_modal": False,
                "one_sense_per_discourse": True,
            },
        ).status_code
        == 404
    )
    assert client.delete(f"/session/{missing_session}").status_code == 404
    assert (
        client.post(
            "/session/continue/cached", data={"session_id": str(missing_session)}
        ).status_code
        == 302
    )
    assert (
        client.get(f"/session/{missing_session}/document/0/annotate").status_code == 302
    )

    created = client.post(
        "/session",
        data={"gazetteer": "geonames", "spacy_model": "en_core_web_sm"},
        files=[("files", ("paris.txt", b"Paris", "text/plain"))],
    )
    assert created.status_code == 302
    session_id = UUID(created.headers["location"].split("/")[2])
    assert client.get(f"/session/{session_id}/document/0/annotate").status_code == 200
    assert (
        client.post(
            "/session/continue/cached", data={"session_id": str(session_id)}
        ).status_code
        == 302
    )
    assert (
        client.post(
            f"/session/{session_id}/documents", data={"spacy_model": "en_core_web_sm"}
        ).status_code
        == 422
    )

    missing_document = f"/session/{session_id}/document/9"
    invalid_document_responses = [
        client.post(f"{missing_document}/parse"),
        client.get(f"{missing_document}/progress"),
        client.get(f"{missing_document}/text"),
        client.delete(missing_document),
        client.post(f"{missing_document}/get_candidates", json={}),
        client.post(
            f"{missing_document}/annotation",
            json={"text": "Paris", "start": 0, "end": 5},
        ),
        client.put(
            f"{missing_document}/annotation",
            json={"text": "Paris", "start": 0, "end": 5, "loc_id": "2988507"},
        ),
        client.patch(
            f"{missing_document}/annotation",
            json={
                "old_start": 0,
                "old_end": 5,
                "new_start": 0,
                "new_end": 5,
                "new_text": "Paris",
            },
        ),
        client.delete(f"{missing_document}/annotation", params={"start": 0, "end": 5}),
    ]
    assert [response.status_code for response in invalid_document_responses] == [
        422
    ] * len(invalid_document_responses)

    added = client.post(
        f"/session/{session_id}/documents",
        data={"spacy_model": "en_core_web_sm"},
        files=[("files", ("berlin.txt", b"Berlin", "text/plain"))],
    )
    assert added.status_code == 200
    documents = client.get(f"/session/{session_id}/documents")
    assert documents.status_code == 200
    assert [document["filename"] for document in documents.json()] == [
        "paris.txt",
        "berlin.txt",
    ]

    def fake_parse(cls, db, document_id):
        document = cls.read(db, document_id)
        document.spacy_applied = True
        db.add(document)
        db.commit()
        db.refresh(document)
        return document

    monkeypatch.setattr(DocumentRepository, "parse", classmethod(fake_parse))
    document_url = f"/session/{session_id}/document/0"
    assert client.post(f"{document_url}/parse").json() == {
        "status": "success",
        "message": None,
        "parsed": True,
    }
    assert client.get(f"{document_url}/progress").status_code == 200
    assert client.get(f"{document_url}/text").json()["pre_annotated_text"] == "Paris"
    assert client.get(f"/session/{session_id}/document/99/progress").status_code == 422
    assert (
        client.post(
            f"/session/{session_id}/documents", data={"spacy_model": "en_core_web_sm"}
        ).status_code
        == 422
    )

    assert (
        client.post(
            f"{document_url}/annotation",
            json={"text": "Paris", "start": 0, "end": 5},
        ).status_code
        == 200
    )
    assert (
        client.put(
            f"{document_url}/annotation",
            json={"text": "Paris", "start": 0, "end": 5, "loc_id": "2988507"},
        ).status_code
        == 200
    )
    assert (
        client.patch(
            f"{document_url}/annotation",
            json={
                "old_start": 0,
                "old_end": 5,
                "new_start": 0,
                "new_end": 5,
                "new_text": "Paris",
            },
        ).status_code
        == 200
    )
    assert (
        client.patch(
            f"{document_url}/annotation",
            json={
                "old_start": 20,
                "old_end": 25,
                "new_start": 20,
                "new_end": 25,
                "new_text": "missing",
            },
        ).status_code
        == 404
    )

    download = client.get(f"/session/{session_id}/annotations/download")
    assert download.status_code == 200
    session_json = download.json()
    round_trip = {
        "gazetteer": session_json["gazetteer"],
        "documents": [
            {
                "filename": document["filename"],
                "spacy_model": document["spacy_model"],
                "spacy_applied": document["spacy_applied"],
                "text": document["text"],
                "toponyms": [
                    {
                        "text": toponym["text"],
                        "start": toponym["start"],
                        "end": toponym["end"],
                        "loc_id": toponym["loc_id"],
                    }
                    for toponym in document["toponyms"]
                ],
            }
            for document in session_json["documents"]
        ],
    }
    golden_path = Path(__file__).parent / "fixtures" / "annotations_round_trip.json"
    assert round_trip == json.loads(golden_path.read_text(encoding="utf-8"))

    imported = client.post(
        "/session/continue/file",
        files={
            "session_file": ("annotations.json", download.content, "application/json")
        },
    )
    assert imported.status_code == 302
    imported_id = UUID(imported.headers["location"].split("/")[2])
    assert client.get(f"/session/{imported_id}/settings").status_code == 200
    assert (
        client.put(
            f"/session/{imported_id}/settings",
            json={
                "auto_close_annotation_modal": False,
                "one_sense_per_discourse": True,
            },
        ).status_code
        == 200
    )

    assert client.delete(f"/session/{session_id}/document/1").status_code == 200
    assert (
        client.delete(
            f"/session/{session_id}/document/0/annotation?start=0&end=5"
        ).status_code
        == 200
    )
    assert client.delete(f"/session/{session_id}").status_code == 200
    assert client.delete(f"/session/{missing_session}").status_code == 404
    assert client.delete(f"/session/{imported_id}").status_code == 200
