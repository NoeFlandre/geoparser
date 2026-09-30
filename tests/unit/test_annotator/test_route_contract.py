"""TestClient coverage for the annotator's public routes and import/export path."""

import json
from importlib import import_module
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from sqlmodel import Session

from geoparser.annotator.db.crud import DocumentRepository, SessionRepository
from geoparser.annotator.db.models.session import AnnotatorSessionCreate

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


def _registered_routes(app, route_modules: tuple[Any, ...]) -> list[Any]:
    """Include FastAPI's top-level route objects and each included router."""
    routes = [route for route in app.routes if hasattr(route, "path")]
    for module in route_modules:
        routes.extend(module.router.routes)
    return routes


def _route_map(routes: list[Any]) -> set[tuple[str, tuple[str, ...]]]:
    """Normalize route objects to stable path and method pairs."""
    return {
        (route.path, tuple(sorted(getattr(route, "methods", None) or ())))
        for route in routes
    }


def test_app_route_map_is_pinned(annotator_client):
    """The refactor preserves every registered method and URL path."""
    _, _, annotator_app = annotator_client
    from geoparser.annotator.routes import (
        annotations,
        documents,
        pages,
        sessions,
        settings,
    )

    # FastAPI can keep included routers as wrappers instead of flattening them.
    route_modules = (pages, sessions, documents, annotations, settings)
    actual = _route_map(_registered_routes(annotator_app.app, route_modules))

    assert actual == EXPECTED_ROUTE_MAP


@pytest.mark.parametrize("follow_redirects", [False], indirect=True)
def test_annotator_api_round_trip_and_route_statuses(annotator_client, monkeypatch):
    """Exercise session, document, annotation and settings routes end to end."""
    client, engine, _annotator_app = annotator_client
    missing_session = UUID("00000000-0000-0000-0000-000000000000")
    _assert_session_routes_and_redirects(client, missing_session)
    _assert_empty_session_routes(client, engine)
    session_id = _create_paris_session(client)
    _assert_missing_document_routes(client, session_id)
    document_url = _add_berlin_and_parse(client, session_id, monkeypatch)
    _assert_annotation_updates(client, document_url)
    imported_id = _assert_annotation_export_import(client, session_id)
    _assert_session_deletions(client, missing_session, session_id, imported_id)


def _assert_session_routes_and_redirects(client, missing_session: UUID) -> None:
    """Keep the app landing pages, missing-session responses, and redirects stable."""
    statuses = [
        client.get("/").status_code,
        client.get("/start_new_session").status_code,
        client.get("/continue_session").status_code,
        client.get(f"/session/{missing_session}/documents").status_code,
        client.get(f"/session/{missing_session}/annotations/download").status_code,
        client.get(f"/session/{missing_session}/settings").status_code,
        client.put(
            f"/session/{missing_session}/settings",
            json={
                "auto_close_annotation_modal": False,
                "one_sense_per_discourse": True,
            },
        ).status_code,
        client.delete(f"/session/{missing_session}").status_code,
        client.post(
            "/session/continue/cached", data={"session_id": str(missing_session)}
        ).status_code,
        client.post("/session/continue/file").status_code,
        client.get(f"/session/{missing_session}/document/0/annotate").status_code,
    ]
    assert statuses == [
        200,
        200,
        200,
        404,
        404,
        404,
        404,
        404,
        302,
        302,
        302,
    ]


def _assert_empty_session_routes(client, engine) -> None:
    """Check the empty-session annotation page and its document redirect."""
    with Session(engine) as db:
        empty_session = SessionRepository.create(
            db, AnnotatorSessionCreate(gazetteer="geonames")
        )
    empty_document_page = f"/session/{empty_session.id}/document/0/annotate"
    assert client.get(empty_document_page).status_code == 200
    redirect_to_first_document = client.get(
        f"/session/{empty_session.id}/document/1/annotate"
    )
    assert redirect_to_first_document.status_code == 302
    assert redirect_to_first_document.headers["location"].endswith(
        f"/session/{empty_session.id}/document/0/annotate"
    )


def _create_paris_session(client) -> UUID:
    """Create a session with one document and validate cached continuation."""
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
    return session_id
    return session_id


def _assert_missing_document_routes(client, session_id: UUID) -> None:
    """Exercise invalid-document handling across the document route methods."""
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


def _add_berlin_and_parse(client, session_id: UUID, monkeypatch) -> str:
    """Add a document and verify parsing, progress, and annotation routes."""
    added = client.post(
        f"/session/{session_id}/documents",
        data={"spacy_model": "en_core_web_sm"},
        files=[("files", ("berlin.txt", b"Berlin", "text/plain"))],
    )
    documents = client.get(f"/session/{session_id}/documents")
    assert (added.status_code, documents.status_code) == (200, 200)
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
    _assert_document_parse_routes(client, document_url, session_id)
    return document_url


def _assert_document_parse_routes(client, document_url: str, session_id: UUID) -> None:
    """Validate parse state transitions and neighboring document endpoints."""
    missing_annotation = client.delete(
        f"{document_url}/annotation", params={"start": 1, "end": 3}
    )
    first_parse = client.post(f"{document_url}/parse")
    second_parse = client.post(f"{document_url}/parse")
    progress = client.get(f"{document_url}/progress")
    text = client.get(f"{document_url}/text")
    invalid_progress = client.get(f"/session/{session_id}/document/99/progress")
    extra_document = client.post(
        f"/session/{session_id}/documents", data={"spacy_model": "en_core_web_sm"}
    )

    assert [
        missing_annotation.status_code,
        first_parse.status_code,
        second_parse.status_code,
        progress.status_code,
        text.status_code,
        invalid_progress.status_code,
        extra_document.status_code,
    ] == [404, 200, 200, 200, 200, 422, 422]
    assert first_parse.json() == {
        "status": "success",
        "message": None,
        "parsed": True,
    }
    assert second_parse.json()["parsed"] is False
    assert text.json()["pre_annotated_text"] == "Paris"


def _assert_annotation_updates(client, document_url: str) -> None:
    """Check annotation create, update, move, and invalid-move responses."""
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


def _assert_annotation_export_import(client, session_id: UUID) -> UUID:
    """Round-trip an annotated session through the JSON download and import API."""
    download = client.get(f"/session/{session_id}/annotations/download")
    assert download.status_code == 200
    session_json = download.json()
    round_trip = _round_trip_payload(session_json)
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
    settings_read = client.get(f"/session/{imported_id}/settings")
    settings_write = client.put(
        f"/session/{imported_id}/settings",
        json={
            "auto_close_annotation_modal": False,
            "one_sense_per_discourse": True,
        },
    )
    assert (settings_read.status_code, settings_write.status_code) == (200, 200)
    return imported_id


def _round_trip_payload(session_json: dict[str, Any]) -> dict[str, Any]:
    """Project the exported session onto the stable round-trip fixture fields."""
    return {
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


def _assert_session_deletions(
    client, missing_session: UUID, session_id: UUID, imported_id: UUID
) -> None:
    """Check document and session deletion status codes."""
    statuses = [
        client.delete(f"/session/{session_id}/document/1").status_code,
        client.delete(
            f"/session/{session_id}/document/0/annotation?start=0&end=5"
        ).status_code,
        client.delete(f"/session/{session_id}").status_code,
        client.delete(f"/session/{missing_session}").status_code,
        client.delete(f"/session/{imported_id}").status_code,
    ]
    assert statuses == [200, 200, 200, 404, 200]


def test_legacy_import_with_no_files_returns_empty_result(
    annotator_client, monkeypatch, tmp_path
):
    """Legacy import succeeds cleanly when the configured directory is empty."""
    client, _engine, _annotator_app = annotator_client
    empty_legacy_dir = tmp_path / "no-legacy-files"
    empty_legacy_dir.mkdir()
    sessions_routes = import_module("geoparser.annotator.routes.sessions")
    monkeypatch.setattr(
        sessions_routes,
        "get_database_location",
        lambda: empty_legacy_dir / "annotator.db",
    )

    response = client.post("/session/read/legacy-files")

    assert response.status_code == 200
    assert response.json() == {
        "status": "success",
        "message": None,
        "files_found": 0,
        "files_loaded": 0,
        "files_failed": [],
    }
