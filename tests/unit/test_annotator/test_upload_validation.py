"""Malformed uploads produce clear client errors and preserve legacy files."""

import json
import uuid
from importlib import import_module

import pytest
from sqlmodel import Session, select


@pytest.mark.unit
def test_non_utf8_text_upload_returns_422(annotator_client):
    """A text file with Latin-1 bytes gets a client error instead of a 500."""
    client, _, _ = annotator_client

    response = client.post(
        "/session",
        data={"gazetteer": "geonames", "spacy_model": "en_core_web_sm"},
        files=[("files", ("latin1.txt", b"Z\xfcrich", "text/plain"))],
    )

    assert response.status_code == 422
    assert response.json()["status"] == "error"
    assert "UTF-8" in response.json()["message"]


@pytest.mark.unit
def test_non_utf8_session_upload_returns_422(annotator_client):
    """Session uploads also reject invalid UTF-8 with a readable 422."""
    client, _, _ = annotator_client

    response = client.post(
        "/session/continue/file",
        files={"session_file": ("invalid.json", b"\xff", "application/json")},
    )

    assert response.status_code == 422
    assert response.json()["status"] == "error"
    assert "UTF-8" in response.json()["message"]


@pytest.mark.unit
def test_wrong_shape_session_json_returns_422(annotator_client):
    """Valid JSON without the required document list is a client error."""
    client, _, _ = annotator_client

    response = client.post(
        "/session/continue/file",
        files={
            "session_file": (
                "missing-documents.json",
                b'{"gazetteer":"geonames"}',
                "application/json",
            )
        },
    )

    assert response.status_code == 422
    assert response.json()["status"] == "error"
    assert "session" in response.json()["message"].lower()


@pytest.fixture
def legacy_import_with_schema_error(annotator_client, tmp_path, monkeypatch):
    """Import one valid UTF-8 session beside one invalid legacy session."""
    client, engine, _annotator_app = annotator_client
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    session_routes = import_module("geoparser.annotator.routes.sessions")
    monkeypatch.setattr(
        session_routes, "get_database_location", lambda: legacy_dir / "annotator.db"
    )

    good_file = legacy_dir / "good.json"
    good_payload = {
        "session_id": str(uuid.uuid4()),
        "gazetteer": "geonames",
        "documents": [
            {
                "filename": "zurich.txt",
                "spacy_model": "en_core_web_sm",
                "text": "Zürich",
                "toponyms": [{"text": "Zürich", "start": 0, "end": 6, "loc_id": ""}],
            }
        ],
    }
    good_file.write_bytes(json.dumps(good_payload, ensure_ascii=False).encode("utf-8"))
    bad_file = legacy_dir / "bad.json"
    bad_payload = {
        "session_id": str(uuid.uuid4()),
        "gazetteer": "geonames",
        "documents": [
            {
                "filename": "bad.txt",
                "spacy_model": "en_core_web_sm",
                "text": "Paris",
                "toponyms": [
                    {"text": "Paris", "start": "invalid", "end": 5, "loc_id": ""}
                ],
            }
        ],
    }
    bad_file.write_text(json.dumps(bad_payload), encoding="utf-8")

    response = client.post("/session/read/legacy-files")
    from geoparser.annotator.db.models.document import AnnotatorDocument

    with Session(engine) as db:
        documents = db.exec(select(AnnotatorDocument)).all()
    return response, good_file, bad_file, [document.text for document in documents]


@pytest.mark.unit
def test_legacy_import_reports_schema_failure_counts(legacy_import_with_schema_error):
    response, _, _, _ = legacy_import_with_schema_error
    assert response.status_code == 200
    assert response.json()["files_found"] == 2
    assert response.json()["files_loaded"] == 1
    assert response.json()["files_failed"] == ["bad.json"]


@pytest.mark.unit
def test_legacy_import_removes_only_the_valid_file(legacy_import_with_schema_error):
    _, good_file, bad_file, _ = legacy_import_with_schema_error
    assert not good_file.exists()
    assert bad_file.exists()


@pytest.mark.unit
def test_legacy_import_saves_valid_unicode_documents(legacy_import_with_schema_error):
    _, _, _, document_texts = legacy_import_with_schema_error
    assert document_texts == ["Zürich"]


@pytest.fixture
def legacy_import_with_invalid_utf8(annotator_client, tmp_path, monkeypatch):
    """Attempt to import one undecodable legacy file."""
    client, _, _annotator_app = annotator_client
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    session_routes = import_module("geoparser.annotator.routes.sessions")
    monkeypatch.setattr(
        session_routes, "get_database_location", lambda: legacy_dir / "annotator.db"
    )
    bad_file = legacy_dir / "invalid-utf8.json"
    bad_file.write_bytes(b"{\xff}")

    response = client.post("/session/read/legacy-files")
    return response, bad_file


@pytest.mark.unit
def test_legacy_import_reports_invalid_utf8_counts(legacy_import_with_invalid_utf8):
    response, _ = legacy_import_with_invalid_utf8
    assert response.status_code == 200
    assert response.json()["files_found"] == 1
    assert response.json()["files_loaded"] == 0
    assert response.json()["files_failed"] == ["invalid-utf8.json"]


@pytest.mark.unit
def test_legacy_import_keeps_invalid_utf8_file(legacy_import_with_invalid_utf8):
    _, bad_file = legacy_import_with_invalid_utf8
    assert bad_file.exists()
