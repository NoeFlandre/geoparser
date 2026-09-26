"""Malformed uploads produce clear client errors and preserve legacy files."""

import builtins
import json
import sys
import uuid
from importlib import import_module
from pathlib import Path
from types import ModuleType

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select


@pytest.fixture
def client_and_engine(monkeypatch):
    """An annotator client backed by an isolated in-memory database."""
    # These routes never invoke spaCy. Keep the optional ML stack out of this
    # input-validation test so test collection does not need to initialize
    # torch on a mounted HDD.
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
    get_db = import_module("geoparser.annotator.db.db").get_db
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
        with TestClient(annotator_app.app, raise_server_exceptions=False) as client:
            yield client, engine, annotator_app
    finally:
        annotator_app.app.dependency_overrides.pop(get_db, None)
        engine.dispose()


@pytest.mark.unit
def test_non_utf8_text_upload_returns_422(client_and_engine):
    """A text file with Latin-1 bytes gets a client error instead of a 500."""
    client, _, _ = client_and_engine

    response = client.post(
        "/session",
        data={"gazetteer": "geonames", "spacy_model": "en_core_web_sm"},
        files=[("files", ("latin1.txt", b"Z\xfcrich", "text/plain"))],
    )

    assert response.status_code == 422
    assert response.json()["status"] == "error"
    assert "UTF-8" in response.json()["message"]


@pytest.mark.unit
def test_non_utf8_session_upload_returns_422(client_and_engine):
    """Session uploads also reject invalid UTF-8 with a readable 422."""
    client, _, _ = client_and_engine

    response = client.post(
        "/session/continue/file",
        files={"session_file": ("invalid.json", b"\xff", "application/json")},
    )

    assert response.status_code == 422
    assert response.json()["status"] == "error"
    assert "UTF-8" in response.json()["message"]


@pytest.mark.unit
def test_wrong_shape_session_json_returns_422(client_and_engine):
    """Valid JSON without the required document list is a client error."""
    client, _, _ = client_and_engine

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


@pytest.mark.unit
def test_legacy_import_reports_schema_failure_and_keeps_bad_file(
    client_and_engine, tmp_path, monkeypatch
):
    """A bad session is reported while good UTF-8 sessions still load."""
    client, engine, _annotator_app = client_and_engine
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    session_routes = import_module("geoparser.annotator.routes.sessions")
    monkeypatch.setattr(session_routes, "db_location", legacy_dir / "annotator.db")

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

    real_open = builtins.open

    def cp1252_default_open(file, *args, **kwargs):
        if Path(file).resolve() == good_file.resolve() and "encoding" not in kwargs:
            kwargs["encoding"] = "cp1252"
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", cp1252_default_open)

    response = client.post("/session/read/legacy-files")

    assert response.status_code == 200
    assert response.json()["files_found"] == 2
    assert response.json()["files_loaded"] == 1
    assert response.json()["files_failed"] == ["bad.json"]
    assert not good_file.exists()
    assert bad_file.exists()
    from geoparser.annotator.db.models.document import AnnotatorDocument

    with Session(engine) as db:
        documents = db.exec(select(AnnotatorDocument)).all()
    assert [document.text for document in documents] == ["Zürich"]


@pytest.mark.unit
def test_legacy_import_reports_invalid_utf8_and_keeps_file(
    client_and_engine, tmp_path, monkeypatch
):
    """An undecodable legacy file is listed and left available to recover."""
    client, _, _annotator_app = client_and_engine
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    session_routes = import_module("geoparser.annotator.routes.sessions")
    monkeypatch.setattr(session_routes, "db_location", legacy_dir / "annotator.db")
    bad_file = legacy_dir / "invalid-utf8.json"
    bad_file.write_bytes(b"{\xff}")

    response = client.post("/session/read/legacy-files")

    assert response.status_code == 200
    assert response.json()["files_found"] == 1
    assert response.json()["files_loaded"] == 0
    assert response.json()["files_failed"] == ["invalid-utf8.json"]
    assert bad_file.exists()
