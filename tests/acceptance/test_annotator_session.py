from typing import Any

import pytest
from fastapi.testclient import TestClient
from pytest_bdd import given, parsers, scenarios, then, when
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

pytestmark = pytest.mark.acceptance
scenarios("features/annotator_session.feature")


@pytest.fixture
def annotator_state(andorra_gazetteer) -> Any:
    from geoparser.annotator.app import app
    from geoparser.annotator.db.db import get_db

    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    SQLModel.metadata.create_all(engine)

    def override_get_db():
        with Session(engine) as db:
            yield db

    app.dependency_overrides[get_db] = override_get_db
    try:
        with TestClient(app, follow_redirects=False) as client:
            yield {"client": client}
    finally:
        app.dependency_overrides.pop(get_db, None)
        engine.dispose()


def _document_url(state: dict[str, Any]) -> str:
    return f"/session/{state['session_id']}/document/0"


@given("the annotator app is running on a test client")
def running_app(annotator_state: dict[str, Any]) -> None:
    assert annotator_state["client"].get("/").status_code == 200


@when(parsers.parse('I upload "{text}" for the Andorra gazetteer'))
def upload(annotator_state: dict[str, Any], text: str) -> None:
    response = annotator_state["client"].post(
        "/session",
        files=[("files", ("andorra.txt", text.encode("utf-8"), "text/plain"))],
        data={"gazetteer": "andorranames", "spacy_model": "en_core_web_sm"},
    )
    assert response.status_code == 302, response.text
    # Redirects to /session/<id>/document/0/annotate
    annotator_state["session_id"] = response.headers["location"].split("/")[2]
    annotator_state["text"] = text


@when(parsers.parse("I mark characters {start:d} to {end:d} as a toponym"))
def mark(annotator_state: dict[str, Any], start: int, end: int) -> None:
    text = annotator_state["text"][start:end]
    response = annotator_state["client"].post(
        f"{_document_url(annotator_state)}/annotation",
        json={"text": text, "start": start, "end": end},
    )
    assert response.status_code == 200, response.text
    annotator_state["span"] = {"text": text, "start": start, "end": end}


@when(parsers.parse('I select candidate "{identifier}" for that toponym'))
def select(annotator_state: dict[str, Any], identifier: str) -> None:
    client = annotator_state["client"]
    span = annotator_state["span"]
    candidates = client.post(
        f"{_document_url(annotator_state)}/get_candidates",
        json={**span, "query_text": span["text"]},
    )
    assert candidates.status_code == 200, candidates.text
    offered = [candidate["loc_id"] for candidate in candidates.json()["candidates"]]
    assert identifier in offered
    response = client.put(
        f"{_document_url(annotator_state)}/annotation",
        json={**span, "loc_id": identifier},
    )
    assert response.status_code == 200, response.text


@then(
    parsers.parse(
        'the exported session places characters {start:d} to {end:d} at "{identifier}"'
    )
)
def exported(
    annotator_state: dict[str, Any], start: int, end: int, identifier: str
) -> None:
    response = annotator_state["client"].get(
        f"/session/{annotator_state['session_id']}/annotations/download"
    )
    assert response.status_code == 200
    (document,) = response.json()["documents"]
    assert [
        (toponym["start"], toponym["end"], toponym["loc_id"])
        for toponym in document["toponyms"]
    ] == [(start, end, identifier)]
