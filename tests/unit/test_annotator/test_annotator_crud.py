"""Focused database repository tests for the annotator."""

import json
import uuid
from io import BytesIO
from types import SimpleNamespace
from typing import Any, cast

import pytest
from fastapi import UploadFile
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from geoparser.annotator.db.crud import (
    DocumentRepository,
    SessionRepository,
    SessionSettingsRepository,
    ToponymRepository,
)
from geoparser.annotator.db.crud import document as document_crud
from geoparser.annotator.db.crud import toponym as toponym_crud
from geoparser.annotator.db.models.document import AnnotatorDocumentCreate
from geoparser.annotator.db.models.session import AnnotatorSessionCreate
from geoparser.annotator.db.models.settings import (
    AnnotatorSessionSettingsCreate,
    AnnotatorSessionSettingsUpdate,
)
from geoparser.annotator.db.models.toponym import (
    AnnotatorToponymCreate,
    AnnotatorToponymUpdate,
)
from geoparser.annotator.exceptions import (
    DocumentNotFoundException,
    InvalidUploadException,
    ToponymNotFoundException,
    ToponymOverlapException,
)
from geoparser.annotator.models.api import CandidatesGet


@pytest.fixture
def db_session():
    """Create a fresh SQLite database for each repository test."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


def test_session_repository_exports_and_updates_child_rows(db_session):
    """Session CRUD preserves related documents, toponyms, and settings."""
    session = SessionRepository.create(
        db_session,
        AnnotatorSessionCreate(
            gazetteer="geonames",
            documents=[
                AnnotatorDocumentCreate(
                    filename="paris.txt",
                    spacy_model="en_core_web_sm",
                    text="Paris",
                    toponyms=[
                        AnnotatorToponymCreate(
                            text="Paris", start=0, end=5, loc_id="2988507"
                        )
                    ],
                )
            ],
        ),
    )

    assert SessionRepository.read_all(db_session, gazetteer="geonames") == [session]
    assert SessionRepository.read(db_session, session.id).documents[0].doc_index == 0
    exported = SessionRepository.read_to_json(db_session, session.id)
    assert exported["documents"][0]["toponyms"][0]["loc_id"] == "2988507"

    updated = SessionSettingsRepository.update(
        db_session,
        AnnotatorSessionSettingsUpdate(
            id=session.settings.id, auto_close_annotation_modal=False
        ),
    )
    assert updated.auto_close_annotation_modal is False
    assert SessionSettingsRepository.read(db_session, updated.id) == updated
    assert SessionSettingsRepository.read_all(db_session, id=updated.id) == [updated]
    assert SessionSettingsRepository.delete(db_session, updated.id).id == updated.id


def test_document_repository_upload_validation_and_reindexing(db_session):
    """Uploads decode as UTF-8 and deleting a row keeps document indices dense."""
    session = SessionRepository.create(
        db_session, AnnotatorSessionCreate(gazetteer="geonames")
    )
    files = [
        UploadFile(filename="first.txt", file=BytesIO("Zürich".encode())),
        UploadFile(filename="second.txt", file=BytesIO(b"Paris")),
    ]
    DocumentRepository.validate_text_files(files)
    assert [file.file.tell() for file in files] == [0, 0]

    documents = DocumentRepository.create_from_text_files(
        db_session, files, session.id, "en_core_web_sm"
    )
    assert [(document.doc_index, document.text) for document in documents] == [
        (0, "Zürich"),
        (1, "Paris"),
    ]

    DocumentRepository.delete(db_session, documents[0].id)
    assert [document.doc_index for document in session.documents] == [0]
    assert session.documents[0].filename == "second.txt"

    invalid_file = UploadFile(filename="invalid.txt", file=BytesIO(b"\xff"))
    with pytest.raises(InvalidUploadException, match="UTF-8"):
        DocumentRepository.validate_text_files([invalid_file])


def test_create_from_text_files_applies_recognizer_and_accepts_none_results(
    db_session, monkeypatch
):
    """Uploaded documents keep recognized spans and tolerate a None prediction."""
    session = SessionRepository.create(
        db_session, AnnotatorSessionCreate(gazetteer="geonames")
    )

    class Recognizer:
        def __init__(self, model_name):
            self.model_name = model_name
            self.calls = 0

        def predict(self, texts):
            self.calls += 1
            return [None] if self.calls == 1 else [[(0, 5)]]

    monkeypatch.setattr(document_crud, "SpacyRecognizer", Recognizer)
    files = [
        UploadFile(filename="unparsed.txt", file=BytesIO(b"Paris")),
        UploadFile(filename="parsed.txt", file=BytesIO(b"Paris")),
    ]

    documents = DocumentRepository.create_from_text_files(
        db_session,
        files,
        session.id,
        "en_core_web_sm",
        apply_spacy=True,
    )

    assert [document.spacy_applied for document in documents] == [True, True]
    assert [
        [toponym.text for toponym in document.toponyms] for document in documents
    ] == [[], ["Paris"]]


def test_toponym_repository_checks_overlap_and_supports_update_delete(db_session):
    """Toponym writes reject overlaps and apply valid edits by row ID."""
    session = SessionRepository.create(
        db_session,
        AnnotatorSessionCreate(
            gazetteer="geonames",
            documents=[
                AnnotatorDocumentCreate(
                    filename="places.txt",
                    spacy_model="en_core_web_sm",
                    text="Paris Berlin",
                    toponyms=[AnnotatorToponymCreate(text="Paris", start=0, end=5)],
                )
            ],
        ),
    )
    document = session.documents[0]
    toponym = document.toponyms[0]

    with pytest.raises(ToponymOverlapException):
        ToponymRepository.create(
            db_session,
            AnnotatorToponymCreate(text="aris", start=1, end=5),
            additional={"document_id": document.id},
        )

    updated = ToponymRepository.update(
        db_session,
        AnnotatorToponymUpdate(id=toponym.id, loc_id="2988507"),
    )
    assert updated.loc_id == "2988507"
    assert ToponymRepository.read(db_session, toponym.id).loc_id == "2988507"
    assert ToponymRepository.delete(db_session, toponym.id).id == toponym.id
    assert ToponymRepository.read_all(db_session, document_id=document.id) == []


def test_missing_document_progress_uses_repository_not_found_error(db_session):
    """A missing row from the progress helper uses the CRUD error contract."""
    with pytest.raises(DocumentNotFoundException):
        DocumentRepository.get_document_progress(db_session, uuid.uuid4())


@pytest.mark.parametrize(
    ("data", "gazetteer", "expected"),
    [
        ({}, "geonames", "place:1"),
        ({"name": "Paris"}, "unknown", "place:1"),
        ({"name": None, "feature_name": None}, "geonames", "place:1"),
        (
            {
                "name": "Paris",
                "feature_name": "PPL",
                "country_name": "France",
                "admin1_name": "Ile-de-France",
                "admin2_name": "Paris",
            },
            "geonames",
            "Paris (PPL) in Paris, Ile-de-France, France",
        ),
    ],
)
def test_location_description_uses_mapped_attributes_and_fallbacks(
    data, gazetteer, expected
):
    """Description generation handles known mappings and its identifier fallback."""
    feature = SimpleNamespace(data=data, identifier="place:1")

    assert (
        ToponymRepository._generate_location_description(cast(Any, feature), gazetteer)
        == expected
    )


def test_candidate_descriptions_include_existing_location(monkeypatch):
    """Candidate descriptions preserve feature attributes and append annotations."""
    from geoparser.annotator.db.crud import toponym as toponym_module

    candidate = SimpleNamespace(
        identifier="candidate:1", data={"name": "Paris"}, geometry=None
    )
    existing = SimpleNamespace(
        identifier="existing:1", data={"name": "Lyon"}, geometry=None
    )
    gazetteer = SimpleNamespace(
        search=lambda text, method: [candidate], find=lambda loc_id: existing
    )
    monkeypatch.setattr(toponym_module, "get_gazetteer", lambda name: gazetteer)
    monkeypatch.setattr(
        ToponymRepository,
        "_get_wgs84_coordinates",
        classmethod(lambda cls, feature: (48.0, 2.0)),
    )

    descriptions, existing_appended = ToponymRepository.get_candidate_descriptions(
        "geonames", cast(Any, SimpleNamespace(loc_id="existing:1")), "Paris", ""
    )

    assert existing_appended is True
    assert [entry["loc_id"] for entry in descriptions] == [
        "candidate:1",
        "existing:1",
    ]
    assert descriptions[0]["attributes"] == candidate.data
    assert descriptions[1]["latitude"] == 48.0


def test_candidate_descriptions_use_query_and_skip_existing_lookup(monkeypatch):
    """An existing candidate in search results is not redundantly fetched."""
    from geoparser.annotator.db.crud import toponym as toponym_module

    candidate = SimpleNamespace(
        identifier="place:1", data={"name": "Paris"}, geometry=None
    )
    searches = []
    gazetteer = SimpleNamespace(
        search=lambda text, method: searches.append((text, method)) or [candidate],
        find=lambda loc_id: pytest.fail(
            "candidate should already be in search results"
        ),
    )
    monkeypatch.setattr(toponym_module, "get_gazetteer", lambda name: gazetteer)
    monkeypatch.setattr(
        ToponymRepository,
        "_get_wgs84_coordinates",
        classmethod(lambda cls, feature: (None, None)),
    )

    descriptions, existing_appended = ToponymRepository.get_candidate_descriptions(
        "geonames",
        cast(Any, SimpleNamespace(loc_id="place:1")),
        "Paris",
        "Paris, Texas",
    )

    assert searches == [("Paris, Texas", "exact")]
    assert len(descriptions) == 1
    assert existing_appended is False


def test_candidate_descriptions_do_not_claim_unresolved_existing_location(monkeypatch):
    """A stale location ID is not reported as a candidate that was appended."""
    candidate = SimpleNamespace(
        identifier="candidate:1", data={"name": "Paris"}, geometry=None
    )
    gazetteer = SimpleNamespace(
        search=lambda text, method: [candidate], find=lambda loc_id: None
    )
    monkeypatch.setattr(toponym_crud, "get_gazetteer", lambda name: gazetteer)
    monkeypatch.setattr(
        ToponymRepository,
        "_get_wgs84_coordinates",
        classmethod(lambda cls, feature: (None, None)),
    )

    descriptions, existing_appended = ToponymRepository.get_candidate_descriptions(
        "geonames", cast(Any, SimpleNamespace(loc_id="missing:1")), "Paris", ""
    )

    assert [entry["loc_id"] for entry in descriptions] == ["candidate:1"]
    assert existing_appended is False


@pytest.mark.parametrize(
    ("existing_appended", "expected_existing"),
    [(True, {"loc_id": "place:1"}), (False, None)],
)
def test_get_candidates_returns_session_candidate_payload(
    db_session, monkeypatch, existing_appended, expected_existing
):
    """Candidate lookup combines the document toponym and gazetteer filters."""
    session = SessionRepository.create(
        db_session,
        AnnotatorSessionCreate(
            gazetteer="geonames",
            documents=[
                AnnotatorDocumentCreate(
                    filename="places.txt",
                    spacy_model="en_core_web_sm",
                    text="Paris",
                    toponyms=[AnnotatorToponymCreate(text="Paris", start=0, end=5)],
                )
            ],
        ),
    )
    monkeypatch.setattr(
        ToponymRepository,
        "get_candidate_descriptions",
        classmethod(
            lambda cls, gazetteer, toponym, text, query_text: (
                [{"loc_id": "place:1"}],
                existing_appended,
            )
        ),
    )

    payload = ToponymRepository.get_candidates(
        session.documents[0],
        "geonames",
        CandidatesGet(start=0, end=5, text="Paris"),
    )

    assert payload == {
        "candidates": [{"loc_id": "place:1"}],
        "filter_attributes": [
            "feature_name",
            "country_name",
            "admin1_name",
            "admin2_name",
        ],
        "existing_loc_id": "",
        "existing_candidate": expected_existing,
    }


def test_get_candidates_raises_when_span_is_not_a_toponym(db_session):
    """A candidate request for an unknown span uses the repository exception."""
    session = SessionRepository.create(
        db_session,
        AnnotatorSessionCreate(gazetteer="geonames"),
    )
    document = DocumentRepository.create(
        db_session,
        AnnotatorDocumentCreate(
            filename="places.txt", spacy_model="en_core_web_sm", text="Paris"
        ),
        additional={"session_id": session.id},
    )

    with pytest.raises(ToponymNotFoundException):
        ToponymRepository.get_candidates(
            document, "geonames", CandidatesGet(start=0, end=5)
        )


def test_annotate_many_propagates_one_sense_to_matching_toponyms(db_session):
    """One-sense-per-discourse applies an annotation to repeated mentions only."""
    session = SessionRepository.create(
        db_session,
        AnnotatorSessionCreate(
            gazetteer="geonames",
            settings=AnnotatorSessionSettingsCreate(one_sense_per_discourse=True),
            documents=[
                AnnotatorDocumentCreate(
                    filename="places.txt",
                    spacy_model="en_core_web_sm",
                    text="Paris, Paris",
                    toponyms=[
                        AnnotatorToponymCreate(text="Paris", start=0, end=5),
                        AnnotatorToponymCreate(text="Paris", start=7, end=12),
                    ],
                )
            ],
        ),
    )
    document = session.documents[0]

    annotated = ToponymRepository.annotate_many(
        db_session,
        document,
        AnnotatorToponymCreate(text="Paris", start=0, end=5, loc_id="place:1"),
    )

    assert [toponym.loc_id for toponym in annotated] == ["place:1", "place:1"]


def test_annotate_many_raises_when_span_is_not_in_document(db_session):
    """Annotating a span absent from the document raises the CRUD error."""
    session = SessionRepository.create(
        db_session,
        AnnotatorSessionCreate(
            gazetteer="geonames",
            documents=[
                AnnotatorDocumentCreate(
                    filename="places.txt",
                    spacy_model="en_core_web_sm",
                    text="Paris",
                )
            ],
        ),
    )

    with pytest.raises(ToponymNotFoundException):
        ToponymRepository.annotate_many(
            db_session,
            session.documents[0],
            AnnotatorToponymCreate(text="Paris", start=0, end=5, loc_id="place:1"),
        )


def test_annotate_many_does_not_propagate_an_empty_location_id(db_session):
    """Clearing one annotation does not clear matching rows in one-sense mode."""
    session = SessionRepository.create(
        db_session,
        AnnotatorSessionCreate(
            gazetteer="geonames",
            settings=AnnotatorSessionSettingsCreate(one_sense_per_discourse=True),
            documents=[
                AnnotatorDocumentCreate(
                    filename="places.txt",
                    spacy_model="en_core_web_sm",
                    text="Paris, Paris",
                    toponyms=[
                        AnnotatorToponymCreate(text="Paris", start=0, end=5),
                        AnnotatorToponymCreate(text="Paris", start=7, end=12),
                    ],
                )
            ],
        ),
    )

    annotated = ToponymRepository.annotate_many(
        db_session,
        session.documents[0],
        AnnotatorToponymCreate(text="Paris", start=0, end=5, loc_id=""),
    )

    assert [toponym.loc_id for toponym in annotated] == ["", ""]


def test_validate_overlap_skips_incomplete_span_and_accepts_disjoint_span(db_session):
    """Partial updates skip overlap checks and separate spans remain valid."""
    session = SessionRepository.create(
        db_session,
        AnnotatorSessionCreate(
            gazetteer="geonames",
            documents=[
                AnnotatorDocumentCreate(
                    filename="places.txt",
                    spacy_model="en_core_web_sm",
                    text="Paris Berlin",
                    toponyms=[AnnotatorToponymCreate(text="Paris", start=0, end=5)],
                )
            ],
        ),
    )
    document = session.documents[0]

    assert ToponymRepository.validate_overlap(
        db_session, AnnotatorToponymUpdate(id=uuid.uuid4()), None
    )
    assert ToponymRepository.validate_overlap(
        db_session,
        AnnotatorToponymUpdate(id=uuid.uuid4(), start=None, end=5),
        document.id,
    )
    assert ToponymRepository.validate_overlap(
        db_session,
        AnnotatorToponymUpdate(id=uuid.uuid4(), start=0, end=None),
        document.id,
    )
    assert ToponymRepository.validate_overlap(
        db_session,
        AnnotatorToponymCreate(text="Berlin", start=6, end=12),
        document.id,
    )


def test_remove_duplicates_drops_existing_spans_and_sorts_new_spans():
    """Recognizer output is deduplicated against old rows and returned by span."""
    old = [AnnotatorToponymCreate(text="Paris", start=0, end=5)]
    new = [
        AnnotatorToponymCreate(text="Berlin", start=7, end=13),
        AnnotatorToponymCreate(text="Paris", start=0, end=5),
        AnnotatorToponymCreate(text="Lyon", start=15, end=19),
    ]

    unique = ToponymRepository._remove_duplicates(old, new)

    assert [(toponym.start, toponym.text) for toponym in unique] == [
        (7, "Berlin"),
        (15, "Lyon"),
    ]


def test_document_parse_deduplicates_recognized_spans(db_session, monkeypatch):
    """Parsing adds only new spans and marks the document as parsed."""
    session = SessionRepository.create(
        db_session,
        AnnotatorSessionCreate(
            gazetteer="geonames",
            documents=[
                AnnotatorDocumentCreate(
                    filename="places.txt",
                    spacy_model="en_core_web_sm",
                    text="Paris, Berlin",
                    toponyms=[AnnotatorToponymCreate(text="Paris", start=0, end=5)],
                )
            ],
        ),
    )

    class Recognizer:
        def __init__(self, model_name):
            self.model_name = model_name

        def predict(self, texts):
            assert texts == ["Paris, Berlin"]
            return [[(0, 5), (7, 13)]]

    monkeypatch.setattr(document_crud, "SpacyRecognizer", Recognizer)

    parsed = DocumentRepository.parse(db_session, session.documents[0].id)

    assert parsed.spacy_applied is True
    assert [(toponym.start, toponym.text) for toponym in parsed.toponyms] == [
        (0, "Paris"),
        (7, "Berlin"),
    ]


def test_pre_annotated_text_escapes_text_and_marks_only_resolved_spans(db_session):
    """Annotation rendering escapes user text and preserves span state."""
    session = SessionRepository.create(
        db_session,
        AnnotatorSessionCreate(
            gazetteer="geonames",
            documents=[
                AnnotatorDocumentCreate(
                    filename="places.txt",
                    spacy_model="en_core_web_sm",
                    text="Paris & Lyon",
                    toponyms=[
                        AnnotatorToponymCreate(
                            text="Paris", start=0, end=5, loc_id="place:1"
                        ),
                        AnnotatorToponymCreate(text="Lyon", start=8, end=12),
                    ],
                )
            ],
        ),
    )

    rendered = DocumentRepository.get_pre_annotated_text(
        db_session, session.documents[0].id
    )

    assert (
        'class="toponym annotated" data-start="0" data-end="5">Paris</span>' in rendered
    )
    assert 'class="toponym " data-start="8" data-end="12">Lyon</span>' in rendered
    assert "&amp;" in rendered


def test_create_from_json_preserves_session_id_and_nested_records(db_session):
    """Valid JSON import restores child rows and its ID when requested."""
    session_id = uuid.uuid4()
    content = {
        "session_id": str(session_id),
        "gazetteer": "geonames",
        "documents": [
            {
                "filename": "places.txt",
                "spacy_model": "en_core_web_sm",
                "text": "Paris",
                "toponyms": [
                    {"text": "Paris", "start": 0, "end": 5, "loc_id": "place:1"}
                ],
            }
        ],
    }

    imported = SessionRepository.create_from_json(
        db_session, json.dumps(content), keep_id=True
    )

    assert imported.id == session_id
    assert imported.documents[0].spacy_applied is True
    assert imported.documents[0].toponyms[0].loc_id == "place:1"


def test_invalid_json_session_import_creates_no_partial_rows(db_session):
    """Malformed nested imports fail before persisting a session or documents."""
    content = {
        "gazetteer": "geonames",
        "documents": [{"filename": "incomplete.txt", "text": "Paris"}],
    }

    with pytest.raises(InvalidUploadException, match="Invalid session JSON"):
        SessionRepository.create_from_json(db_session, json.dumps(content))

    assert SessionRepository.read_all(db_session) == []


@pytest.mark.parametrize("json_str", ["[]", '{"gazetteer": "geonames"}'])
def test_session_import_rejects_non_object_or_missing_documents(db_session, json_str):
    """Malformed top-level shapes fail through the same upload error contract."""
    with pytest.raises(InvalidUploadException, match="Invalid session JSON"):
        SessionRepository.create_from_json(db_session, json_str)

    assert SessionRepository.read_all(db_session) == []


@pytest.mark.parametrize(
    ("create", "item", "message"),
    [
        (
            DocumentRepository.create,
            AnnotatorDocumentCreate(filename="a.txt", spacy_model="m", text="Paris"),
            "document cannot be created without link to session",
        ),
        (
            SessionSettingsRepository.create,
            AnnotatorSessionSettingsCreate(),
            "settings cannot be created without link to session",
        ),
        (
            ToponymRepository.create,
            AnnotatorToponymCreate(text="Paris", start=0, end=5),
            "toponym cannot be created without link to document",
        ),
    ],
)
@pytest.mark.parametrize("additional", [None, {"unrelated": 1}])
def test_child_rows_require_their_parent_link(
    db_session, create, item, message, additional
):
    """Documents, settings and toponyms cannot be orphaned."""
    with pytest.raises(ValueError, match=message):
        create(db_session, item, additional=additional)
