"""Exact contracts of the annotator repositories, pinned for mutation testing."""

import uuid
from io import BytesIO
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest
from fastapi import UploadFile
from shapely.geometry import Point
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from geoparser.annotator.constants import DEFAULT_SESSION_SETTINGS
from geoparser.annotator.db.crud import (
    DocumentRepository,
    SessionRepository,
    SessionSettingsRepository,
    ToponymRepository,
)
from geoparser.annotator.db.crud import document as document_crud
from geoparser.annotator.db.crud import toponym as toponym_crud
from geoparser.annotator.db.crud.base import BaseRepository
from geoparser.annotator.db.models.document import AnnotatorDocumentCreate
from geoparser.annotator.db.models.session import AnnotatorSessionCreate
from geoparser.annotator.db.models.settings import AnnotatorSessionSettingsCreate
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
def db():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


def _session(db, texts=("Paris",), toponyms=None):
    """A session with one document per text, each with optional toponyms."""
    toponyms = toponyms or {}
    return SessionRepository.create(
        db,
        AnnotatorSessionCreate(
            gazetteer="geonames",
            documents=[
                AnnotatorDocumentCreate(
                    filename=f"{index}.txt",
                    spacy_model="en_core_web_sm",
                    text=text,
                    toponyms=toponyms.get(index, []),
                )
                for index, text in enumerate(texts)
            ],
        ),
    )


def _upload(content: bytes, filename: str | None = "a.txt") -> UploadFile:
    return UploadFile(file=BytesIO(content), filename=filename)


# --- BaseRepository ------------------------------------------------------


class _Thing(SQLModel):
    name: str = "table default"


class _ThingCreate(SQLModel):
    name: str = "create default"


class _ThingRepository(BaseRepository[_Thing]):
    model = _Thing


@pytest.mark.unit
def test_mapping_drops_fields_the_caller_did_not_set():
    assert _ThingRepository.get_mapped_class(_ThingCreate()).name == "table default"


@pytest.mark.unit
def test_missing_row_names_the_model_and_id(db):
    missing = uuid.uuid4()

    with pytest.raises(DocumentNotFoundException) as error:
        DocumentRepository.read(db, missing)

    assert str(error.value) == f"AnnotatorDocument with ID {missing} not found."


# --- DocumentRepository --------------------------------------------------


@pytest.mark.unit
def test_reading_an_upload_does_not_rewind_by_default():
    upload = _upload(b"Paris")

    assert DocumentRepository._read_uploaded_text(upload) == "Paris"
    assert upload.file.read() == b""


@pytest.mark.unit
def test_an_unnamed_undecodable_upload_is_named_generically():
    with pytest.raises(InvalidUploadException) as error:
        DocumentRepository._read_uploaded_text(_upload(b"\xff", filename=None))

    assert str(error.value) == "Text file 'uploaded_file' must be valid UTF-8."


@pytest.mark.unit
def test_an_unnamed_upload_decodes_with_an_empty_filename():
    upload = _upload(b"Paris", filename=None)

    assert DocumentRepository._decode_text_files([upload]) == [("", "Paris")]


@pytest.mark.unit
def test_toponyms_come_from_the_recognizer_for_that_text():
    recognizer = Mock()
    recognizer.predict.return_value = [[(0, 5)]]

    toponyms = DocumentRepository._extract_toponyms("Paris is big", recognizer)

    recognizer.predict.assert_called_once_with(["Paris is big"])
    assert [(t.text, t.start, t.end) for t in toponyms] == [("Paris", 0, 5)]


@pytest.mark.unit
def test_highest_index_is_the_largest_doc_index(db):
    session = _session(db, texts=("a", "b", "c"))

    assert DocumentRepository.get_highest_index(db, session.id) == 2


@pytest.mark.unit
def test_reindexing_keeps_the_existing_document_order(db):
    session = _session(db, texts=("a", "b", "c"))
    first, second, third = session.documents
    first.doc_index = 5
    db.add(first)
    db.commit()

    DocumentRepository._reindex_documents(db, session.id)

    db.refresh(first)
    db.refresh(second)
    db.refresh(third)
    assert (second.doc_index, third.doc_index, first.doc_index) == (0, 1, 2)


@pytest.mark.unit
def test_text_uploads_skip_spacy_unless_asked(db, monkeypatch):
    session = _session(db, texts=())
    monkeypatch.setattr(
        document_crud,
        "SpacyRecognizer",
        Mock(side_effect=AssertionError("spaCy must not load")),
    )

    (document,) = DocumentRepository.create_from_text_files(
        db, [_upload(b"Paris")], session.id, "en_core_web_sm"
    )

    assert document.spacy_applied is False


@pytest.mark.unit
def test_text_uploads_load_the_requested_spacy_model(db, monkeypatch):
    session = _session(db, texts=())
    recognizer = Mock()
    recognizer.predict.return_value = [[]]
    factory = Mock(return_value=recognizer)
    monkeypatch.setattr(document_crud, "SpacyRecognizer", factory)

    DocumentRepository.create_from_text_files(
        db, [_upload(b"Paris")], session.id, "xx_model", apply_spacy=True
    )

    factory.assert_called_once_with(model_name="xx_model")


@pytest.mark.unit
def test_pre_annotated_text_wraps_each_toponym_in_a_span(db):
    session = _session(
        db,
        texts=("Paris & Bern",),
        toponyms={
            0: [
                AnnotatorToponymCreate(text="Paris", start=0, end=5, loc_id="1"),
                AnnotatorToponymCreate(text="Bern", start=8, end=12),
            ]
        },
    )

    html = DocumentRepository.get_pre_annotated_text(db, session.documents[0].id)

    assert str(html) == (
        '<span class="toponym annotated" data-start="0" data-end="5">Paris</span>'
        " &amp; "
        '<span class="toponym " data-start="8" data-end="12">Bern</span>'
    )


@pytest.mark.unit
def test_progress_counts_annotated_toponyms_per_document(db):
    session = _session(
        db,
        texts=("Paris Bern Rome", "Oslo", "none"),
        toponyms={
            0: [
                AnnotatorToponymCreate(text="Paris", start=0, end=5, loc_id="1"),
                AnnotatorToponymCreate(text="Bern", start=6, end=10, loc_id="2"),
                AnnotatorToponymCreate(text="Rome", start=11, end=15),
            ],
            1: [AnnotatorToponymCreate(text="Oslo", start=0, end=4, loc_id="3")],
        },
    )
    _session(db, texts=("other session",))

    progress = list(DocumentRepository.get_progress(db, session_id=session.id))

    assert [
        (p["doc_index"], p["annotated_toponyms"], p["total_toponyms"]) for p in progress
    ] == [(0, 2, 3), (1, 1, 1), (2, 0, 0)]
    assert progress[0]["progress_percentage"] == pytest.approx(200 / 3)
    assert progress[1]["progress_percentage"] == 100
    assert progress[2]["progress_percentage"] == 0


@pytest.mark.unit
def test_document_progress_is_for_that_document_only(db):
    session = _session(db, texts=("a", "b"))
    second = session.documents[1]

    assert DocumentRepository.get_document_progress(db, second.id)["doc_index"] == 1


@pytest.mark.unit
def test_progress_of_a_missing_document_names_it(db):
    missing = uuid.uuid4()

    with pytest.raises(DocumentNotFoundException) as error:
        DocumentRepository.get_document_progress(db, missing)

    assert str(error.value) == f"AnnotatorDocument with ID {missing} not found."


@pytest.mark.unit
def test_parsing_uses_the_documents_spacy_model(db, monkeypatch):
    session = _session(db)
    recognizer = Mock()
    recognizer.predict.return_value = [[]]
    factory = Mock(return_value=recognizer)
    monkeypatch.setattr(document_crud, "SpacyRecognizer", factory)

    DocumentRepository.parse(db, session.documents[0].id)

    factory.assert_called_once_with(model_name="en_core_web_sm")


@pytest.mark.unit
@pytest.mark.parametrize(
    "repository",
    [DocumentRepository, SessionRepository, SessionSettingsRepository],
)
def test_read_all_applies_its_filters(db, repository):
    kept = _session(db)
    _session(db)
    filters = {
        DocumentRepository: {"session_id": kept.id},
        SessionRepository: {"id": kept.id},
        SessionSettingsRepository: {"session_id": kept.id},
    }[repository]

    assert len(repository.read_all(db, **filters)) == 1


@pytest.mark.unit
def test_toponym_read_all_applies_its_filters(db):
    session = _session(
        db,
        texts=("Paris", "Bern"),
        toponyms={
            0: [AnnotatorToponymCreate(text="Paris", start=0, end=5)],
            1: [AnnotatorToponymCreate(text="Bern", start=0, end=4)],
        },
    )

    found = ToponymRepository.read_all(db, document_id=session.documents[0].id)

    assert [toponym.text for toponym in found] == ["Paris"]


# --- Guards and exact messages ------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("create", "item", "message"),
    [
        (
            DocumentRepository.create,
            AnnotatorDocumentCreate(filename="a.txt", spacy_model="m", text="x"),
            "document cannot be created without link to session",
        ),
        (
            SessionSettingsRepository.create,
            AnnotatorSessionSettingsCreate(),
            "settings cannot be created without link to session",
        ),
        (
            ToponymRepository.create,
            AnnotatorToponymCreate(text="x", start=0, end=1),
            "toponym cannot be created without link to document",
        ),
    ],
)
@pytest.mark.parametrize("additional", [{}, {"unrelated": 1}])
def test_orphan_rows_are_rejected_with_an_exact_message(
    db, create, item, message, additional
):
    with pytest.raises(ValueError) as error:
        create(db, item, additional=additional)

    assert str(error.value) == message


@pytest.mark.unit
def test_settings_creation_honours_exclude(db):
    session = _session(db)
    default = DEFAULT_SESSION_SETTINGS["one_sense_per_discourse"]

    settings = SessionSettingsRepository.create(
        db,
        AnnotatorSessionSettingsCreate(one_sense_per_discourse=not default),
        exclude=["one_sense_per_discourse"],
        additional={"session_id": session.id},
    )

    assert settings.one_sense_per_discourse == default


@pytest.mark.unit
def test_toponym_creation_honours_exclude(db):
    session = _session(db)

    toponym = ToponymRepository.create(
        db,
        AnnotatorToponymCreate(text="Paris", start=0, end=5, loc_id="1"),
        exclude=["loc_id"],
        additional={"document_id": session.documents[0].id},
    )

    assert toponym.loc_id == ""


@pytest.mark.unit
def test_session_json_keeps_a_fresh_id_by_default(db):
    given = uuid.uuid4()
    content = f'{{"session_id": "{given}", "gazetteer": "geonames", "documents": []}}'

    assert SessionRepository.create_from_json(db, content).id != given


@pytest.mark.unit
def test_malformed_session_json_has_an_exact_message(db):
    with pytest.raises(InvalidUploadException) as error:
        SessionRepository.create_from_json(db, "[]")

    assert str(error.value) == (
        "Invalid session JSON: required fields are missing or malformed."
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("[]", "session JSON must contain an object"),
        ('{"documents": {}}', "session JSON must contain a documents list"),
    ],
)
def test_session_json_shape_errors_are_exact(content, message):
    with pytest.raises(TypeError) as error:
        SessionRepository._parse_json_content(content)

    assert str(error.value) == message


# --- ToponymRepository ---------------------------------------------------


def _document_with(db, *spans):
    session = _session(
        db,
        texts=("Paris Bern Rome",),
        toponyms={
            0: [
                AnnotatorToponymCreate(text=text, start=start, end=end)
                for text, start, end in spans
            ]
        },
    )
    return session.documents[0]


@pytest.mark.unit
def test_an_overlapping_toponym_is_rejected_with_both_spans(db):
    document = _document_with(db, ("Paris", 0, 5))

    with pytest.raises(ToponymOverlapException) as error:
        ToponymRepository.create(
            db,
            AnnotatorToponymCreate(text="aris", start=1, end=5),
            additional={"document_id": document.id},
        )

    assert str(error.value).startswith("Toponyms overlap: [")
    assert "text='Paris'" in str(error.value)
    assert "start=1" in str(error.value)


@pytest.mark.unit
@pytest.mark.parametrize(("start", "end"), [(5, 6), (6, 10)])
def test_adjacent_toponyms_do_not_overlap(db, start, end):
    document = _document_with(db, ("Paris", 0, 5), ("Bern", 10, 14))

    ToponymRepository.create(
        db,
        AnnotatorToponymCreate(text="x", start=start, end=end),
        additional={"document_id": document.id},
    )


@pytest.mark.unit
def test_an_updated_toponym_does_not_overlap_itself(db):
    document = _document_with(db, ("Paris", 0, 5))
    (toponym,) = document.toponyms

    ToponymRepository.update(
        db,
        AnnotatorToponymUpdate(id=toponym.id, start=0, end=4, text="Pari"),
        document_id=document.id,
    )


@pytest.mark.unit
def test_an_update_falls_back_to_the_rows_own_document(db):
    document = _document_with(db, ("Paris", 0, 5), ("Bern", 6, 10))
    _paris, bern = document.toponyms
    bern.start = 3

    with pytest.raises(ToponymOverlapException):
        ToponymRepository.update(db, bern)


@pytest.mark.unit
def test_lookup_needs_both_offsets_to_match(db):
    document = _document_with(db, ("Paris", 0, 5))

    assert ToponymRepository.get_toponym(document, 0, 4) is None
    assert ToponymRepository.get_toponym(document, 1, 5) is None


@pytest.mark.unit
@pytest.mark.parametrize(
    ("text", "loc_id", "same", "expected"),
    [
        ("Paris", "", False, True),
        ("Bern", "", False, False),
        ("Paris", "1", False, False),
        ("Paris", "", True, False),
    ],
)
def test_unannotated_repeat_needs_same_text_no_location_and_another_row(
    text, loc_id, same, expected
):
    selected = SimpleNamespace(text="Paris", loc_id="9")
    candidate = selected if same else SimpleNamespace(text=text, loc_id=loc_id)
    if same:
        selected.loc_id = loc_id

    assert (
        ToponymRepository._is_unannotated_repeat(
            cast(Any, candidate), cast(Any, selected)
        )
        is expected
    )


@pytest.mark.unit
def test_projected_coordinates_are_returned_as_wgs84_latitude_longitude():
    feature = SimpleNamespace(geometry=Point(2600000, 1200000), crs="EPSG:2056")

    latitude, longitude = ToponymRepository._get_wgs84_coordinates(cast(Any, feature))

    assert latitude == pytest.approx(46.95, abs=0.01)
    assert longitude == pytest.approx(7.44, abs=0.01)


def _feature(identifier="123", name="Paris"):
    return SimpleNamespace(
        identifier=identifier,
        geometry=Point(2.35, 48.85),
        crs="EPSG:4326",
        data={"name": name, "feature_name": "city", "country_name": "France"},
    )


@pytest.mark.unit
def test_candidate_entries_carry_the_ui_payload(monkeypatch):
    described = []
    monkeypatch.setattr(
        ToponymRepository,
        "_generate_location_description",
        classmethod(lambda cls, f, name: described.append(name) or f"about {name}"),
    )

    entry = ToponymRepository._candidate_entry(cast(Any, _feature()), "geonames")

    assert entry == {
        "loc_id": "123",
        "description": "about geonames",
        "attributes": _feature().data,
        "latitude": pytest.approx(48.85),
        "longitude": pytest.approx(2.35),
    }
    assert described == ["geonames"]


@pytest.mark.unit
def test_candidate_descriptions_search_and_describe_with_the_gazetteer_name(
    monkeypatch,
):
    gazetteer = Mock()
    gazetteer.search.return_value = [_feature("1")]
    gazetteer.find.return_value = _feature("2")
    opened = Mock(return_value=gazetteer)
    monkeypatch.setattr(toponym_crud, "get_gazetteer", opened)
    names = []
    monkeypatch.setattr(
        ToponymRepository,
        "_candidate_entry",
        classmethod(lambda cls, f, name: names.append(name) or f.identifier),
    )
    toponym = SimpleNamespace(loc_id="2")

    entries, appended = ToponymRepository.get_candidate_descriptions(
        "geonames", cast(Any, toponym), "Paris", ""
    )

    opened.assert_called_once_with("geonames")
    gazetteer.find.assert_called_once_with("2")
    assert (entries, appended) == (["1", "2"], True)
    assert names == ["geonames", "geonames"]


@pytest.mark.unit
def test_get_candidates_passes_the_request_through(db, monkeypatch):
    document = _document_with(db, ("Paris", 0, 5))
    describe = Mock(return_value=([], False))
    monkeypatch.setattr(ToponymRepository, "get_candidate_descriptions", describe)

    ToponymRepository.get_candidates(
        document,
        "geonames",
        CandidatesGet(start=0, end=5, text="Paris", query_text="Par"),
    )

    describe.assert_called_once_with("geonames", document.toponyms[0], "Paris", "Par")


@pytest.mark.unit
def test_get_candidates_rejects_an_unknown_span(db):
    document = _document_with(db, ("Paris", 0, 5))

    with pytest.raises(ToponymNotFoundException):
        ToponymRepository.get_candidates(
            document, "geonames", CandidatesGet(start=0, end=4)
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("request_fields", "expected"),
    [
        (
            {"start": None, "end": None, "text": None, "query_text": None},
            (0, 0, "", ""),
        ),
        ({"start": 3, "end": 7, "text": "a", "query_text": "b"}, (3, 7, "a", "b")),
    ],
)
def test_candidate_request_defaults(request_fields, expected):
    request = CandidatesGet(**request_fields)

    assert ToponymRepository._normalize_candidate_request(request) == expected


@pytest.mark.unit
def test_unknown_gazetteers_offer_no_filter_attributes():
    payload = ToponymRepository._candidate_payload(
        [],
        cast(Any, SimpleNamespace(loc_id="")),
        "unknown-gazetteer",
        existing_candidate_is_appended=False,
    )

    assert payload["filter_attributes"] == []


@pytest.mark.unit
def test_an_imported_document_must_list_its_toponyms(db):
    content = (
        '{"gazetteer": "geonames", "documents": '
        '[{"filename": "a.txt", "spacy_model": "m", "text": "Paris"}]}'
    )

    with pytest.raises(InvalidUploadException) as error:
        SessionRepository.create_from_json(db, content)

    assert isinstance(error.value.__cause__, TypeError)


@pytest.mark.unit
@pytest.mark.parametrize("toponyms", ["null", "{}", '"Paris"'])
def test_imported_document_rejects_non_list_toponyms(db, toponyms):
    content = (
        '{"gazetteer": "geonames", "documents": [{"filename": "a.txt", '
        '"spacy_model": "m", "text": "Paris", "toponyms": '
        f"{toponyms}" + "}] }"
    )

    with pytest.raises(InvalidUploadException) as error:
        SessionRepository.create_from_json(db, content)

    assert isinstance(error.value.__cause__, TypeError)


@pytest.mark.unit
def test_imported_toponyms_are_kept(db):
    content = (
        '{"gazetteer": "geonames", "documents": [{"filename": "a.txt", '
        '"spacy_model": "m", "text": "Paris", "spacy_applied": false, '
        '"toponyms": '
        '[{"text": "Paris", "start": 0, "end": 5, "loc_id": "1"}]}]}'
    )

    session = SessionRepository.create_from_json(db, content)

    (document,) = session.documents
    assert [(t.text, t.loc_id) for t in document.toponyms] == [("Paris", "1")]
    assert document.spacy_applied is True


@pytest.mark.unit
@pytest.mark.parametrize("entry", ["null", "[]", '"text"'])
def test_a_non_object_document_entry_is_an_invalid_upload(db, entry):
    content = f'{{"gazetteer": "geonames", "documents": [{entry}]}}'

    with pytest.raises(InvalidUploadException) as error:
        SessionRepository.create_from_json(db, content)

    assert isinstance(error.value.__cause__, TypeError)
