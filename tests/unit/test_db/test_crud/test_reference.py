"""
Unit tests for geoparser/db/crud/reference.py

Tests the ReferenceRepository class with custom query methods.
"""

import pytest
from sqlmodel import Session

from geoparser.db.crud import ReferenceRepository
from geoparser.db.models import (
    Document,
    DocumentCreate,
    Project,
    ProjectCreate,
    Recognizer,
    RecognizerCreate,
    ReferenceCreate,
    ReferenceUpdate,
)


@pytest.mark.unit
class TestReferenceRepositoryUpdate:
    """Test the update method of ReferenceRepository."""

    def test_updates_reference_and_refreshes_text(
        self,
        test_session: Session,
        reference_factory,
        document_factory,
    ):
        """Test that update refreshes the text field when positions change."""
        # Arrange
        document = document_factory(text="Hello World, this is a test.")
        reference = reference_factory(
            start=0, end=5, document_id=document.id
        )  # "Hello"

        # Verify initial state
        test_session.refresh(reference)
        assert reference.text == "Hello"

        # Act - Update to point to "World"
        update = ReferenceUpdate(id=reference.id, start=6, end=11)
        updated_ref = ReferenceRepository.update(
            test_session, db_obj=reference, obj_in=update
        )

        # Assert - Text should be updated to "World"
        test_session.refresh(updated_ref)
        assert updated_ref.text == "World"
        assert updated_ref.start == 6
        assert updated_ref.end == 11

    def test_updates_text_when_document_changes(
        self,
        test_session: Session,
        reference_factory,
        document_factory,
    ):
        """Test that update refreshes text when document_id changes."""
        # Arrange
        doc1 = document_factory(text="First document text")
        doc2 = document_factory(text="Second document text")
        reference = reference_factory(start=0, end=5, document_id=doc1.id)  # "First"

        test_session.refresh(reference)
        assert reference.text == "First"

        # Act - Update to point to second document
        update = ReferenceUpdate(id=reference.id, document_id=doc2.id, start=0, end=6)
        updated_ref = ReferenceRepository.update(
            test_session, db_obj=reference, obj_in=update
        )

        # Assert - Text should be from second document
        test_session.refresh(updated_ref)
        assert updated_ref.text == "Second"
        assert updated_ref.document_id == doc2.id


@pytest.mark.unit
class TestReferenceRepositoryGetByDocument:
    """Test the get_by_document method of ReferenceRepository."""

    def test_returns_references_for_document(
        self,
        test_session: Session,
        document_factory,
        reference_factory,
    ):
        """Test that get_by_document returns all references for a document."""
        # Arrange
        document = document_factory(text="New York and Paris are cities.")
        ref1 = reference_factory(start=0, end=8, document_id=document.id)  # "New York"
        ref2 = reference_factory(start=13, end=18, document_id=document.id)  # "Paris"

        # Act
        references = ReferenceRepository.get_by_document(test_session, document.id)

        # Assert
        assert len(references) == 2
        reference_ids = [r.id for r in references]
        assert ref1.id in reference_ids
        assert ref2.id in reference_ids

    def test_returns_empty_list_for_document_without_references(
        self, test_session: Session, document_factory
    ):
        """Test that get_by_document returns empty list for document without references."""
        # Arrange
        document = document_factory()

        # Act
        references = ReferenceRepository.get_by_document(test_session, document.id)

        # Assert
        assert references == []


@pytest.mark.unit
class TestReferenceRepositoryGetByDocumentAndSpan:
    """Test the get_by_document_and_span method of ReferenceRepository."""

    def test_returns_reference_for_matching_span(
        self,
        test_session: Session,
        document_factory,
        reference_factory,
    ):
        """Test that get_by_document_and_span returns reference for matching document and span."""
        # Arrange
        document = document_factory(text="New York is a city")
        reference = reference_factory(start=0, end=8, document_id=document.id)

        # Act
        found_ref = ReferenceRepository.get_by_document_and_span(
            test_session, document.id, 0, 8
        )

        # Assert
        assert found_ref is not None
        assert found_ref.id == reference.id
        assert found_ref.start == 0
        assert found_ref.end == 8

    def test_returns_none_for_non_matching_span(
        self,
        test_session: Session,
        document_factory,
        reference_factory,
    ):
        """Test that get_by_document_and_span returns None for non-matching span."""
        # Arrange
        document = document_factory(text="New York is a city")
        reference_factory(start=0, end=8, document_id=document.id)

        # Act - Query with different span
        found_ref = ReferenceRepository.get_by_document_and_span(
            test_session, document.id, 9, 11
        )

        # Assert
        assert found_ref is None

    @pytest.mark.parametrize(("span", "reference_index"), [((0, 8), 0), ((13, 18), 1)])
    def test_span_lookup_returns_the_matching_reference(
        self, test_session, document_factory, reference_factory, span, reference_index
    ):
        """Each distinct span in one document selects its own reference."""
        # Arrange
        document = document_factory(text="New York and Paris are cities")
        ref1 = reference_factory(start=0, end=8, document_id=document.id)
        ref2 = reference_factory(start=13, end=18, document_id=document.id)

        # Act
        found = ReferenceRepository.get_by_document_and_span(
            test_session, document.id, *span
        )

        # Assert
        assert found is not None
        assert found.id == (ref1, ref2)[reference_index].id


@pytest.fixture
def reference(session):
    """A reference over "Paris and Berlin", covering "Paris"."""
    project = Project(**ProjectCreate(name="demo").model_dump())
    session.add(project)
    session.commit()
    session.refresh(project)

    document = Document(
        **DocumentCreate(text="Paris and Berlin", project_id=project.id).model_dump()
    )
    session.add(document)
    session.commit()
    session.refresh(document)

    recognizer = Recognizer(
        **RecognizerCreate(id="rec-1", name="TestRecognizer", config={}).model_dump()
    )
    session.add(recognizer)
    session.commit()

    return ReferenceRepository.create(
        session,
        ReferenceCreate(
            start=0, end=5, document_id=document.id, recognizer_id=recognizer.id
        ),
    )


@pytest.mark.unit
class TestReferenceText:
    """The cached slice of document text."""

    def test_create_stores_the_span_text(self, reference):
        assert reference.text == "Paris"

    def test_moving_only_the_end_keeps_the_existing_start(self, session, reference):
        updated = ReferenceRepository.update(
            session,
            db_obj=reference,
            obj_in=ReferenceUpdate(id=reference.id, end=16),
        )

        assert (updated.start, updated.end) == (0, 16)
        assert updated.text == "Paris and Berlin"

    def test_moving_only_the_start_keeps_the_existing_end(self, session, reference):
        updated = ReferenceRepository.update(
            session,
            db_obj=reference,
            obj_in=ReferenceUpdate(id=reference.id, start=1),
        )

        assert (updated.start, updated.end) == (1, 5)
        assert updated.text == "aris"

    def test_moving_both_ends_re_slices_the_text(self, session, reference):
        updated = ReferenceRepository.update(
            session,
            db_obj=reference,
            obj_in=ReferenceUpdate(id=reference.id, start=10, end=16),
        )

        assert updated.text == "Berlin"


@pytest.mark.unit
class TestAdditionalReferenceSpanLookups:
    """References are identified by both their document and span."""

    def test_finds_the_reference_at_that_span(self, session, reference):
        found = ReferenceRepository.get_by_document_and_span(
            session, reference.document_id, 0, 5
        )

        assert found is not None
        assert found.id == reference.id

    @pytest.mark.parametrize(("start", "end"), [(0, 4), (1, 5), (10, 16)])
    def test_returns_nothing_for_a_different_span(self, session, reference, start, end):
        assert (
            ReferenceRepository.get_by_document_and_span(
                session, reference.document_id, start, end
            )
            is None
        )

    def test_create_slices_from_the_start_offset(self, session, reference):
        created = ReferenceRepository.create(
            session,
            ReferenceCreate(
                start=10,
                end=16,
                document_id=reference.document_id,
                recognizer_id=reference.recognizer_id,
            ),
        )

        assert created.text == "Berlin"

    def test_moving_only_the_end_keeps_a_non_zero_start(self, session, reference):
        middle = ReferenceRepository.create(
            session,
            ReferenceCreate(
                start=6,
                end=9,
                document_id=reference.document_id,
                recognizer_id=reference.recognizer_id,
            ),
        )
        updated = ReferenceRepository.update(
            session,
            db_obj=middle,
            obj_in=ReferenceUpdate(id=middle.id, end=16),
        )

        assert (updated.start, updated.end) == (6, 16)
        assert updated.text == "and Berlin"

    def test_returns_only_the_references_of_that_document(self, session, reference):
        original = session.get(Document, reference.document_id)
        other = Document(
            **DocumentCreate(
                text="Rome and Milan", project_id=original.project_id
            ).model_dump()
        )
        session.add(other)
        session.commit()
        session.refresh(other)
        ReferenceRepository.create(
            session,
            ReferenceCreate(
                start=0,
                end=4,
                document_id=other.id,
                recognizer_id=reference.recognizer_id,
            ),
        )

        found = ReferenceRepository.get_by_document(session, reference.document_id)

        assert [item.id for item in found] == [reference.id]

    def test_the_same_span_in_another_document_is_not_returned(
        self, session, reference
    ):
        original = session.get(Document, reference.document_id)
        other = Document(
            **DocumentCreate(
                text="Paris and Berlin", project_id=original.project_id
            ).model_dump()
        )
        session.add(other)
        session.commit()
        session.refresh(other)
        twin = ReferenceRepository.create(
            session,
            ReferenceCreate(
                start=0,
                end=5,
                document_id=other.id,
                recognizer_id=reference.recognizer_id,
            ),
        )

        found = ReferenceRepository.get_by_document_and_span(session, other.id, 0, 5)

        assert found is not None
        assert found.id == twin.id
        assert found.id != reference.id
