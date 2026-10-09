"""
Unit tests for geoparser/services/recognition.py

Tests the RecognitionService class with mocked recognizers.
"""

import uuid
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

import pytest

from geoparser.services.recognition import RecognitionService


@pytest.mark.unit
class TestRecognitionServiceInitialization:
    """Test RecognitionService initialization."""

    def test_creates_with_recognizer(self, mock_spacy_recognizer):
        """Test that RecognitionService can be created with a recognizer."""
        # Arrange & Act
        service = RecognitionService(mock_spacy_recognizer)

        # Assert
        assert service.recognizer == mock_spacy_recognizer


@pytest.mark.unit
class TestRecognitionServicePredict:
    """Test RecognitionService predict method."""

    def test_ensures_recognizer_record_exists(
        self, test_session, mock_spacy_recognizer, document_factory
    ):
        """Test that predict ensures recognizer record exists in database."""
        # Arrange
        document = document_factory(text="Test document")
        mock_spacy_recognizer.predict.return_value = [[(0, 4)]]
        service = RecognitionService(mock_spacy_recognizer)

        # Act
        service.predict([document])

        # Assert - Recognizer record should be created
        from geoparser.db.crud import RecognizerRepository

        recognizer = RecognizerRepository.get(test_session, mock_spacy_recognizer.id)
        assert recognizer is not None
        assert recognizer.id == mock_spacy_recognizer.id

    def test_calls_recognizer_predict(
        self, test_session, mock_spacy_recognizer, document_factory
    ):
        """Test that predict calls the recognizer's predict method."""
        # Arrange
        document = document_factory(text="New York is a city.")
        mock_spacy_recognizer.predict.return_value = [[(0, 8)]]
        service = RecognitionService(mock_spacy_recognizer)

        # Act
        service.predict([document])

        # Assert
        mock_spacy_recognizer.predict.assert_called_once()
        # Check that it was called with the document text
        call_args = mock_spacy_recognizer.predict.call_args[0][0]
        assert call_args == ["New York is a city."]

    def test_creates_references_in_database(
        self, test_session, mock_spacy_recognizer, document_factory
    ):
        """Test that predict creates reference records in the database."""
        # Arrange
        document = document_factory(text="Test document")
        mock_spacy_recognizer.predict.return_value = [[(0, 4), (5, 13)]]
        service = RecognitionService(mock_spacy_recognizer)

        # Act
        service.predict([document])

        # Assert - References should be created
        from sqlmodel import select

        from geoparser.db.models import Reference

        statement = select(Reference).where(Reference.document_id == document.id)
        references = test_session.exec(statement).unique().all()
        assert [(reference.start, reference.end) for reference in references] == [
            (0, 4),
            (5, 13),
        ]

    def test_creates_recognition_record(
        self, test_session, mock_spacy_recognizer, document_factory
    ):
        """Test that predict creates a recognition record marking document as processed."""
        # Arrange
        document = document_factory(text="Test")
        mock_spacy_recognizer.predict.return_value = [[(0, 4)]]
        service = RecognitionService(mock_spacy_recognizer)

        # Act
        service.predict([document])

        # Assert - Recognition record should exist
        from geoparser.db.crud import RecognitionRepository

        recognition = RecognitionRepository.get_by_document_and_recognizer(
            test_session, document.id, mock_spacy_recognizer.id
        )
        assert recognition is not None

    def test_skips_already_processed_documents(
        self, test_session, mock_spacy_recognizer, document_factory, recognizer_factory
    ):
        """Test that predict skips documents already processed by this recognizer."""
        # Arrange
        recognizer_record = recognizer_factory(
            id=mock_spacy_recognizer.id,
            name=mock_spacy_recognizer.name,
            config=mock_spacy_recognizer.config,
        )
        document = document_factory(text="Test")

        # Mark as already processed
        from geoparser.db.crud import RecognitionRepository
        from geoparser.db.models import RecognitionCreate

        RecognitionRepository.create(
            test_session,
            RecognitionCreate(
                document_id=document.id, recognizer_id=recognizer_record.id
            ),
        )

        mock_spacy_recognizer.predict.return_value = [[(0, 4)]]
        service = RecognitionService(mock_spacy_recognizer)

        # Act
        service.predict([document])

        # Assert - Predict should not be called since document was already processed
        mock_spacy_recognizer.predict.assert_not_called()

    def test_handles_none_predictions(
        self, test_session, mock_spacy_recognizer, document_factory
    ):
        """Test that predict handles None predictions (unavailable) correctly."""
        # Arrange
        document = document_factory(text="Test")
        mock_spacy_recognizer.predict.return_value = [None]  # Prediction not available
        service = RecognitionService(mock_spacy_recognizer)

        # Act
        service.predict([document])

        # Assert - No references should be created, no recognition record
        from sqlmodel import select

        from geoparser.db.crud import RecognitionRepository
        from geoparser.db.models import Reference

        statement = select(Reference).where(Reference.document_id == document.id)
        references = test_session.exec(statement).all()
        assert len(references) == 0

        recognition = RecognitionRepository.get_by_document_and_recognizer(
            test_session, document.id, mock_spacy_recognizer.id
        )
        assert recognition is None

    def test_handles_empty_document_list(self, mock_spacy_recognizer):
        """Test that predict handles empty document list gracefully."""
        # Arrange
        service = RecognitionService(mock_spacy_recognizer)

        # Act
        service.predict([])

        # Assert - Should not call predict on recognizer
        mock_spacy_recognizer.predict.assert_not_called()

    def test_processes_multiple_documents(
        self, test_session, mock_spacy_recognizer, document_factory
    ):
        """Test that predict handles multiple documents correctly."""
        # Arrange
        doc1 = document_factory(text="New York")
        doc2 = document_factory(text="Paris")
        mock_spacy_recognizer.predict.return_value = [[(0, 8)], [(0, 5)]]
        service = RecognitionService(mock_spacy_recognizer)

        # Act
        service.predict([doc1, doc2])

        # Assert - Both documents should have references
        from sqlmodel import select

        from geoparser.db.models import Reference

        statement1 = select(Reference).where(Reference.document_id == doc1.id)
        refs1 = test_session.exec(statement1).unique().all()
        assert len(refs1) == 1

        statement2 = select(Reference).where(Reference.document_id == doc2.id)
        refs2 = test_session.exec(statement2).unique().all()
        assert len(refs2) == 1


@pytest.mark.unit
class TestRecognitionServiceFit:
    """Preparation of annotated documents for recognizer training."""

    def test_reports_when_recognizer_does_not_implement_fit(self):
        """The missing-fit error identifies the recognizer kind and name."""
        recognizer = SimpleNamespace(name="ManualRecognizer")

        with pytest.raises(
            ValueError,
            match="Recognizer 'ManualRecognizer'",
        ):
            RecognitionService(cast(Any, recognizer)).fit([])

    def test_fits_only_annotated_documents_and_forwards_spans_and_options(
        self, mock_spacy_recognizer
    ):
        """Annotations become offsets while unannotated text is omitted."""
        documents = [
            SimpleNamespace(
                text="New York, Paris",
                toponyms=[
                    SimpleNamespace(start=0, end=8),
                    SimpleNamespace(start=10, end=15),
                ],
            ),
            SimpleNamespace(text="No places here", toponyms=[]),
            SimpleNamespace(
                text="London",
                toponyms=[SimpleNamespace(start=0, end=6)],
            ),
        ]
        fit = Mock()
        mock_spacy_recognizer.fit = fit
        service = RecognitionService(mock_spacy_recognizer)

        service.fit(cast(Any, documents), output_path="model", epochs=4)

        fit.assert_called_once_with(
            ["New York, Paris", "London"],
            [[(0, 8), (10, 15)], [(0, 6)]],
            output_path="model",
            epochs=4,
        )


@pytest.mark.unit
class TestRecognitionFailures:
    """What happens when persisting predictions fails, and on cache misses."""

    def test_rolls_back_and_reraises_when_recording_fails(
        self, mock_spacy_recognizer, document_factory
    ):
        """A failure while recording leaves no partial batch behind."""
        document = document_factory(text="New York is a city.")
        mock_spacy_recognizer.predict.return_value = [[(0, 8)]]
        service = RecognitionService(mock_spacy_recognizer)

        with (
            patch.object(
                service,
                "_record_reference_predictions",
                side_effect=RuntimeError("boom"),
            ),
            pytest.raises(RuntimeError, match="boom"),
        ):
            service.predict([document])

    def test_rolls_back_first_core_insert_when_second_insert_fails(
        self,
        test_session,
        mock_spacy_recognizer,
        document_factory,
        recognizer_factory,
    ):

        from sqlalchemy.orm import Session
        from sqlalchemy.sql.dml import Insert
        from sqlmodel import select

        from geoparser.db.models import Recognition, Reference

        document = document_factory(text="Paris Berlin")
        recognizer_factory(
            id=mock_spacy_recognizer.id,
            name=mock_spacy_recognizer.name,
            config=mock_spacy_recognizer.config,
        )
        mock_spacy_recognizer.predict.return_value = [[(0, 5), (6, 12)]]
        service = RecognitionService(mock_spacy_recognizer)
        original_execute = Session.execute
        insert_count = 0

        def fail_second_insert(session, statement, *args, **kwargs):
            nonlocal insert_count
            if isinstance(statement, Insert):
                insert_count += 1
                if insert_count == 2:
                    raise RuntimeError("recognition marker insert failed")
            return original_execute(session, statement, *args, **kwargs)

        with (
            patch.object(Session, "execute", new=fail_second_insert),
            pytest.raises(RuntimeError, match="recognition marker insert failed"),
        ):
            service.predict([document])

        assert insert_count == 2
        assert (
            test_session.exec(
                select(Reference).where(Reference.document_id == document.id)
            ).all()
            == []
        )
        assert (
            test_session.exec(
                select(Recognition).where(Recognition.document_id == document.id)
            ).all()
            == []
        )

    def test_cuts_the_reference_text_from_the_document(self, mock_spacy_recognizer):
        """The span's text comes from the document, with no query."""
        from types import SimpleNamespace

        document = SimpleNamespace(id=uuid.uuid4(), text="New York is a city.")
        service = RecognitionService(mock_spacy_recognizer)

        reference = service._create_reference_record(
            cast(Any, document), 0, 8, "recognizer"
        )

        assert reference["text"] == "New York"
        assert reference["document_id"] == document.id
        assert reference["recognizer_id"] == "recognizer"
        assert isinstance(reference["id"], uuid.UUID)

    def test_leaves_the_text_empty_for_a_document_without_one(
        self, mock_spacy_recognizer
    ):
        """A document with no string text yields a reference without text."""
        from types import SimpleNamespace

        document = SimpleNamespace(id=uuid.uuid4())
        service = RecognitionService(mock_spacy_recognizer)

        reference = service._create_reference_record(cast(Any, document), 0, 8, "r")

        assert reference["text"] is None


@pytest.mark.unit
class TestRecordReferencePredictions:
    """Writing a recognizer's spans."""

    @staticmethod
    def _record(documents, predictions):
        """Run the recorder, returning the reference and recognition writes."""
        service = RecognitionService(Mock())
        references, recognitions = [], []
        with (
            patch.object(
                service,
                "_create_reference_record",
                side_effect=lambda d, start, end, r: references.append(
                    (d.id, start, end)
                ),
            ),
            patch.object(
                service,
                "_create_recognition_record",
                side_effect=lambda d, r: recognitions.append(d),
            ),
        ):
            service._record_reference_predictions(Mock(), documents, predictions, "rec")
        return references, recognitions

    def test_tolerates_fewer_predictions_than_documents(self):
        """
        A recognizer that returns too few results is not an error.

        The documents it did cover are recorded and the rest are left for a
        later run, rather than the whole batch failing.
        """
        # Arrange
        documents = [SimpleNamespace(id="d1"), SimpleNamespace(id="d2")]

        # Act
        references, recognitions = self._record(documents, [[(0, 5)]])

        # Assert
        assert references == [("d1", 0, 5)]
        assert recognitions == ["d1"]

    def test_skips_a_none_prediction_and_keeps_going(self):
        """
        None means "could not process this document", not "stop".

        Breaking out here would silently drop every later document whenever
        one came back unprocessed.
        """
        # Arrange
        documents = [
            SimpleNamespace(id="d1"),
            SimpleNamespace(id="d2"),
            SimpleNamespace(id="d3"),
        ]

        # Act
        references, recognitions = self._record(documents, [[(0, 1)], None, [(2, 3)]])

        # Assert
        assert references == [("d1", 0, 1), ("d3", 2, 3)]
        assert recognitions == ["d1", "d3"]

    def test_marks_a_document_processed_only_once_per_recognizer(self):
        """Each covered document gets exactly one recognition record."""
        # Arrange
        documents = [SimpleNamespace(id="d1")]

        # Act
        _, recognitions = self._record(documents, [[(0, 1), (2, 3)]])

        # Assert
        assert recognitions == ["d1"]


@pytest.mark.unit
class TestRecognitionBatchPersistence:
    """The services stage validated mappings in core bulk writes."""

    def test_recognition_core_executes_ordered_insertions(
        self, recorded_recognition_batch
    ):
        (
            session,
            reference_statement,
            _,
            recognition_statement,
            _,
            _,
            _,
        ) = recorded_recognition_batch
        assert len(session.execute.call_args_list) == 2
        assert (reference_statement.table.name, recognition_statement.table.name) == (
            "reference",
            "recognition",
        )

    def test_recognition_core_stores_reference_rows(self, recorded_recognition_batch):
        _, _, ids, _, document, reference_rows, _ = recorded_recognition_batch
        assert reference_rows == [
            {
                "id": ids[0],
                "start": 0,
                "end": 5,
                "text": "Paris",
                "document_id": document.id,
                "recognizer_id": "rec",
            },
            {
                "id": ids[1],
                "start": 6,
                "end": 12,
                "text": "Berlin",
                "document_id": document.id,
                "recognizer_id": "rec",
            },
        ]

    def test_recognition_core_stores_processing_marker(
        self, recorded_recognition_batch
    ):
        _, _, ids, _, document, _, recognition_rows = recorded_recognition_batch
        assert recognition_rows == [
            {"id": ids[2], "document_id": document.id, "recognizer_id": "rec"}
        ]

    def test_recognition_core_avoids_orm_additions(self, recorded_recognition_batch):
        session = recorded_recognition_batch[0]
        session.add_all.assert_not_called()
        session.commit.assert_not_called()


@pytest.fixture
def recorded_recognition_batch():
    """Record one recognition batch and expose its staged database rows."""
    ids = [uuid.uuid4() for _ in range(3)]
    document = SimpleNamespace(id=uuid.uuid4(), text="Paris Berlin")
    service = RecognitionService(Mock())
    session = Mock()
    with patch("geoparser.services.recognition.uuid.uuid4", side_effect=ids):
        service._record_reference_predictions(
            session, cast(Any, [document]), [[(0, 5), (6, 12)]], "rec"
        )
    reference_statement, reference_rows = session.execute.call_args_list[0].args
    recognition_statement, recognition_rows = session.execute.call_args_list[1].args
    return (
        session,
        reference_statement,
        ids,
        recognition_statement,
        document,
        reference_rows,
        recognition_rows,
    )


@pytest.mark.unit
class TestRecognitionBatchStatusQueries:
    """Status filtering uses one set-based query per service batch."""

    def test_recognition_filters_documents_with_one_lookup(self):
        """Document status checks do not query once per document."""
        documents = [SimpleNamespace(id="d1"), SimpleNamespace(id="d2")]
        service = RecognitionService(Mock())
        session = Mock()

        with patch(
            "geoparser.services.recognition.RecognitionRepository.get_processed_document_ids",
            return_value={"d2"},
        ) as lookup:
            remaining = service._filter_unprocessed_documents(
                session, cast(Any, documents), "rec"
            )

        assert remaining == [documents[0]]
        lookup.assert_called_once_with(session, ["d1", "d2"], "rec")
