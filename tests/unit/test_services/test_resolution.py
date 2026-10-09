"""
Unit tests for geoparser/services/resolution.py

Tests the ResolutionService class with mocked resolvers.
"""

import uuid
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

import pytest

from geoparser.services.resolution import ResolutionService


@pytest.mark.unit
class TestResolutionServiceInitialization:
    """Test ResolutionService initialization."""

    def test_creates_with_resolver(self, mock_sentencetransformer_resolver):
        """Test that ResolutionService can be created with a resolver."""
        # Arrange & Act
        service = ResolutionService(mock_sentencetransformer_resolver)

        # Assert
        assert service.resolver == mock_sentencetransformer_resolver


@pytest.mark.unit
class TestResolutionServicePredict:
    """Test ResolutionService predict method."""

    def test_ensures_resolver_record_exists(
        self, test_session, mock_sentencetransformer_resolver, document_factory
    ):
        """Test that predict ensures resolver record exists in database."""
        # Arrange
        document = document_factory(text="Test document")
        mock_sentencetransformer_resolver.predict.return_value = [[]]
        service = ResolutionService(mock_sentencetransformer_resolver)

        # Act
        service.predict([document])

        # Assert - Resolver record should be created
        from geoparser.db.crud import ResolverRepository

        resolver = ResolverRepository.get(
            test_session, mock_sentencetransformer_resolver.id
        )
        assert resolver is not None
        assert resolver.id == mock_sentencetransformer_resolver.id

    def test_calls_resolver_predict(
        self,
        test_session,
        mock_sentencetransformer_resolver,
        document_factory,
        reference_factory,
    ):
        """Test that predict calls the resolver's predict method."""
        # Arrange
        document = document_factory(text="New York is a city.")
        reference_factory(start=0, end=8, document_id=document.id)
        test_session.refresh(document)

        mock_sentencetransformer_resolver.predict.return_value = [
            [None]
        ]  # Return None to skip referent creation
        service = ResolutionService(mock_sentencetransformer_resolver)

        # Act
        service.predict([document])

        # Assert
        mock_sentencetransformer_resolver.predict.assert_called_once()
        # Check that it was called with the document text and reference boundaries
        call_args = mock_sentencetransformer_resolver.predict.call_args
        assert call_args[0][0] == ["New York is a city."]
        assert call_args[0][1] == [[(0, 8)]]

    def test_creates_resolution_record(
        self,
        test_session,
        mock_sentencetransformer_resolver,
        document_factory,
        reference_factory,
    ):
        """Test that predict creates a resolution record marking reference as processed."""
        # Arrange
        from unittest.mock import Mock

        document = document_factory(text="Test")
        reference = reference_factory(start=0, end=4, document_id=document.id)
        test_session.refresh(document)

        mock_sentencetransformer_resolver.predict.return_value = [
            [("geonames", "123456")]
        ]
        service = ResolutionService(mock_sentencetransformer_resolver)

        # Mock the gazetteer lookup validating the predicted feature
        fake_feature = Mock(identifier="123456")
        with patch("geoparser.services.resolution.Gazetteer") as mock_gazetteer:
            mock_gazetteer.return_value.find.return_value = fake_feature

            # Act
            service.predict([document])

        # Assert - Resolution record should exist
        from geoparser.db.crud import ResolutionRepository

        resolution = ResolutionRepository.get_by_reference_and_resolver(
            test_session, reference.id, mock_sentencetransformer_resolver.id
        )
        assert resolution is not None

    def test_raises_when_predicted_feature_does_not_exist(
        self,
        test_session,
        mock_sentencetransformer_resolver,
        document_factory,
        reference_factory,
    ):
        """A prediction pointing at a non-existent feature raises clearly."""
        document = document_factory(text="Test")
        reference_factory(start=0, end=4, document_id=document.id)
        test_session.refresh(document)

        mock_sentencetransformer_resolver.predict.return_value = [
            [("geonames", "does-not-exist")]
        ]
        service = ResolutionService(mock_sentencetransformer_resolver)

        with patch("geoparser.services.resolution.Gazetteer") as mock_gazetteer:
            mock_gazetteer.return_value.find.return_value = None

            with pytest.raises(ValueError, match="does not exist in gazetteer"):
                service.predict([document])

    def test_invalid_prediction_does_not_leave_partial_batch_writes(
        self,
        test_session,
        mock_sentencetransformer_resolver,
        document_factory,
        reference_factory,
    ):
        """All referent validation happens before the batch is committed."""
        from unittest.mock import Mock

        document = document_factory(text="Test Test")
        first = reference_factory(start=0, end=4, document_id=document.id)
        second = reference_factory(start=5, end=9, document_id=document.id)
        test_session.refresh(document)
        mock_sentencetransformer_resolver.predict.return_value = [
            [("geonames", "123"), ("geonames", "missing")]
        ]
        service = ResolutionService(mock_sentencetransformer_resolver)
        feature = Mock(identifier="123")

        with patch("geoparser.services.resolution.Gazetteer") as gazetteer:
            gazetteer.return_value.find.side_effect = [feature, None]
            with pytest.raises(ValueError, match="does not exist in gazetteer"):
                service.predict([document])

        from geoparser.db.crud import ReferentRepository, ResolutionRepository

        assert (
            ReferentRepository.get_by_reference(test_session, first.id),
            ReferentRepository.get_by_reference(test_session, second.id),
            ResolutionRepository.get_by_reference(test_session, first.id),
            ResolutionRepository.get_by_reference(test_session, second.id),
        ) == ([], [], [], [])

    def test_rolls_back_referent_insert_when_resolution_insert_fails(
        self,
        test_session,
        mock_sentencetransformer_resolver,
        document_factory,
        reference_factory,
        resolver_factory,
    ):
        from types import SimpleNamespace

        from sqlalchemy.orm import Session
        from sqlalchemy.sql.dml import Insert

        document = document_factory(text="Paris Berlin")
        first = reference_factory(start=0, end=5, document_id=document.id)
        second = reference_factory(start=6, end=12, document_id=document.id)
        test_session.refresh(document)
        resolver_factory(
            id=mock_sentencetransformer_resolver.id,
            name=mock_sentencetransformer_resolver.name,
            config=mock_sentencetransformer_resolver.config,
        )
        mock_sentencetransformer_resolver.predict.return_value = [
            [("geonames", "1"), ("geonames", "2")]
        ]
        service = ResolutionService(mock_sentencetransformer_resolver)
        original_execute = Session.execute
        insert_count = 0

        def fail_second_insert(session, statement, *args, **kwargs):
            nonlocal insert_count
            if isinstance(statement, Insert):
                insert_count += 1
                if insert_count == 2:
                    raise RuntimeError("resolution marker insert failed")
            return original_execute(session, statement, *args, **kwargs)

        with (
            patch("geoparser.services.resolution.Gazetteer") as gazetteer,
            patch.object(Session, "execute", new=fail_second_insert),
            pytest.raises(RuntimeError, match="resolution marker insert failed"),
        ):
            gazetteer.return_value.find.side_effect = [
                SimpleNamespace(identifier="1"),
                SimpleNamespace(identifier="2"),
            ]
            service.predict([document])

        from geoparser.db.crud import ReferentRepository, ResolutionRepository

        assert (
            insert_count,
            ReferentRepository.get_by_reference(test_session, first.id),
            ReferentRepository.get_by_reference(test_session, second.id),
            ResolutionRepository.get_by_reference(test_session, first.id),
            ResolutionRepository.get_by_reference(test_session, second.id),
        ) == (2, [], [], [], [])

    def test_skips_references_when_resolver_returns_none(
        self,
        test_session,
        mock_sentencetransformer_resolver,
        document_factory,
        reference_factory,
    ):
        """Test that predict skips references when resolver returns None (no valid prediction)."""
        # Arrange
        document = document_factory(text="Test")
        reference = reference_factory(start=0, end=4, document_id=document.id)
        test_session.refresh(document)

        # Resolver returns None indicating it couldn't make a valid prediction
        mock_sentencetransformer_resolver.predict.return_value = [[None]]
        service = ResolutionService(mock_sentencetransformer_resolver)

        # Act
        service.predict([document])

        # Assert - No resolution record should be created
        from geoparser.db.crud import ResolutionRepository

        resolution = ResolutionRepository.get_by_reference_and_resolver(
            test_session, reference.id, mock_sentencetransformer_resolver.id
        )
        assert resolution is None

    def test_skips_already_processed_references(
        self,
        test_session,
        mock_sentencetransformer_resolver,
        document_factory,
        reference_factory,
        resolver_factory,
    ):
        """Test that predict skips references already processed by this resolver."""
        # Arrange
        resolver_record = resolver_factory(
            id=mock_sentencetransformer_resolver.id,
            name=mock_sentencetransformer_resolver.name,
            config=mock_sentencetransformer_resolver.config,
        )
        document = document_factory(text="Test")
        reference = reference_factory(start=0, end=4, document_id=document.id)
        test_session.refresh(document)

        # Mark as already processed
        from geoparser.db.crud import ResolutionRepository
        from geoparser.db.models import ResolutionCreate

        ResolutionRepository.create(
            test_session,
            ResolutionCreate(reference_id=reference.id, resolver_id=resolver_record.id),
        )

        mock_sentencetransformer_resolver.predict.return_value = [[("geonames", "123")]]
        service = ResolutionService(mock_sentencetransformer_resolver)

        # Act
        service.predict([document])

        # Assert - Predict should not be called since reference was already processed
        mock_sentencetransformer_resolver.predict.assert_not_called()

    def test_handles_none_predictions(
        self,
        test_session,
        mock_sentencetransformer_resolver,
        document_factory,
        reference_factory,
    ):
        """Test that predict handles None predictions (unavailable) correctly."""
        # Arrange
        document = document_factory(text="Test")
        reference = reference_factory(start=0, end=4, document_id=document.id)
        test_session.refresh(document)

        mock_sentencetransformer_resolver.predict.return_value = [
            [None]
        ]  # Prediction not available
        service = ResolutionService(mock_sentencetransformer_resolver)

        # Act
        service.predict([document])

        # Assert - No referents should be created, no resolution record
        from sqlmodel import select

        from geoparser.db.crud import ResolutionRepository
        from geoparser.db.models import Referent

        statement = select(Referent).where(Referent.reference_id == reference.id)
        referents = test_session.exec(statement).all()
        assert len(referents) == 0

        resolution = ResolutionRepository.get_by_reference_and_resolver(
            test_session, reference.id, mock_sentencetransformer_resolver.id
        )
        assert resolution is None

    def test_handles_empty_document_list(self, mock_sentencetransformer_resolver):
        """Test that predict handles empty document list gracefully."""
        # Arrange
        service = ResolutionService(mock_sentencetransformer_resolver)

        # Act
        service.predict([])

        # Assert - Should not call predict on resolver
        mock_sentencetransformer_resolver.predict.assert_not_called()

    def test_processes_multiple_documents(
        self,
        test_session,
        mock_sentencetransformer_resolver,
        document_factory,
        reference_factory,
    ):
        """Test that predict handles multiple documents correctly."""
        # Arrange
        from unittest.mock import Mock

        doc1 = document_factory(text="New York")
        doc2 = document_factory(text="Paris")
        ref1 = reference_factory(start=0, end=8, document_id=doc1.id)
        ref2 = reference_factory(start=0, end=5, document_id=doc2.id)
        test_session.refresh(doc1)
        test_session.refresh(doc2)

        mock_sentencetransformer_resolver.predict.return_value = [
            [("geonames", "123456")],
            [("geonames", "123456")],
        ]
        service = ResolutionService(mock_sentencetransformer_resolver)

        # Mock the gazetteer lookup validating the predicted features
        fake_feature = Mock(identifier="123456")
        with patch("geoparser.services.resolution.Gazetteer") as mock_gazetteer:
            mock_gazetteer.return_value.find.return_value = fake_feature

            # Act
            service.predict([doc1, doc2])

        # Assert - Both references should have resolutions
        from geoparser.db.crud import ResolutionRepository

        resolution1 = ResolutionRepository.get_by_reference_and_resolver(
            test_session, ref1.id, mock_sentencetransformer_resolver.id
        )
        resolution2 = ResolutionRepository.get_by_reference_and_resolver(
            test_session, ref2.id, mock_sentencetransformer_resolver.id
        )
        assert resolution1 is not None
        assert resolution2 is not None


@pytest.mark.unit
class TestResolutionServiceFit:
    """Test ResolutionService fit method."""

    def test_raises_error_if_resolver_has_no_fit_method(self):
        """Test that fit raises error if resolver doesn't implement fit."""
        from types import SimpleNamespace

        # Arrange
        mock_manual_resolver = SimpleNamespace(name="manual")
        service = ResolutionService(cast(Any, mock_manual_resolver))

        # Act & Assert
        with pytest.raises(ValueError) as error:
            service.fit([])

        assert str(error.value) == ("Resolver 'manual' does not implement a fit method")

    def test_calls_resolver_fit_with_training_data(
        self,
        test_session,
        mock_sentencetransformer_resolver,
    ):
        """Test that fit calls resolver's fit method with prepared training data."""
        # Arrange
        # Mock fit method
        mock_sentencetransformer_resolver.fit = lambda *args, **kwargs: None

        service = ResolutionService(mock_sentencetransformer_resolver)

        # This test would require more complex setup with documents, references, and referents
        # For now, we just test that it doesn't error with empty documents
        # Act & Assert - Should not raise error
        service.fit([], output_path="/tmp/model")


@pytest.mark.unit
def test_training_reads_each_toponyms_location_once(mock_sentencetransformer_resolver):
    """A referent's location opens the gazetteer, so it is read only once."""
    from types import SimpleNamespace
    from unittest.mock import PropertyMock

    location = SimpleNamespace(gazetteer_name="geonames", identifier="1")
    reads = PropertyMock(return_value=location)
    reference_type = type("Ref", (), {"location": reads, "start": 0, "end": 5})
    document = SimpleNamespace(toponyms=[reference_type()])
    service = ResolutionService(mock_sentencetransformer_resolver)

    spans, pairs = service._annotated_pairs(cast(Any, document))

    assert (spans, pairs) == ([(0, 5)], [("geonames", "1")])
    assert reads.call_count == 1


def _toponym(start: int, end: int, gazetteer: str | None = None, identifier: str = ""):
    """A reference, optionally resolved to a gazetteer feature."""
    location = (
        SimpleNamespace(gazetteer_name=gazetteer, identifier=identifier)
        if gazetteer
        else None
    )
    return SimpleNamespace(start=start, end=end, location=location)


def _document(text: str, toponyms: list) -> Any:
    """A document stub carrying only what the service reads."""
    return SimpleNamespace(text=text, toponyms=toponyms)


@pytest.mark.unit
class TestAnnotatedPairs:
    """Extracting one document's resolved toponyms."""

    def test_pairs_each_span_with_its_referent(self):
        """Spans and referents come back aligned, in document order."""
        # Arrange
        doc = _document(
            "Paris and Berlin",
            [_toponym(0, 5, "geonames", "1"), _toponym(10, 16, "geonames", "2")],
        )

        # Act
        spans, referents = ResolutionService._annotated_pairs(doc)

        # Assert
        assert spans == [(0, 5), (10, 16)]
        assert referents == [("geonames", "1"), ("geonames", "2")]

    def test_drops_toponyms_that_were_never_resolved(self):
        """An unresolved toponym contributes neither a span nor a referent."""
        # Arrange
        doc = _document(
            "Paris and Nowhere",
            [_toponym(0, 5, "geonames", "1"), _toponym(10, 17)],
        )

        # Act
        spans, referents = ResolutionService._annotated_pairs(doc)

        # Assert
        assert spans == [(0, 5)]
        assert referents == [("geonames", "1")]

    def test_returns_two_empty_lists_for_an_unannotated_document(self):
        """Nothing resolved means nothing to train on."""
        # Arrange
        doc = _document("Nothing here", [_toponym(0, 7)])

        # Act
        spans, referents = ResolutionService._annotated_pairs(doc)

        # Assert
        assert spans == []
        assert referents == []


@pytest.mark.unit
class TestFitDataFlow:
    """What reaches the resolver's own fit method."""

    @staticmethod
    def _service_with_resolver():
        """A service whose resolver records how fit was called."""
        resolver = Mock()
        resolver.name = "TestResolver"
        return ResolutionService(resolver), resolver

    def test_passes_texts_references_and_referents_in_step(self):
        """The three lists line up index for index."""
        # Arrange
        service, resolver = self._service_with_resolver()
        documents = [
            _document("Paris", [_toponym(0, 5, "geonames", "1")]),
            _document("Berlin", [_toponym(0, 6, "geonames", "2")]),
        ]

        # Act
        service.fit(documents)

        # Assert
        texts, references, referents = resolver.fit.call_args.args
        assert texts == ["Paris", "Berlin"]
        assert references == [[(0, 5)], [(0, 6)]]
        assert referents == [[("geonames", "1")], [("geonames", "2")]]

    def test_skips_documents_with_no_resolved_toponyms(self):
        """A document with nothing annotated is left out of training."""
        # Arrange
        service, resolver = self._service_with_resolver()
        documents = [
            _document("Paris", [_toponym(0, 5, "geonames", "1")]),
            _document("Unannotated", [_toponym(0, 5)]),
        ]

        # Act
        service.fit(documents)

        # Assert
        texts, references, referents = resolver.fit.call_args.args
        assert texts == ["Paris"]
        assert references == [[(0, 5)]]
        assert referents == [[("geonames", "1")]]

    def test_forwards_training_parameters(self):
        """Extra keyword arguments reach the resolver untouched."""
        # Arrange
        service, resolver = self._service_with_resolver()
        documents = [_document("Paris", [_toponym(0, 5, "geonames", "1")])]

        # Act
        service.fit(documents, epochs=7, output_path="/tmp/out")

        # Assert
        assert resolver.fit.call_args.kwargs == {"epochs": 7, "output_path": "/tmp/out"}

    def test_rejects_a_resolver_that_cannot_be_trained(self):
        """A resolver without a fit method is reported by name."""
        # Arrange
        resolver = SimpleNamespace(name="ManualResolver")
        service = ResolutionService(cast(Any, resolver))

        # Act & Assert
        with pytest.raises(ValueError, match="ManualResolver"):
            service.fit([_document("Paris", [_toponym(0, 5, "geonames", "1")])])


@pytest.mark.unit
class TestRecordReferentPredictions:
    """Writing a resolver's referents."""

    @staticmethod
    def _record(references, predictions):
        """Run the recorder, returning the referent and resolution writes."""
        service = ResolutionService(Mock())
        referents, resolutions = [], []
        with (
            patch.object(
                service,
                "_create_referent_record",
                side_effect=lambda ref, g, i, r: referents.append((ref, g, i)),
            ),
            patch.object(
                service,
                "_create_resolution_record",
                side_effect=lambda ref, r: resolutions.append(ref),
            ),
        ):
            service._record_referent_prediction_groups(
                Mock(), [references], [predictions], "res"
            )
        return referents, resolutions

    def test_tolerates_fewer_referents_than_references(self):
        """A short prediction list leaves the remaining references alone."""
        # Arrange
        references = [
            SimpleNamespace(id="r1", start=0, end=1),
            SimpleNamespace(id="r2", start=2, end=3),
        ]

        # Act
        referents, resolutions = self._record(references, [("geonames", "1")])

        # Assert
        assert referents == [("r1", "geonames", "1")]
        assert resolutions == ["r1"]

    def test_skips_an_unresolved_reference_and_keeps_going(self):
        """A None referent does not end the document's processing."""
        # Arrange
        references = [
            SimpleNamespace(id="r1"),
            SimpleNamespace(id="r2"),
            SimpleNamespace(id="r3"),
        ]

        # Act
        referents, resolutions = self._record(
            references, [("geonames", "1"), None, ("geonames", "3")]
        )

        # Assert
        assert referents == [("r1", "geonames", "1"), ("r3", "geonames", "3")]
        assert resolutions == ["r1", "r3"]

    def test_records_the_gazetteer_and_identifier_it_was_given(self):
        """Both halves of the referent reach the repository."""
        # Arrange
        references = [SimpleNamespace(id="r1")]

        # Act
        referents, _ = self._record(references, [("swissnames3d", "42")])

        # Assert
        assert referents == [("r1", "swissnames3d", "42")]


@pytest.mark.unit
class TestReferentValidation:
    """Checking a predicted referent against the installed gazetteer."""

    @staticmethod
    def _create(gazetteer_name, identifier, found=True):
        """Run the record creation, returning the Gazetteer mock."""
        service = ResolutionService(Mock())
        feature = SimpleNamespace(identifier=identifier) if found else None
        with patch("geoparser.services.resolution.Gazetteer") as gazetteer:
            gazetteer.return_value.find.return_value = feature
            service._create_referent_record(
                uuid.uuid4(), gazetteer_name, identifier, "res"
            )
        return gazetteer

    def test_looks_the_identifier_up_in_the_named_gazetteer(self):
        """
        The lookup uses both halves of the predicted referent.

        Opening the wrong gazetteer, or looking up the wrong identifier, would
        either reject a valid prediction or accept a bogus one, depending on
        what happened to be installed.
        """
        # Act
        gazetteer = self._create("swissnames3d", "42")

        # Assert
        gazetteer.assert_called_once_with("swissnames3d")
        gazetteer.return_value.find.assert_called_once_with("42")

    def test_rejects_an_identifier_the_gazetteer_does_not_have(self):
        """An unknown feature is an error naming both the id and gazetteer."""
        # Act & Assert
        with pytest.raises(ValueError, match=r"'999'.*'geonames'"):
            self._create("geonames", "999", found=False)


@pytest.mark.unit
class TestResolutionBatchPersistence:
    """The services stage validated mappings in core bulk writes."""

    def test_resolution_core_executes_ordered_insertions(
        self, recorded_resolution_batch
    ):
        session, referent_statement, _, resolution_statement, _, _, _ = (
            recorded_resolution_batch
        )
        assert len(session.execute.call_args_list) == 2
        assert (referent_statement.table.name, resolution_statement.table.name) == (
            "referent",
            "resolution",
        )

    def test_resolution_core_stores_referent_rows(self, recorded_resolution_batch):
        _, _, ids, _, references, referent_rows, _ = recorded_resolution_batch
        assert referent_rows == [
            {
                "id": ids[0],
                "reference_id": references[0].id,
                "gazetteer_name": "geonames",
                "feature_identifier": "123",
                "resolver_id": "res",
            },
            {
                "id": ids[2],
                "reference_id": references[1].id,
                "gazetteer_name": "geonames",
                "feature_identifier": "123",
                "resolver_id": "res",
            },
        ]

    def test_resolution_core_stores_processing_markers(self, recorded_resolution_batch):
        _, _, ids, _, references, _, resolution_rows = recorded_resolution_batch
        assert resolution_rows == [
            {"id": ids[1], "reference_id": references[0].id, "resolver_id": "res"},
            {"id": ids[3], "reference_id": references[1].id, "resolver_id": "res"},
        ]

    def test_resolution_core_avoids_orm_additions(self, recorded_resolution_batch):
        session = recorded_resolution_batch[0]
        session.add_all.assert_not_called()
        session.commit.assert_not_called()


@pytest.fixture
def recorded_resolution_batch():
    """Record one resolution batch and expose its staged database rows."""
    ids = [uuid.uuid4() for _ in range(4)]
    references = [
        SimpleNamespace(id=uuid.uuid4()),
        SimpleNamespace(id=uuid.uuid4()),
    ]
    service = ResolutionService(Mock())
    feature = SimpleNamespace(identifier="123")
    session = Mock()
    with patch("geoparser.services.resolution.Gazetteer") as gazetteer:
        gazetteer.return_value.find.return_value = feature
        with patch("geoparser.services.resolution.uuid.uuid4", side_effect=ids):
            service._record_referent_prediction_groups(
                session,
                cast(Any, [references]),
                [[("geonames", "123"), ("geonames", "123")]],
                "res",
            )
    referent_statement, referent_rows = session.execute.call_args_list[0].args
    resolution_statement, resolution_rows = session.execute.call_args_list[1].args
    return (
        session,
        referent_statement,
        ids,
        resolution_statement,
        references,
        referent_rows,
        resolution_rows,
    )


@pytest.mark.unit
class TestResolutionBatchStatusQueries:
    """Status filtering uses one set-based query per service batch."""

    def test_resolution_filters_references_with_one_lookup(self):
        """Reference status checks do not query once per reference."""
        references = [
            SimpleNamespace(id="r1", start=0, end=1),
            SimpleNamespace(id="r2", start=2, end=3),
        ]
        documents = [SimpleNamespace(text="text", references=references)]
        service = ResolutionService(Mock())
        session = Mock()

        with patch(
            "geoparser.services.resolution.ResolutionRepository.get_processed_reference_ids",
            return_value={"r2"},
        ) as lookup:
            texts, boundaries, remaining = service._collect_unprocessed(
                session, cast(Any, documents), "res"
            )

        assert texts == ["text"]
        assert boundaries == [[(0, 1)]]
        assert remaining == [[references[0]]]
        lookup.assert_called_once_with(session, ["r1", "r2"], "res")
