"""
Tests for how the services record what a pluggable module predicted.

Recognizers and resolvers are supplied by the caller, so the services treat a
short prediction list leniently -- the trailing items are simply left
unprocessed -- and skip individual predictions that come back as None. Both
behaviours are easy to break into either a crash or a silent early exit.
"""

import uuid
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from geoparser.services.recognition import RecognitionService
from geoparser.services.resolution import ResolutionService


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
class TestBatchPersistence:
    """The services stage model rows and issue one write per prediction batch."""

    def test_recognition_records_all_rows_with_one_add_all(self):
        """References and the processing marker are submitted together."""
        document = SimpleNamespace(id=uuid.uuid4(), text="Paris Berlin")
        service = RecognitionService(Mock())

        session = Mock()
        service._record_reference_predictions(
            session, [document], [[(0, 5), (6, 12)]], "rec"
        )

        session.add_all.assert_called_once()
        assert len(session.add_all.call_args.args[0]) == 3

    def test_resolution_records_all_rows_with_one_add_all(self):
        """Referents and resolution markers are submitted together."""
        references = [
            SimpleNamespace(id=uuid.uuid4()),
            SimpleNamespace(id=uuid.uuid4()),
        ]
        service = ResolutionService(Mock())
        feature = SimpleNamespace(identifier="123")
        session = Mock()

        with patch("geoparser.services.resolution.Gazetteer") as gazetteer:
            gazetteer.return_value.find.return_value = feature
            service._record_referent_prediction_groups(
                session,
                [references],
                [[("geonames", "123"), ("geonames", "123")]],
                "res",
            )

        session.add_all.assert_called_once()
        assert len(session.add_all.call_args.args[0]) == 4


@pytest.mark.unit
class TestBatchStatusQueries:
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
            remaining = service._filter_unprocessed_documents(session, documents, "rec")

        assert remaining == [documents[0]]
        lookup.assert_called_once_with(session, ["d1", "d2"], "rec")

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
                session, documents, "res"
            )

        assert texts == ["text"]
        assert boundaries == [[(0, 1)]]
        assert remaining == [[references[0]]]
        lookup.assert_called_once_with(session, ["r1", "r2"], "res")
