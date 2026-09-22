"""
Tests for the chunked, checkpointed runner.

The models are mocked: what is being tested is that work is divided, recorded
and resumed correctly, which is what decides whether a job that hits its
walltime has made progress or wasted a reservation.
"""

import pytest

from scripts.benchmark.checkpoint import Checkpoint, RunIdentity
from scripts.benchmark.corpus import Document, GoldSpan
from scripts.benchmark.runner import chunks, gold_annotations, score


def document(identifier="0"):
    """Build a document with one located gold span."""
    return Document(identifier, "Zurich", (GoldSpan(0, 6, "Zurich", 47.3769, 8.5417),))


def identity():
    """Build a run identity."""
    return RunIdentity("swapped", "abc", "geonames", 0.0, None, "commit")


class TestChunks:
    """Splitting the corpus into units of saved progress."""

    def test_splits_evenly(self):
        """Test an exact multiple."""
        assert chunks([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]

    def test_keeps_a_short_final_chunk(self):
        """Test that nothing is dropped."""
        assert chunks([1, 2, 3], 2) == [[1, 2], [3]]

    def test_handles_an_empty_sequence(self):
        """Test that finished work produces no chunks."""
        assert chunks([], 5) == []

    def test_chunk_of_one(self):
        """Test saving after every document."""
        assert chunks([1, 2], 1) == [[1], [2]]

    @pytest.mark.parametrize("size", [0, -1])
    def test_rejects_a_non_positive_size(self, size):
        """Test the size that would loop forever."""
        with pytest.raises(ValueError, match="at least 1"):
            chunks([1], size)


class TestGoldAnnotations:
    """Gold spans become annotations carrying their document."""

    def test_carries_coordinates_and_document(self):
        """Test that scoring can tell two documents apart."""
        annotations = gold_annotations([document("a"), document("b")])

        assert [a.document_id for a in annotations] == ["a", "b"]
        assert all(a.latitude == 47.3769 for a in annotations)


class TestScore:
    """Turning a checkpoint into a result."""

    def test_scores_both_phases_when_present(self):
        """Test a complete run."""
        state = Checkpoint(identity())
        placed = {
            "start": 0,
            "end": 6,
            "identifier": None,
            "document_id": "0",
            "latitude": 47.3769,
            "longitude": 8.5417,
        }
        state.record("recognition", "0", [placed])
        state.record("resolution", "0", [placed])

        result = score("swapped", [document()], state, models={}, device="cpu")

        assert result.recognition["f1"] == 1.0
        assert result.resolution["accuracy_at_161km"] == 1.0

    def test_omits_a_phase_that_did_not_run(self):
        """Test that an unrun phase is absent rather than zero."""
        state = Checkpoint(identity())

        result = score("swapped", [document()], state, models={}, device="cpu")

        assert result.recognition == {}
        assert result.resolution == {}

    def test_carries_provenance_into_the_result(self):
        """Test that the report can say where and with what this ran."""
        state = Checkpoint(identity())
        state.elapsed_seconds = 9.0

        result = score(
            "swapped", [document()], state, models={"resolver": "jina"}, device="cuda"
        )

        assert result.device == "cuda"
        assert result.models == {"resolver": "jina"}
        assert result.elapsed_seconds == 9.0
