"""
Tests for resumable checkpoints.

The behaviour that matters is refusal: resuming across a change of commit,
corpus or settings would blend two experiments into one set of numbers, and
nothing downstream could tell. These tests pin that it starts over instead.
"""

import json
from dataclasses import asdict
from typing import Any

import pytest

from scripts.benchmark.checkpoint import (
    SCHEMA_VERSION,
    Checkpoint,
    RunIdentity,
    load,
    resolution_reuse_compatible,
    save,
)


def identity(**overrides: Any) -> RunIdentity:
    """Build a run identity with sensible defaults."""
    fields: dict[str, Any] = {
        "pipeline": "swapped",
        "corpus_digest": "abc123",
        "gazetteer": "geonames",
        "gazetteer_sha256": "gazetteer-sha256",
        "min_similarity": 0.0,
        "limit": None,
        "commit": "deadbee",
    }
    fields.update(overrides)
    return RunIdentity(**fields)


class TestCheckpointState:
    """Recording and reading back per-document predictions."""

    def test_records_and_reports_completion(self):
        """Test that a recorded document counts as done."""
        state = Checkpoint(identity())

        state.record("recognition", "3", [{"start": 0, "end": 4}])

        assert state.completed("recognition") == {"3"}
        assert state.completed("resolution") == set()

    def test_recording_twice_does_not_duplicate(self):
        """Test that a document reprocessed after a crash counts once."""
        state = Checkpoint(identity())

        state.record("resolution", "1", [{"start": 0, "end": 4}])
        state.record("resolution", "1", [{"start": 0, "end": 4}])

        assert len(state.annotations("resolution")) == 1

    def test_annotations_flattens_every_document(self):
        """Test that scoring sees all documents' annotations."""
        state = Checkpoint(identity())
        state.record("resolution", "1", [{"a": 1}])
        state.record("resolution", "2", [{"a": 2}, {"a": 3}])

        assert len(state.annotations("resolution")) == 3

    def test_rejects_an_unknown_phase(self):
        """Test that a typo cannot silently create a third phase."""
        state = Checkpoint(identity())

        with pytest.raises(ValueError, match="Unknown phase"):
            state.record("recogniton", "1", [])


class TestRoundTrip:
    """Persisting and resuming."""

    def test_saves_and_resumes(self, tmp_path):
        """Test that saved work is skipped on the next run."""
        path = tmp_path / "checkpoint.json"
        state = Checkpoint(identity())
        state.record("resolution", "7", [{"start": 1, "end": 2}])
        state.elapsed_seconds = 12.5
        state.phase_elapsed_seconds["resolution"] = 8.0
        save(path, state)

        resumed, reasons = load(path, identity())

        assert reasons == []
        assert resumed.completed("resolution") == {"7"}
        assert resumed.elapsed_seconds == 12.5
        assert resumed.phase_elapsed_seconds == {"resolution": 8.0}

    def test_creates_parent_directories(self, tmp_path):
        """Test that a fresh output directory does not need to exist."""
        path = tmp_path / "nested" / "deeper" / "checkpoint.json"

        save(path, Checkpoint(identity()))

        assert path.exists()

    def test_leaves_no_temporary_file(self, tmp_path):
        """Test that the atomic write cleans up after itself."""
        path = tmp_path / "checkpoint.json"

        save(path, Checkpoint(identity()))

        assert list(tmp_path.iterdir()) == [path]


class TestRefusalToResume:
    """When continuing would blend two experiments."""

    def test_missing_checkpoint_starts_fresh_without_complaint(self, tmp_path):
        """Test that a first run is not an error."""
        state, reasons = load(tmp_path / "absent.json", identity())

        assert reasons == []
        assert state.completed("resolution") == set()

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("commit", "0ther"),
            ("corpus_digest", "different"),
            ("min_similarity", 0.5),
            ("gazetteer", "swissnames3d"),
            ("gazetteer_sha256", "different-gazetteer-content"),
            ("limit", 10),
            ("pipeline", "upstream"),
            ("schema_version", SCHEMA_VERSION + 1),
            ("execution_settings", {"document_chunk_size": 7}),
        ],
    )
    def test_starts_fresh_when_the_run_changed(self, tmp_path, field, value):
        """Test that each identity field is load-bearing."""
        path = tmp_path / "checkpoint.json"
        state = Checkpoint(identity())
        state.record("resolution", "7", [{"start": 1, "end": 2}])
        save(path, state)

        resumed, reasons = load(path, identity(**{field: value}))

        assert resumed.completed("resolution") == set()
        assert any(field in reason for reason in reasons)

    def test_starts_fresh_on_a_corrupt_checkpoint(self, tmp_path):
        """Test that a half-written file does not stop the run."""
        path = tmp_path / "checkpoint.json"
        path.write_text('{"identity": {"pipeline"', encoding="utf-8")

        state, reasons = load(path, identity())

        assert state.completed("resolution") == set()
        assert any("unreadable" in reason for reason in reasons)

    def test_starts_fresh_when_identity_fields_are_missing(self, tmp_path):
        """Test an older checkpoint whose shape no longer matches."""
        path = tmp_path / "checkpoint.json"
        path.write_text(json.dumps({"identity": {"pipeline": "swapped"}}), "utf-8")

        state, reasons = load(path, identity())

        assert state.completed("resolution") == set()
        assert any("unreadable" in reason for reason in reasons)

    def test_starts_fresh_on_a_checkpoint_without_a_gazetteer_hash(self, tmp_path):
        """Old name-only gazetteer evidence cannot resume under a pinned run."""
        path = tmp_path / "checkpoint.json"
        old_identity = asdict(identity())
        old_identity.pop("gazetteer_sha256")
        path.write_text(json.dumps({"identity": old_identity}), encoding="utf-8")

        resumed, reasons = load(path, identity())

        assert resumed.completed("resolution") == set()
        assert any("gazetteer_sha256" in reason for reason in reasons)


class TestRunIdentity:
    """Comparing two runs."""

    def test_identical_identities_have_no_differences(self):
        """Test the resumable case."""
        assert identity().differences(identity()) == []

    def test_names_every_differing_field(self):
        """Test that the reason given to the operator is specific."""
        other = identity(commit="0ther", min_similarity=0.5)

        assert set(identity().differences(other)) == {"commit", "min_similarity"}

    def test_model_pins_are_load_bearing(self):
        first = identity(model_pins={"resolver": "minilm@one"})
        second = identity(model_pins={"resolver": "minilm@two"})

        assert first.differences(second) == ["model_pins"]

    def test_accelerator_is_load_bearing_for_resume(self):
        first = identity(accelerator="NVIDIA A100")
        second = identity(accelerator="NVIDIA T4")

        assert first.differences(second) == ["accelerator"]

    def test_gold_span_resolution_reuse_requires_matching_provenance(self):
        common = {
            "corpus_digest": "same-corpus",
            "gazetteer": "geonames",
            "gazetteer_sha256": "same-gazetteer-content",
            "min_similarity": 0.0,
            "limit": None,
            "commit": "same-code",
            "model_pins": {
                "recognizer": "different-recognizer",
                "resolver": "minilm@pin",
                "context_sentence_splitter": "xx_sent_ud_sm@3.8.0",
            },
            "execution_settings": {
                "document_chunk_size": 5,
                "recognition": {"batch_size": 8},
                "resolution": {
                    "context_encoding_batch_size": 32,
                    "candidate_encoding_batch_size": 32,
                },
            },
        }
        first = identity(pipeline="xlm-minilm", **common)
        second = identity(pipeline="otter-cross-minilm", **common)

        assert resolution_reuse_compatible(first, second)
        assert resolution_reuse_compatible(
            first,
            identity(
                pipeline="otter-cross-minilm",
                **{
                    **common,
                    "execution_settings": {
                        **common["execution_settings"],
                        "recognition": {"batch_size": 16},
                    },
                },
            ),
        )
        assert not resolution_reuse_compatible(
            first,
            identity(
                pipeline="otter-cross-minilm",
                **{**common, "corpus_digest": "other"},
            ),
        )
        assert not resolution_reuse_compatible(
            first,
            identity(
                pipeline="otter-cross-minilm",
                **{
                    **common,
                    "model_pins": {
                        **common["model_pins"],
                        "resolver": "minilm@other",
                    },
                },
            ),
        )
        assert not resolution_reuse_compatible(
            first,
            identity(
                pipeline="otter-cross-minilm",
                **{
                    **common,
                    "execution_settings": {
                        **common["execution_settings"],
                        "resolution": {
                            **common["execution_settings"]["resolution"],
                            "candidate_encoding_batch_size": 16,
                        },
                    },
                },
            ),
        )

    def test_gold_span_resolution_reuse_requires_a_gazetteer_hash(self):
        first = identity(pipeline="xlm-minilm", gazetteer_sha256=None)
        second = identity(pipeline="otter-cross-minilm", gazetteer_sha256=None)

        assert not resolution_reuse_compatible(first, second)

    def test_gold_span_resolution_reuse_requires_the_same_gazetteer_hash(self):
        first = identity(pipeline="xlm-minilm", gazetteer_sha256="hash-one")
        second = identity(pipeline="otter-cross-minilm", gazetteer_sha256="hash-two")

        assert not resolution_reuse_compatible(first, second)
