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
    recognition_reuse_compatible,
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
        "execution_settings": {
            "document_chunk_size": 5,
            "recognition": {"batch_size": 8},
            "resolution": {
                "context_encoding_batch_size": 32,
                "candidate_encoding_batch_size": 32,
            },
        },
    }
    fields.update(overrides)
    return RunIdentity(**fields)


def _assert_phase_completed(state: Checkpoint, phase: str, document_id: str) -> None:
    assert state.completed(phase) == {document_id}


def _assert_phase_setting_retained(
    state: Checkpoint, reference: RunIdentity, phase: str
) -> None:
    assert (
        state.identity.execution_settings[phase] == reference.execution_settings[phase]
    )


def _assert_reenabled_phase_results(state: Checkpoint) -> None:
    assert state.completed("recognition") == set()
    assert state.completed("resolution") == {"kept"}


def _assert_reenabled_phase_timing(state: Checkpoint) -> None:
    assert state.elapsed_seconds == 7.0
    assert state.phase_elapsed_seconds == {"resolution": 7.0}


def _assert_reenabled_phase_diagnostics_cleared(state: Checkpoint) -> None:
    assert "recognition" not in state.failures
    assert "recognition" not in state.phase_provenance


def _assert_inactive_phase_data_retained(state: Checkpoint) -> None:
    assert state.elapsed_seconds == 13.0
    assert state.phase_elapsed_seconds == {"recognition": 4.0, "resolution": 9.0}
    assert state.failures == {"recognition": {"1": {"error": "preserve"}}}
    assert state.phase_provenance == {"recognition": {"source_pipeline": "original"}}


def _assert_serialized_checkpoint_schema(payload: dict[str, Any]) -> None:
    assert set(payload) == {
        "identity",
        "recognition",
        "resolution",
        "elapsed_seconds",
        "phase_elapsed_seconds",
        "failures",
        "phase_provenance",
    }
    assert set(payload["identity"]) == set(asdict(identity()))
    assert payload["identity"]["schema_version"] == SCHEMA_VERSION


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

        _assert_serialized_checkpoint_schema(json.loads(path.read_text("utf-8")))
        resumed, reasons = load(path, identity())

        assert reasons == []
        assert resumed.completed("resolution") == {"7"}
        assert resumed.elapsed_seconds == 12.5
        assert resumed.phase_elapsed_seconds == {"resolution": 8.0}

    def test_phase_filtered_resume_retains_stored_settings_for_unrequested_phase(
        self, tmp_path
    ):
        path = tmp_path / "checkpoint.json"
        full_identity = identity()
        state = Checkpoint(full_identity)
        state.record("recognition", "1", [{"start": 0, "end": 1}])
        state.record("resolution", "1", [{"start": 0, "end": 1, "kb_id": "Q1"}])
        state.elapsed_seconds = 13.0
        state.phase_elapsed_seconds = {"recognition": 4.0, "resolution": 9.0}
        state.failures = {"recognition": {"1": {"error": "preserve"}}}
        state.phase_provenance = {"recognition": {"source_pipeline": "original"}}
        save(path, state)
        resolution_identity = identity(
            execution_settings={
                **full_identity.execution_settings,
                "recognition": None,
            }
        )

        resumed, reasons = load(path, resolution_identity)

        assert reasons == []
        _assert_phase_setting_retained(resumed, full_identity, "recognition")
        _assert_phase_completed(resumed, "recognition", "1")
        _assert_phase_completed(resumed, "resolution", "1")
        _assert_inactive_phase_data_retained(resumed)

        save(path, resumed)
        full_resume, reasons = load(path, full_identity)

        assert reasons == []
        _assert_phase_completed(full_resume, "recognition", "1")
        _assert_phase_completed(full_resume, "resolution", "1")
        _assert_inactive_phase_data_retained(full_resume)

    def test_reenabling_phase_discards_data_without_active_stored_settings(
        self, tmp_path
    ):
        path = tmp_path / "checkpoint.json"
        full_identity = identity()
        stored_identity = identity(
            execution_settings={
                **full_identity.execution_settings,
                "recognition": None,
            }
        )
        state = Checkpoint(stored_identity)
        state.record("recognition", "stale", [])
        state.record("resolution", "kept", [])
        state.elapsed_seconds = 10.0
        state.phase_elapsed_seconds = {"recognition": 3.0, "resolution": 7.0}
        state.failures = {"recognition": {"stale": {"error": "old"}}}
        state.phase_provenance = {"recognition": {"source_pipeline": "old"}}
        save(path, state)

        resumed, reasons = load(path, full_identity)

        assert reasons == []
        _assert_reenabled_phase_results(resumed)
        _assert_reenabled_phase_timing(resumed)
        _assert_reenabled_phase_diagnostics_cleared(resumed)

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
        assert reasons == [f"changed: {field}"]

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

    def test_starts_fresh_when_phase_settings_are_missing(self, tmp_path):
        path = tmp_path / "checkpoint.json"
        payload = {
            "identity": asdict(identity()),
            "recognition": {"stale": []},
            "resolution": {},
        }
        del payload["identity"]["execution_settings"]["recognition"]
        path.write_text(json.dumps(payload), encoding="utf-8")

        state, reasons = load(path, identity())

        assert state.completed("recognition") == set()
        assert state.completed("resolution") == set()
        assert any("execution_settings" in reason for reason in reasons)

    def test_starts_fresh_on_a_checkpoint_without_a_gazetteer_hash(self, tmp_path):
        """Old name-only gazetteer evidence cannot resume under a pinned run."""
        path = tmp_path / "checkpoint.json"
        old_identity = asdict(identity())
        old_identity.pop("gazetteer_sha256")
        path.write_text(json.dumps({"identity": old_identity}), encoding="utf-8")

        resumed, reasons = load(path, identity())

        assert resumed.completed("resolution") == set()
        assert any("gazetteer_sha256" in reason for reason in reasons)


def _gold_span_resolution_reuse_pair():
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
    return common, first, second


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

    def test_phase_filtered_identity_matches_shared_active_settings(self):
        full = identity()
        resolution_only = identity(
            execution_settings={
                **full.execution_settings,
                "recognition": None,
            }
        )

        assert resolution_only.differences(full) == []
        assert full.differences(resolution_only) == []

    @pytest.mark.parametrize("phase", ["recognition", "resolution"])
    def test_changed_active_phase_settings_are_load_bearing(self, phase):
        original = identity().execution_settings
        changed_settings = {
            **original,
            phase: {**original[phase], "changed": True},
        }

        assert "execution_settings" in identity().differences(
            identity(execution_settings=changed_settings)
        )

    def test_legacy_identity_without_phase_keys_is_incompatible(self):
        legacy_settings = identity().execution_settings.copy()
        del legacy_settings["recognition"]

        assert identity().differences(identity(execution_settings=legacy_settings)) == [
            "execution_settings"
        ]

    @pytest.mark.parametrize(
        "settings",
        [
            None,
            {},
            {
                "document_chunk_size": None,
                "recognition": {"batch_size": 8},
                "resolution": {
                    "context_encoding_batch_size": 32,
                    "candidate_encoding_batch_size": 32,
                },
            },
        ],
    )
    def test_incomplete_execution_settings_are_incompatible(self, settings):
        assert "execution_settings" in identity().differences(
            identity(execution_settings=settings)
        )
        assert "execution_settings" in identity(
            execution_settings=settings
        ).differences(identity())

    def test_additional_execution_settings_are_load_bearing(self):
        settings = {**identity().execution_settings, "extra": True}

        assert "execution_settings" in identity().differences(
            identity(execution_settings=settings)
        )

    def test_document_chunk_size_is_load_bearing(self):
        settings = identity().execution_settings
        settings["document_chunk_size"] = 6

        assert "execution_settings" in identity().differences(
            identity(execution_settings=settings)
        )

    def test_gold_span_resolution_reuse_requires_matching_provenance(self):
        _common, first, second = _gold_span_resolution_reuse_pair()

        assert resolution_reuse_compatible(first, second)

    def test_recognition_reuse_allows_different_resolvers_with_same_recognizer(self):
        common = {
            "corpus_digest": "same-corpus",
            "limit": None,
            "commit": "same-code",
            "scoring_view": "strict-raw-v1",
            "accelerator": "cuda (H100)",
            "model_pins": {
                "recognizer": "gliner@pin",
                "resolver": "minilm@pin",
            },
            "execution_settings": {
                "document_chunk_size": 5,
                "recognition": {"method": "per_document"},
                "resolution": {"batch_size": 32},
            },
        }
        first = identity(pipeline="gliner-minilm", **common)
        second = identity(
            pipeline="gliner-jina",
            **{
                **common,
                "model_pins": {
                    **common["model_pins"],
                    "resolver": "jina@pin",
                    "reranker": "reranker@pin",
                },
            },
        )

        assert recognition_reuse_compatible(first, second)

    @pytest.mark.parametrize(
        "change",
        [
            {"model_pins": {"recognizer": "other@pin"}},
            {"corpus_digest": "other-corpus"},
            {"scoring_view": "newsli-exact-full-record-dedup-v1"},
            {"commit": "other-code"},
            {"accelerator": "cpu"},
        ],
    )
    def test_recognition_reuse_rejects_a_changed_component_or_input(self, change):
        common = {
            "corpus_digest": "same-corpus",
            "limit": None,
            "commit": "same-code",
            "scoring_view": "strict-raw-v1",
            "accelerator": "cuda (H100)",
            "model_pins": {"recognizer": "gliner@pin"},
            "execution_settings": {
                "document_chunk_size": 5,
                "recognition": {"method": "per_document"},
                "resolution": None,
            },
        }
        first = identity(pipeline="gliner-minilm", **common)
        changed = identity(pipeline="gliner-jina", **{**common, **change})

        assert not recognition_reuse_compatible(first, changed)

    def test_gold_span_resolution_reuse_requires_resolver_pin(self):
        first = identity(model_pins={"reranker": "reranker@one"})
        second = identity(model_pins={"reranker": "reranker@one"})

        assert not resolution_reuse_compatible(first, second)

    def test_gold_span_resolution_reuse_allows_recognition_batch_changes(self):
        common, first, _second = _gold_span_resolution_reuse_pair()
        execution_settings = {
            **common["execution_settings"],
            "recognition": {"batch_size": 16},
        }
        changed = identity(
            pipeline="otter-cross-minilm",
            **{**common, "execution_settings": execution_settings},
        )

        assert resolution_reuse_compatible(first, changed)

    def test_gold_span_resolution_reuse_rejects_corpus_changes(self):
        common, first, _second = _gold_span_resolution_reuse_pair()
        changed = identity(
            pipeline="otter-cross-minilm",
            **{**common, "corpus_digest": "other"},
        )

        assert not resolution_reuse_compatible(first, changed)

    @pytest.mark.parametrize(
        "field,value",
        [
            ("gazetteer", "other-gazetteer"),
            ("min_similarity", 0.1),
            ("limit", 10),
            ("commit", "other-code"),
            ("schema_version", SCHEMA_VERSION + 1),
            ("scoring_view", "newsli-exact-full-record-dedup-v1"),
            ("accelerator", "cpu"),
        ],
    )
    def test_gold_span_resolution_reuse_rejects_resolution_input_changes(
        self, field, value
    ):
        common, first, _second = _gold_span_resolution_reuse_pair()
        changed = identity(
            pipeline="otter-cross-minilm",
            **{**common, field: value},
        )

        assert not resolution_reuse_compatible(first, changed)

    @pytest.mark.parametrize(
        "role", ["resolver", "reranker", "context_sentence_splitter"]
    )
    def test_gold_span_resolution_reuse_rejects_resolution_model_changes(self, role):
        common, first, _second = _gold_span_resolution_reuse_pair()
        model_pins = {**common["model_pins"], role: "different-pin"}
        changed = identity(
            pipeline="otter-cross-minilm",
            **{**common, "model_pins": model_pins},
        )

        assert not resolution_reuse_compatible(first, changed)

    def test_gold_span_resolution_reuse_rejects_document_chunk_size_changes(self):
        common, first, _second = _gold_span_resolution_reuse_pair()
        execution_settings = {
            **common["execution_settings"],
            "document_chunk_size": 10,
        }
        changed = identity(
            pipeline="otter-cross-minilm",
            **{**common, "execution_settings": execution_settings},
        )

        assert not resolution_reuse_compatible(first, changed)

    def test_gold_span_resolution_reuse_rejects_resolution_batch_changes(self):
        common, first, _second = _gold_span_resolution_reuse_pair()
        resolution = {
            **common["execution_settings"]["resolution"],
            "candidate_encoding_batch_size": 16,
        }
        execution_settings = {
            **common["execution_settings"],
            "resolution": resolution,
        }
        changed = identity(
            pipeline="otter-cross-minilm",
            **{**common, "execution_settings": execution_settings},
        )

        assert not resolution_reuse_compatible(first, changed)

    def test_gold_span_resolution_reuse_requires_a_gazetteer_hash(self):
        first = identity(pipeline="xlm-minilm", gazetteer_sha256=None)
        second = identity(pipeline="otter-cross-minilm", gazetteer_sha256=None)

        assert not resolution_reuse_compatible(first, second)

    def test_gold_span_resolution_reuse_requires_the_same_gazetteer_hash(self):
        first = identity(pipeline="xlm-minilm", gazetteer_sha256="hash-one")
        second = identity(pipeline="otter-cross-minilm", gazetteer_sha256="hash-two")

        assert not resolution_reuse_compatible(first, second)

    @pytest.mark.parametrize("inactive_key", ["document_chunk_size", "resolution"])
    def test_gold_span_resolution_reuse_requires_active_resolution_settings(
        self, inactive_key
    ):
        common, _first, second = _gold_span_resolution_reuse_pair()
        settings = common["execution_settings"].copy()
        settings[inactive_key] = None
        inactive = identity(
            pipeline="xlm-minilm",
            **{**common, "execution_settings": settings},
        )

        assert not resolution_reuse_compatible(inactive, second)
        assert not resolution_reuse_compatible(inactive, inactive)
