"""
Tests for the chunked, checkpointed runner.

The models are mocked: what is being tested is that work is divided, recorded
and resumed correctly, which is what decides whether a job that hits its
walltime has made progress or wasted a reservation.
"""

from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, call

import pytest

from scripts.benchmark import checkpoint as ckpt
from scripts.benchmark import runner
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


@pytest.mark.parametrize(
    ("location", "expected"),
    [
        (None, (None, None)),
        (SimpleNamespace(data=None), (None, None)),
        (
            SimpleNamespace(data={"latitude": "47.3769", "longitude": "8.5417"}),
            (47.3769, 8.5417),
        ),
        (SimpleNamespace(data={"latitude": "unknown", "longitude": 8}), (None, None)),
        (SimpleNamespace(data={"longitude": 8}), (None, None)),
    ],
)
def test_coordinates_require_two_usable_values(location, expected):
    assert runner._coordinates(location) == expected


def test_predictions_are_keyed_by_corpus_id_and_keep_unresolved_spans():
    located = SimpleNamespace(
        identifier="feature-1",
        data={"latitude": "47.3769", "longitude": "8.5417"},
    )
    project = SimpleNamespace(
        get_documents=Mock(
            return_value=[
                SimpleNamespace(
                    toponyms=[
                        SimpleNamespace(start=0, end=6, location=located),
                        SimpleNamespace(start=11, end=15, location=None),
                    ]
                ),
                SimpleNamespace(toponyms=[]),
            ]
        )
    )

    result = runner.predictions_by_document(
        project,
        [101, 202],
        [document("article-a"), document("article-b")],
    )

    assert project.get_documents.call_args.args == ([101, 202],)
    assert result == {
        "article-a": [
            {
                "start": 0,
                "end": 6,
                "identifier": "feature-1",
                "document_id": "article-a",
                "latitude": 47.3769,
                "longitude": 8.5417,
            },
            {
                "start": 11,
                "end": 15,
                "identifier": None,
                "document_id": "article-a",
                "latitude": None,
                "longitude": None,
            },
        ],
        "article-b": [],
    }


def test_build_phase_models_constructs_only_the_recognizer(monkeypatch):
    recognizer = object()
    build_recognizer = Mock(return_value=recognizer)
    build_resolver = Mock()
    monkeypatch.setattr(runner.pipelines, "build_recognizer", build_recognizer)
    monkeypatch.setattr(runner.pipelines, "build_resolver", build_resolver)

    models = runner._build_phase_models(
        runner.RECOGNITION, "swapped", device="cpu", min_similarity=0.2
    )

    assert models == (recognizer, None)
    build_recognizer.assert_called_once_with("swapped", device="cpu")
    build_resolver.assert_not_called()


def test_build_phase_models_constructs_only_the_resolver(monkeypatch):
    resolver = object()
    build_recognizer = Mock()
    build_resolver = Mock(return_value=resolver)
    monkeypatch.setattr(runner.pipelines, "build_recognizer", build_recognizer)
    monkeypatch.setattr(runner.pipelines, "build_resolver", build_resolver)

    models = runner._build_phase_models(
        runner.RESOLUTION, "swapped", device="cpu", min_similarity=0.2
    )

    assert models == (None, resolver)
    build_recognizer.assert_not_called()
    build_resolver.assert_called_once_with("swapped", device="cpu", min_similarity=0.2)


def test_run_phase_skips_models_when_every_document_is_checkpointed(
    tmp_path, monkeypatch
):
    state = Checkpoint(identity())
    state.record(runner.RECOGNITION, "0", [])
    build_models = Mock()
    messages = []
    monkeypatch.setattr(runner, "_build_phase_models", build_models)

    names = runner.run_phase(
        runner.RECOGNITION,
        "swapped",
        [document()],
        state,
        tmp_path / "checkpoint.json",
        device="cpu",
        min_similarity=0.0,
        chunk_size=1,
        log=messages.append,
    )

    assert (names, build_models.call_count, messages) == (
        {},
        0,
        ["  swapped/recognition: already complete (1 documents)"],
    )


def test_run_phase_builds_models_and_runs_only_remaining_documents(
    tmp_path, monkeypatch
):
    state = Checkpoint(identity())
    state.record(runner.RECOGNITION, "0", [])
    model_pair = (object(), None)
    build_models = Mock(return_value=model_pair)
    model_names = Mock(return_value={"recognizer": "gliner-checkpoint"})
    run_chunks = Mock()
    clock = Mock(side_effect=[1.0, 2.0])
    messages = []
    monkeypatch.setattr(runner, "_build_phase_models", build_models)
    monkeypatch.setattr(runner.pipelines, "model_names", model_names)
    monkeypatch.setattr(runner, "_run_chunks", run_chunks)
    monkeypatch.setattr(runner.time, "perf_counter", clock)
    documents = [document("0"), document("1")]

    names = runner.run_phase(
        runner.RECOGNITION,
        "swapped",
        documents,
        state,
        tmp_path / "checkpoint.json",
        device="cpu",
        min_similarity=0.2,
        chunk_size=2,
        log=messages.append,
    )

    execution, remaining, models = run_chunks.call_args.args
    assert (names, execution.phase, remaining, models, state.elapsed_seconds) == (
        {"recognizer": "gliner-checkpoint"},
        runner.RECOGNITION,
        [documents[1]],
        model_pair,
        1.0,
    )
    assert messages == ["  swapped/recognition: 1 of 2 documents to do on cpu"]


def test_run_chunks_records_each_result_before_saving(tmp_path, monkeypatch):
    documents = [document("a"), document("b")]
    state = Checkpoint(identity())
    execution = runner._PhaseExecution(
        runner.RECOGNITION,
        documents,
        state,
        tmp_path / "checkpoint.json",
        "cpu",
        1,
        lambda message: None,
    )
    process = Mock(
        side_effect=lambda phase, chunk, **models: {
            chunk[0].identifier: [{"document_id": chunk[0].identifier}]
        }
    )
    save = Mock()
    clear_cuda = Mock()
    monkeypatch.setattr(runner, "_process_chunk", process)
    monkeypatch.setattr(ckpt, "save", save)
    monkeypatch.setattr(runner, "_empty_cuda_cache", clear_cuda)

    runner._run_chunks(execution, documents, (object(), None))

    assert state.recognition == {
        "a": [{"document_id": "a"}],
        "b": [{"document_id": "b"}],
    }
    assert (save.call_count, process.call_count, clear_cuda.call_args.args) == (
        2,
        2,
        ("cpu",),
    )


def test_empty_cuda_cache_does_nothing_for_cpu(monkeypatch):
    torch = ModuleType("torch")
    torch.__dict__["cuda"] = SimpleNamespace(empty_cache=Mock())
    monkeypatch.setitem(__import__("sys").modules, "torch", torch)

    runner._empty_cuda_cache("cpu")

    torch.cuda.empty_cache.assert_not_called()


def test_empty_cuda_cache_releases_cuda_memory(monkeypatch):
    torch = ModuleType("torch")
    torch.__dict__["cuda"] = SimpleNamespace(empty_cache=Mock())
    monkeypatch.setitem(__import__("sys").modules, "torch", torch)

    runner._empty_cuda_cache("cuda:0")

    torch.cuda.empty_cache.assert_called_once_with()


@pytest.mark.parametrize(
    ("phase", "expected_events", "expected_manual_call"),
    [
        (runner.RECOGNITION, ["create", "recognize", "delete"], None),
        (
            runner.RESOLUTION,
            ["create", "recognize", "resolve", "delete"],
            ("gold", ["Zurich"], [[(0, 6)]]),
        ),
    ],
)
def test_process_chunk_runs_the_requested_phase_and_deletes_project(
    monkeypatch, phase, expected_events, expected_manual_call
):
    import geoparser.project as project_module
    from geoparser.modules.recognizers import manual

    events = []
    projects = []

    class FakeProject:
        def __init__(self, name):
            self.name = name
            projects.append(self)

        def create_documents(self, texts):
            events.append(("create", texts))
            return ["database-document"]

        def run_recognizer(self, recognizer):
            events.append(("recognize", recognizer))

        def run_resolver(self, resolver):
            events.append(("resolve", resolver))

        def delete(self):
            events.append(("delete",))

    manual_recognizer = Mock(return_value="gold-recognizer")
    recognizer = object()
    resolver = object()
    predictions = {"doc": []}
    monkeypatch.setattr(project_module, "Project", FakeProject)
    monkeypatch.setattr(manual, "ManualRecognizer", manual_recognizer)
    monkeypatch.setattr(
        runner, "predictions_by_document", Mock(return_value=predictions)
    )

    result = runner._process_chunk(
        phase,
        [document("doc")],
        recognizer=recognizer,
        resolver=resolver,
    )

    assert (result, [event[0] for event in events], manual_recognizer.call_args) == (
        predictions,
        expected_events,
        None if expected_manual_call is None else call(*expected_manual_call),
    )
