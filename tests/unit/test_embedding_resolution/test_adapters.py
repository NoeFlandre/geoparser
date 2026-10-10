"""Batching, shape, normalization and empty-input behaviour of the encoding adapters."""

import dataclasses
import math
from typing import Any, ClassVar

import huggingface_hub
import numpy as np
import pytest
import sentence_transformers

from scripts.embedding_resolution.adapters import (
    EmbeddingAdapter,
    SentenceTransformerEncoder,
)
from scripts.embedding_resolution.models import GEO_MINILM, JINA_V5_TEXT_SMALL


def small_model(**overrides):
    """A three-dimensional stand-in that keeps the real model's pin and key."""
    return dataclasses.replace(GEO_MINILM, dimension=3, **overrides)


class RecordingEncoder:
    """Returns [len(text), 3, 4] per string and records every batch it sees."""

    def __init__(self, rows=None):
        self.calls = []
        self.batch_sizes = []
        self.rows = rows

    def encode(self, texts, *, prompt, batch_size):
        self.calls.append((list(texts), prompt))
        self.batch_sizes.append(batch_size)
        if self.rows is not None:
            return self.rows(texts)
        return np.array([[len(text), 3.0, 4.0] for text in texts])


def test_the_configured_batch_size_reaches_every_encode_call():
    encoder = RecordingEncoder()
    adapter = EmbeddingAdapter(small_model(), encoder, batch_size=64)

    adapter.encode_documents(["a", "bb", "ccc"])

    assert encoder.batch_sizes == [64]


def test_batches_are_split_in_order_and_sized_by_batch_size():
    encoder = RecordingEncoder()
    adapter = EmbeddingAdapter(small_model(), encoder, batch_size=2)

    vectors = adapter.encode_documents(["a", "bb", "ccc", "dddd", "eeeee"])

    assert [len(texts) for texts, _ in encoder.calls] == [2, 2, 1]
    assert [text for texts, _ in encoder.calls for text in texts] == [
        "a",
        "bb",
        "ccc",
        "dddd",
        "eeeee",
    ]
    assert vectors.shape == (5, 3)


def test_each_row_is_normalized_to_unit_length_in_the_same_direction():
    adapter = EmbeddingAdapter(small_model(), RecordingEncoder(), batch_size=8)

    vectors = adapter.encode_queries(["ab"])

    expected = np.array([2.0, 3.0, 4.0]) / math.sqrt(29.0)
    assert np.allclose(vectors[0], expected)
    assert np.isclose(np.linalg.norm(vectors[0]), 1.0)


def test_normalization_is_skipped_when_the_model_does_not_declare_it():
    adapter = EmbeddingAdapter(
        small_model(normalize=False), RecordingEncoder(), batch_size=8
    )

    assert adapter.encode_queries(["ab"]).tolist() == [[2.0, 3.0, 4.0]]


def test_queries_and_documents_are_sent_under_their_own_prompts():
    model = small_model(query_prompt="Query: ", document_prompt="Document: ")
    encoder = RecordingEncoder()
    adapter = EmbeddingAdapter(model, encoder, batch_size=8)

    adapter.encode_queries(["context"])
    adapter.encode_documents(["candidate"])

    assert [prompt for _, prompt in encoder.calls] == ["Query: ", "Document: "]


def test_a_model_without_prompts_sends_an_empty_prompt():
    encoder = RecordingEncoder()
    EmbeddingAdapter(small_model(), encoder, batch_size=8).encode_documents(["x"])

    assert encoder.calls == [(["x"], "")]


def test_empty_input_returns_zero_rows_at_the_pinned_dimension_without_encoding():
    encoder = RecordingEncoder()
    vectors = EmbeddingAdapter(small_model(), encoder, batch_size=4).encode_documents(
        []
    )

    assert vectors.shape == (0, 3)
    assert encoder.calls == []


def test_wrong_output_shape_is_reported_with_both_shapes():
    encoder = RecordingEncoder(rows=lambda texts: np.ones((len(texts), 2)))
    adapter = EmbeddingAdapter(small_model(), encoder, batch_size=8)

    with pytest.raises(ValueError, match=r"shape \(2, 2\), expected \(2, 3\)"):
        adapter.encode_queries(["a", "b"])


def test_a_batch_with_too_few_rows_is_rejected():
    encoder = RecordingEncoder(rows=lambda texts: np.ones((len(texts) - 1, 3)))
    adapter = EmbeddingAdapter(small_model(), encoder, batch_size=8)

    with pytest.raises(ValueError, match="expected"):
        adapter.encode_documents(["a", "b"])


def test_non_finite_values_are_refused():
    encoder = RecordingEncoder(rows=lambda texts: np.full((len(texts), 3), np.nan))
    adapter = EmbeddingAdapter(small_model(), encoder, batch_size=8)

    with pytest.raises(ValueError, match="NaN or infinite"):
        adapter.encode_documents(["a"])


def test_a_zero_length_embedding_is_refused_before_normalization():
    encoder = RecordingEncoder(rows=lambda texts: np.zeros((len(texts), 3)))
    adapter = EmbeddingAdapter(small_model(), encoder, batch_size=8)

    with pytest.raises(ValueError, match="zero-length embedding"):
        adapter.encode_documents(["a"])


def test_non_string_inputs_are_refused_before_any_encoding():
    encoder = RecordingEncoder()
    adapter = EmbeddingAdapter(small_model(), encoder, batch_size=8)

    # Typed as Any: the input is wrong on purpose, and the adapter must refuse it.
    mixed: list[Any] = ["a", 3]
    with pytest.raises(TypeError, match="must be strings, got int"):
        adapter.encode_documents(mixed)
    assert encoder.calls == []


def test_batch_size_below_one_is_rejected_at_construction():
    with pytest.raises(ValueError, match="batch_size must be at least 1"):
        EmbeddingAdapter(small_model(), RecordingEncoder(), batch_size=0)


class FakeSentenceTransformer:
    """Records how it was constructed and how each batch was requested."""

    instances: ClassVar[list] = []

    def __init__(self, repository, **kwargs):
        self.repository = repository
        self.kwargs = kwargs
        self.max_seq_length = None
        self.encode_calls = []
        FakeSentenceTransformer.instances.append(self)

    def encode(self, texts, **options):
        self.encode_calls.append(options)
        return np.ones((len(texts), 1024))


def hub_snapshot(root, repository, revision):
    """The path the Hub cache uses for one commit of one repository."""
    name = f"models--{repository.replace('/', '--')}"
    return root / "hub" / name / "snapshots" / revision


@pytest.fixture
def fake_hub(monkeypatch, tmp_path):
    """Serve an empty local snapshot directory per request, laid out as the Hub cache."""
    downloads = []

    def snapshot_download(*, repo_id, revision):
        downloads.append((repo_id, revision))
        snapshot = hub_snapshot(tmp_path, repo_id, revision)
        snapshot.mkdir(parents=True, exist_ok=True)
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    return downloads


@pytest.fixture
def fake_transformer(monkeypatch, fake_hub):
    FakeSentenceTransformer.instances = []
    monkeypatch.setattr(
        sentence_transformers, "SentenceTransformer", FakeSentenceTransformer
    )
    return FakeSentenceTransformer


def test_the_real_backend_loads_the_pinned_commit_only_from_its_local_snapshot(
    fake_transformer, fake_hub, tmp_path
):
    SentenceTransformerEncoder(JINA_V5_TEXT_SMALL)

    assert fake_hub == [
        (JINA_V5_TEXT_SMALL.repository, JINA_V5_TEXT_SMALL.revision),
    ]
    (loaded,) = fake_transformer.instances
    assert loaded.repository == str(
        hub_snapshot(
            tmp_path, JINA_V5_TEXT_SMALL.repository, JINA_V5_TEXT_SMALL.revision
        )
    )
    assert loaded.kwargs == {"device": "cpu", "trust_remote_code": True}


def test_the_real_backend_keeps_the_pinned_model_and_sequence_length(fake_transformer):
    encoder = SentenceTransformerEncoder(JINA_V5_TEXT_SMALL)

    (loaded,) = fake_transformer.instances
    assert loaded.max_seq_length == JINA_V5_TEXT_SMALL.max_seq_length
    assert encoder.model is JINA_V5_TEXT_SMALL


def test_a_snapshot_of_another_commit_is_refused_before_any_model_loads(
    monkeypatch, tmp_path, fake_transformer
):
    main = tmp_path / "snapshots" / "main"
    main.mkdir(parents=True)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **_: str(main))

    with pytest.raises(ValueError, match="not a directory of the pinned commit"):
        SentenceTransformerEncoder(JINA_V5_TEXT_SMALL)
    assert fake_transformer.instances == []


def test_a_snapshot_path_that_is_a_file_is_refused_before_any_model_loads(
    monkeypatch, tmp_path, fake_transformer
):
    file = tmp_path / JINA_V5_TEXT_SMALL.revision
    file.write_text("not a snapshot")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **_: str(file))

    with pytest.raises(ValueError, match="not a directory of the pinned commit"):
        SentenceTransformerEncoder(JINA_V5_TEXT_SMALL)
    assert fake_transformer.instances == []


def test_the_real_backend_passes_the_documented_task_and_prompt(fake_transformer):
    encoder = SentenceTransformerEncoder(JINA_V5_TEXT_SMALL)

    encoder.encode(["a", "b"], prompt="Query: ", batch_size=64)

    (options,) = fake_transformer.instances[0].encode_calls
    assert options["task"] == "retrieval"
    assert options["prompt"] == "Query: "
    assert options["batch_size"] == 64
    assert options["normalize_embeddings"] is False


def test_the_real_backend_omits_prompt_and_task_when_the_model_has_none(
    fake_transformer,
):
    encoder = SentenceTransformerEncoder(GEO_MINILM)

    encoder.encode(["a"], prompt="", batch_size=8)

    (options,) = fake_transformer.instances[0].encode_calls
    assert "prompt" not in options
    assert "task" not in options
    assert fake_transformer.instances[0].kwargs["trust_remote_code"] is False
