"""Opt-in performance checks over synthetic data only."""

from itertools import count
from typing import Any, cast

import pytest

pytestmark = pytest.mark.benchmark

BENCHMARK_ROUNDS = 5


def _measure(benchmark, function):
    return benchmark.pedantic(function, rounds=BENCHMARK_ROUNDS, iterations=1)


@pytest.mark.parametrize("method", ["exact", "phrase", "partial", "fuzzy"])
def test_gazetteer_search_methods(benchmark, synthetic_gazetteer, method):
    """Search methods use the real read-only artifact query layer."""
    query = "Place 01234"
    results = _measure(
        benchmark,
        lambda: synthetic_gazetteer.search(query, method=method, limit=100, tiers=2),
    )

    assert isinstance(results, list)


def test_project_create_documents_5000(benchmark, benchmark_project):
    texts = [f"Place {index:05d}" for index in range(5_000)]
    ids = _measure(benchmark, lambda: benchmark_project.create_documents(texts))

    assert len(ids) == len(texts)


def test_project_create_references_5000(benchmark, benchmark_project_with_documents):
    project, texts, references, _ = benchmark_project_with_documents
    tags = count()

    def create_references():
        project.create_references(texts, references, f"bench-rec-{next(tags)}")

    _measure(benchmark, create_references)


def test_project_create_referents_5000(
    benchmark, benchmark_project_with_documents, synthetic_gazetteer
):
    project, texts, references, _ = benchmark_project_with_documents
    project.create_references(texts, references, "bench-seed")
    referents = [[("benchmark", "ID00000")] for _ in texts]
    tags = count()

    def create_referents():
        project.create_referents(
            texts, references, referents, f"bench-res-{next(tags)}"
        )

    _measure(benchmark, create_referents)


def test_project_get_documents_5000(benchmark, benchmark_project_with_documents):
    project, _, _, document_ids = benchmark_project_with_documents
    documents = _measure(benchmark, project.get_documents)

    assert len(document_ids) == len(documents) == 5_000


def test_small_project_delete_beside_large_project(benchmark, benchmark_database):
    from uuid import uuid4

    from geoparser.project.project import Project

    large_project = Project(f"large-{uuid4().hex}")
    large_project.create_documents(
        [f"Large document {index:05d}" for index in range(5_000)]
    )
    small_projects = []

    def setup_small_project():
        project = Project(f"small-{uuid4().hex}")
        project.create_documents(["Small project document"])
        small_projects.append(project)

    def delete_small_project():
        small_projects.pop().delete()

    benchmark.pedantic(
        delete_small_project,
        setup=setup_small_project,
        rounds=BENCHMARK_ROUNDS,
        iterations=1,
    )


def test_manual_recognizer_predict_5000(benchmark):
    from geoparser.modules.recognizers.manual import ManualRecognizer

    texts = [f"Place {index:05d}" for index in range(5_000)]
    references = [[(0, 5)] for _ in texts]
    recognizer = ManualRecognizer("benchmark", texts, references)

    predictions = _measure(benchmark, lambda: recognizer.predict(texts))

    assert len(predictions) == len(texts)


def test_manual_resolver_predict_5000(benchmark):
    from geoparser.modules.resolvers.manual import ManualResolver

    texts = [f"Place {index:05d}" for index in range(5_000)]
    references = [[(0, 5)] for _ in texts]
    referents = [[("benchmark", f"ID{index:05d}")] for index in range(5_000)]
    resolver = ManualResolver("benchmark", texts, references, referents)

    predictions = _measure(benchmark, lambda: resolver.predict(texts, references))

    assert len(predictions) == len(texts)


def test_sentence_transformer_candidate_gathering_with_stub_encoder(
    benchmark, synthetic_gazetteer
):
    from geoparser.modules.resolvers.sentencetransformer import (
        SentenceTransformerResolver,
    )

    class StubEncoder:
        def encode(self, sentences, **kwargs):
            return [[1.0, 0.0] for _ in sentences]

    texts = [f"Place {index:05d}" for index in range(1_000)]
    references = [[(0, len(text))] for text in texts]
    resolver = object.__new__(SentenceTransformerResolver)
    resolver.gazetteer = synthetic_gazetteer
    resolver.transformer = cast(Any, StubEncoder())
    resolver.candidate_search_cache = {}

    def gather():
        candidates = [[[]] for _ in texts]
        results = [[None] for _ in texts]
        resolver.candidate_search_cache.clear()
        resolver._gather_candidates(
            texts, references, candidates, results, "exact", tiers=1
        )
        return sum(len(document[0]) for document in candidates)

    result_count = _measure(benchmark, gather)

    assert result_count == len(texts)
