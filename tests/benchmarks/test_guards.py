"""Fast algorithm and import checks that protect benchmarked code paths."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.benchmark


class _CountingText(str):
    comparisons = 0

    def __eq__(self, other):
        type(self).comparisons += 1
        return super().__eq__(other)

    __hash__ = str.__hash__


class _CountingSpan(tuple):
    comparisons = 0

    def __new__(cls, start, end):
        return super().__new__(cls, (start, end))

    def __eq__(self, other):
        type(self).comparisons += 1
        return super().__eq__(other)

    __hash__ = tuple.__hash__


def test_manual_recognizer_lookup_comparisons_scale_linearly():
    from geoparser.modules.recognizers.manual import ManualRecognizer

    size = 256
    texts = [_CountingText(f"Place {index:04d}") for index in range(size)]
    query_texts = [_CountingText(str(text)) for text in texts]
    stored_references = [[(0, 5)] for _ in texts]
    recognizer = ManualRecognizer("complexity", texts, stored_references)
    _CountingText.comparisons = 0

    predictions = recognizer.predict(query_texts)

    assert len(predictions) == size
    assert _CountingText.comparisons <= 4 * size


def test_manual_resolver_lookup_comparisons_scale_linearly():
    from geoparser.modules.resolvers.manual import ManualResolver

    size = 256
    texts = [_CountingText(f"Place {index:04d}") for index in range(size)]
    references = [[_CountingSpan(0, 5)] for _ in texts]
    query_texts = [_CountingText(str(text)) for text in texts]
    query_references = [[_CountingSpan(0, 5)] for _ in texts]
    referents = [[("benchmark", str(index))] for index in range(size)]
    resolver = ManualResolver("complexity", texts, references, referents)
    _CountingText.comparisons = 0
    _CountingSpan.comparisons = 0

    predictions = resolver.predict(query_texts, query_references)

    assert len(predictions) == size
    assert _CountingText.comparisons <= 4 * size
    assert _CountingSpan.comparisons <= 4 * size


def test_importing_geoparser_does_not_load_model_frameworks():
    repository = Path(__file__).resolve().parents[2]
    script = (
        "import sys; import geoparser; "
        "heavy = {'torch', 'transformers', 'spacy'}; "
        "loaded = sorted(name for name in sys.modules "
        "if name.partition('.')[0] in heavy); "
        "assert not loaded, loaded"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(repository), environment.get("PYTHONPATH", "")]
    )

    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    ("table", "column"),
    [
        ("document", "project_id"),
        ("recognition", "document_id"),
        ("recognition", "recognizer_id"),
        ("reference", "document_id"),
        ("reference", "recognizer_id"),
        ("referent", "reference_id"),
        ("referent", "resolver_id"),
        ("resolution", "reference_id"),
        ("resolution", "resolver_id"),
    ],
)
def test_foreign_key_lookup_has_indexed_query_plan(benchmark_database, table, column):
    from sqlalchemy import text

    with benchmark_database.connect() as connection:
        plan = connection.execute(
            text(f"EXPLAIN QUERY PLAN SELECT id FROM {table} WHERE {column} = :value"),
            {"value": "probe"},
        ).all()

    details = " ".join(row[-1] for row in plan)
    assert f"ix_{table}_{column}" in details
