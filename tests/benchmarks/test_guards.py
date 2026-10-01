"""Fast algorithm and import checks that protect benchmarked code paths."""

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

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


_SIZE = 256
_RECOGNIZER_TEXTS = [_CountingText(f"Place {index:04d}") for index in range(_SIZE)]
_QUERY_TEXTS = [_CountingText(str(text)) for text in _RECOGNIZER_TEXTS]
_STORED_REFERENCES = [[(0, 5)] for _ in _RECOGNIZER_TEXTS]
_RESOLVER_REFERENCES = [[_CountingSpan(0, 5)] for _ in _RECOGNIZER_TEXTS]
_QUERY_REFERENCES = [[_CountingSpan(0, 5)] for _ in _RECOGNIZER_TEXTS]
_REFERENTS = [[("benchmark", str(index))] for index in range(_SIZE)]


def test_manual_recognizer_lookup_comparisons_scale_linearly():
    from geoparser.modules.recognizers.manual import ManualRecognizer

    recognizer = ManualRecognizer(
        "complexity", cast(Any, _RECOGNIZER_TEXTS), _STORED_REFERENCES
    )
    _CountingText.comparisons = 0

    predictions = recognizer.predict(cast(Any, _QUERY_TEXTS))

    assert len(predictions) == _SIZE
    assert _CountingText.comparisons <= 4 * _SIZE


def test_manual_resolver_lookup_comparisons_scale_linearly():
    from geoparser.modules.resolvers.manual import ManualResolver

    resolver = ManualResolver(
        "complexity",
        cast(Any, _RECOGNIZER_TEXTS),
        cast(Any, _RESOLVER_REFERENCES),
        _REFERENTS,
    )
    _CountingText.comparisons = 0
    _CountingSpan.comparisons = 0

    predictions = resolver.predict(
        cast(Any, _QUERY_TEXTS), cast(Any, _QUERY_REFERENCES)
    )

    assert len(predictions) == _SIZE
    assert _CountingText.comparisons <= 4 * _SIZE
    assert _CountingSpan.comparisons <= 4 * _SIZE


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


def test_benchmark_sessions_use_the_isolated_database(benchmark_database):
    from geoparser.db.db import get_engine

    assert get_engine() is benchmark_database


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


def test_benchmark_connections_keep_transaction_boundaries(benchmark_database):
    """A second session must neither see nor roll back a writer's pending row."""
    from sqlalchemy import text

    identifier = "0" * 32
    with benchmark_database.connect() as writer:
        writer.execute(
            text("INSERT INTO project (id, name) VALUES (:id, :name)"),
            {"id": identifier, "name": "transaction-boundary"},
        )
        with benchmark_database.connect() as reader:
            assert (
                reader.execute(text("SELECT count(*) FROM project")).scalar_one() == 0
            )
        writer.commit()
    with benchmark_database.connect() as reader:
        assert reader.execute(text("SELECT count(*) FROM project")).scalar_one() == 1
