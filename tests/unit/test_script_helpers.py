"""Byte-level contracts of the helpers shared by the benchmark scripts."""

from __future__ import annotations

from types import SimpleNamespace

from geoparser.evaluation import Annotation, toponym_annotation
from scripts._io import write_json_atomic


def test_write_json_atomic_compact_default(tmp_path):
    path = tmp_path / "nested" / "c.json"
    write_json_atomic(path, {"b": "é", "a": [1]})
    assert path.read_bytes() == b'{"b": "\\u00e9", "a": [1]}'
    assert list(path.parent.iterdir()) == [path]


def test_write_json_atomic_pretty_sorted(tmp_path):
    path = tmp_path / "c.json"
    write_json_atomic(
        path,
        {"b": "é", "a": 1},
        pretty=True,
        fsync=False,
    )
    assert path.read_bytes() == '{\n  "a": 1,\n  "b": "é"\n}\n'.encode()


def test_write_json_atomic_replaces_existing(tmp_path):
    path = tmp_path / "c.json"
    path.write_text("old")
    write_json_atomic(path, [1])
    assert path.read_text() == "[1]"


def _toponym(location):
    return SimpleNamespace(start=1, end=4, location=location)


def test_toponym_annotation_unresolved():
    assert toponym_annotation(_toponym(None)) == Annotation(1, 4)


def test_toponym_annotation_resolved_with_coordinates():
    location = SimpleNamespace(
        identifier="g:1", data={"latitude": "1.5", "longitude": 2}
    )
    assert toponym_annotation(_toponym(location), "d") == Annotation(1, 4, "g:1", "d")
    assert toponym_annotation(
        _toponym(location), "d", with_coordinates=True
    ) == Annotation(1, 4, "g:1", "d", 1.5, 2.0)


def test_toponym_annotation_unusable_coordinates():
    location = SimpleNamespace(identifier="g:1", data={"latitude": "x"})
    assert toponym_annotation(_toponym(location), with_coordinates=True) == Annotation(
        1, 4, "g:1"
    )
