from pathlib import Path
from typing import cast

import pytest

from tests.conftest import (
    _TESTS_ROOT,
    _cleanup_training_outputs,
    directory_marker_for,
    pytest_collection_modifyitems,
)


class _FakeItem:
    def __init__(self, path: Path, markers: tuple[str, ...] = ()) -> None:
        self.path = path
        self.markers = markers
        self.added: list[str] = []

    def get_closest_marker(self, name: str) -> str | None:
        return name if name in self.markers else None

    def add_marker(self, marker: str) -> None:
        self.added.append(marker)


def test_directory_marker_follows_the_top_level_test_directory() -> None:
    assert directory_marker_for(_TESTS_ROOT / "property" / "test_x.py") == "property"
    assert directory_marker_for(_TESTS_ROOT / "unit" / "test_a" / "t.py") == "unit"
    assert directory_marker_for(_TESTS_ROOT / "benchmarks" / "test_b.py") is None
    assert directory_marker_for(_TESTS_ROOT.parent / "geoparser" / "m.py") is None


def test_collection_marks_only_unmarked_tests_in_marked_directories() -> None:
    unmarked = _FakeItem(_TESTS_ROOT / "acceptance" / "test_network_guard.py")
    already_marked = _FakeItem(
        _TESTS_ROOT / "unit" / "test_x.py", markers=("integration",)
    )
    benchmark = _FakeItem(
        _TESTS_ROOT / "benchmarks" / "test_performance.py", markers=("benchmark",)
    )

    pytest_collection_modifyitems(
        cast("list[pytest.Item]", [unmarked, already_marked, benchmark])
    )

    assert unmarked.added == ["acceptance"]
    assert already_marked.added == []
    assert benchmark.added == []


def test_training_output_cleanup_removes_only_known_model_directories(
    tmp_path: Path,
) -> None:
    retained = tmp_path / "fixture-data"
    retained.mkdir()
    (retained / "input.txt").write_text("keep", encoding="utf-8")
    training_output = tmp_path / "trained_model"
    training_output.mkdir()
    (training_output / "checkpoint.bin").write_bytes(b"large model")

    _cleanup_training_outputs(tmp_path)

    assert retained.exists()
    assert training_output.exists() is False
