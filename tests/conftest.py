"""
Root conftest.py for geoparser test suite.

This module provides pytest configuration and imports all fixtures
from the fixtures directory, making them available to all tests.
"""

import os
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
from hypothesis import settings as hypothesis_settings

# Import all fixtures from the fixtures directory
# This makes them available to all tests without explicit imports
pytest_plugins = [
    "tests.fixtures.db",
    "tests.fixtures.models",
    "tests.fixtures.modules",
    "tests.fixtures.gazetteer",
]

hypothesis_settings.register_profile(
    "ci",
    derandomize=True,
    database=None,
    deadline=None,
    max_examples=100,
    print_blob=True,
)
hypothesis_settings.register_profile("dev", max_examples=50, deadline=None)
hypothesis_settings.register_profile(
    "nightly", derandomize=False, max_examples=2000, deadline=None
)
hypothesis_settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_TRAINING_OUTPUT_NAMES = frozenset(
    {
        "initial_model",
        "model1",
        "model2",
        "trained_model",
        "trained_recognizer",
        "trained_resolver",
        "updated_model",
    }
)


def _cleanup_training_outputs(tmp_path: Path) -> None:
    """Remove large model outputs while retaining other test fixtures."""
    for path in tmp_path.iterdir():
        if path.name in _TRAINING_OUTPUT_NAMES and path.is_dir():
            shutil.rmtree(path)


@pytest.fixture(autouse=True)
def cleanup_training_outputs(tmp_path: Path) -> Iterator[None]:
    """Keep per-test model checkpoints from accumulating on disk."""
    yield
    _cleanup_training_outputs(tmp_path)


# Test directories whose items receive a marker named after the directory. Marker
# selection (for example `pytest -m property`) then covers every test in them.
# Markers are registered only in pyproject.toml.
_DIRECTORY_MARKERS = ("unit", "integration", "e2e", "property", "acceptance")
_TESTS_ROOT = Path(__file__).resolve().parent


def directory_marker_for(path: Path) -> str | None:
    """Return the directory marker for a test file, or None outside those directories."""
    resolved = path.resolve()
    if not resolved.is_relative_to(_TESTS_ROOT):
        return None
    top = resolved.relative_to(_TESTS_ROOT).parts[0]
    return top if top in _DIRECTORY_MARKERS else None


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark each test by its directory unless it already has a directory marker."""
    for item in items:
        marker = directory_marker_for(Path(item.path))
        if marker is None:
            continue
        if any(item.get_closest_marker(name) for name in _DIRECTORY_MARKERS):
            continue
        item.add_marker(marker)
