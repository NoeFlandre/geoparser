"""The pin, the blob-ID check, and the refusal to import an unverified checkout."""

import re
import sys
import types
import typing as t
from pathlib import Path

import pytest

from scripts.unitoprank_benchmark import pins
from scripts.unitoprank_benchmark.pins import (
    COMMIT,
    LICENSE_SPDX,
    REPOSITORY,
    REVIEWED_BLOBS,
    UpstreamMismatchError,
    git_blob_id,
    load_rank_toponyms,
    verify_checkout,
)

HELLO_BLOB = "ce013625030ba8dba906f756967f9e9ca394464a"
CHANGED_BLOB = "5ea2ed416fbd4a4cbe227b75fe255dd7fa6bd4d6"
EMPTY_BLOB = "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"


def test_the_pin_names_one_commit_under_its_licence():
    assert re.fullmatch(r"[0-9a-f]{40}", COMMIT)
    assert REPOSITORY == "https://gitlab.com/dlr-dw/UniTopRank"
    assert LICENSE_SPDX == "Apache-2.0"


def test_every_pinned_file_has_a_git_blob_id_and_the_import_chain_is_covered():
    assert len(REVIEWED_BLOBS) == 15
    assert all(re.fullmatch(r"[0-9a-f]{40}", blob) for blob in REVIEWED_BLOBS.values())
    assert {
        "LICENSE",
        "unitorank/ranker.py",
        "unitorank/__init__.py",
    } <= REVIEWED_BLOBS.keys()
    assert "thread_weight_rank_algorithm_3_beam.py" in REVIEWED_BLOBS


def test_git_blob_id_matches_what_git_hash_object_reports():
    assert git_blob_id(b"") == EMPTY_BLOB
    assert git_blob_id(b"hello\n") == HELLO_BLOB


def test_a_matching_tree_has_no_problems(tmp_path: Path):
    (tmp_path / "a.txt").write_bytes(b"hello\n")
    assert verify_checkout(tmp_path, {"a.txt": HELLO_BLOB}) == []


def test_missing_and_changed_files_are_each_reported(tmp_path: Path):
    (tmp_path / "a.txt").write_bytes(b"hello\n")
    (tmp_path / "b.txt").write_bytes(b"changed\n")
    problems = verify_checkout(
        tmp_path, {"a.txt": HELLO_BLOB, "b.txt": HELLO_BLOB, "c.txt": EMPTY_BLOB}
    )
    assert problems == [
        f"changed: b.txt is {CHANGED_BLOB}, pinned {HELLO_BLOB}",
        "missing: c.txt",
    ]


def test_an_empty_directory_fails_on_every_pinned_file(tmp_path: Path):
    problems = verify_checkout(tmp_path)
    assert len(problems) == 15
    assert problems[0] == "missing: LICENSE"


def test_load_refuses_a_checkout_that_does_not_match(tmp_path: Path):
    with pytest.raises(UpstreamMismatchError, match="missing: LICENSE"):
        load_rank_toponyms(tmp_path)


@pytest.fixture
def clean_upstream_modules(monkeypatch: pytest.MonkeyPatch):
    """Keep any fake unitorank package out of the other tests in the session."""
    for name in [
        key for key in sys.modules if key == "unitorank" or key.startswith("unitorank.")
    ]:
        monkeypatch.delitem(sys.modules, name)
    original_path = list(sys.path)
    yield
    for name in [
        key for key in sys.modules if key == "unitorank" or key.startswith("unitorank.")
    ]:
        sys.modules.pop(name, None)
    sys.path[:] = original_path


def _fake_checkout(root: Path) -> Path:
    package = root / "unitorank"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "ranker.py").write_text(
        "class RankerConfig:\n    pass\n\ndef rank_toponyms(**kwargs):\n    return {}\n"
    )
    return root


def test_load_returns_the_ranker_from_a_verified_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_upstream_modules
):
    checkout = _fake_checkout(tmp_path)
    monkeypatch.setattr(pins, "verify_checkout", lambda _path: [])
    rank_toponyms, ranker_config = load_rank_toponyms(checkout)
    assert (
        rank_toponyms(text="", toponyms=[], candidates_by_toponym={}, config=None) == {}
    )
    assert ranker_config.__name__ == "RankerConfig"


def test_load_refuses_a_unitorank_imported_from_somewhere_else(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_upstream_modules
):
    checkout = _fake_checkout(tmp_path / "verified")
    elsewhere = t.cast(t.Any, types.ModuleType("unitorank"))
    elsewhere.__file__ = str(tmp_path / "elsewhere" / "unitorank" / "__init__.py")
    ranker = t.cast(t.Any, types.ModuleType("unitorank.ranker"))
    ranker.rank_toponyms = lambda **kwargs: {}
    ranker.RankerConfig = object
    monkeypatch.setitem(sys.modules, "unitorank", elsewhere)
    monkeypatch.setitem(sys.modules, "unitorank.ranker", ranker)
    monkeypatch.setattr(pins, "verify_checkout", lambda _path: [])
    with pytest.raises(UpstreamMismatchError, match="imported from"):
        load_rank_toponyms(checkout)
