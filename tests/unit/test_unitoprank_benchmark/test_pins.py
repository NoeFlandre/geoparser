"""The pin, the blob-ID check, and the refusal to import an unverified checkout."""

import importlib.machinery
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
    assert len(REVIEWED_BLOBS) == 16
    assert all(re.fullmatch(r"[0-9a-f]{40}", blob) for blob in REVIEWED_BLOBS.values())
    assert {
        "LICENSE",
        "__init__.py",
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


@pytest.mark.parametrize(
    "extra",
    [
        "rapidfuzz.py",
        # The extension suffix this interpreter imports: a CPython Linux suffix is not importable on Windows.
        "rapidfuzz" + importlib.machinery.EXTENSION_SUFFIXES[0],
        "__pycache__/ranker.cpython-312.pyc",
        "unitorank/extra.py",
    ],
)
def test_an_importable_file_the_pin_does_not_name_is_refused(
    tmp_path: Path, extra: str
):
    (tmp_path / "a.txt").write_bytes(b"hello\n")
    (tmp_path / extra).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / extra).write_bytes(b"raise SystemExit('ran')\n")
    problems = verify_checkout(tmp_path, {"a.txt": HELLO_BLOB})
    assert problems == [f"unreviewed: {extra}"]


def test_files_that_cannot_be_imported_and_git_metadata_are_not_refused(tmp_path: Path):
    (tmp_path / "a.txt").write_bytes(b"hello\n")
    (tmp_path / "README.md").write_bytes(b"not code\n")
    (tmp_path / ".git" / "hooks").mkdir(parents=True)
    (tmp_path / ".git" / "hooks" / "pre-commit.py").write_bytes(b"\n")
    assert verify_checkout(tmp_path, {"a.txt": HELLO_BLOB}) == []


def test_an_empty_directory_fails_on_every_pinned_file(tmp_path: Path):
    problems = verify_checkout(tmp_path)
    assert len(problems) == 16
    assert problems[0] == "missing: LICENSE"


def test_load_refuses_a_checkout_that_does_not_match(tmp_path: Path):
    with pytest.raises(UpstreamMismatchError, match="missing: LICENSE"):
        load_rank_toponyms(tmp_path)


def test_load_refuses_an_extra_module_before_any_of_it_runs(
    tmp_path: Path, clean_upstream_modules
):
    """An extra rapidfuzz.py is first on the import path, so it must not run."""
    checkout = _fake_checkout(tmp_path / "verified")
    marker = tmp_path / "rapidfuzz-ran"
    (checkout / "rapidfuzz.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\n"
    )
    with pytest.raises(UpstreamMismatchError, match=r"unreviewed: rapidfuzz\.py"):
        load_rank_toponyms(checkout)
    assert not marker.exists()


def _unitorank_module_names() -> list[str]:
    """Return the name of every unitorank module currently imported."""
    return [
        key for key in sys.modules if key == "unitorank" or key.startswith("unitorank.")
    ]


@pytest.fixture
def clean_upstream_modules():
    """Keep any fake unitorank package out of the other tests in the session."""
    saved = {name: sys.modules.pop(name) for name in _unitorank_module_names()}
    original_path = list(sys.path)
    yield
    for name in _unitorank_module_names():
        sys.modules.pop(name, None)
    sys.modules.update(saved)
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
    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    rank_toponyms, ranker_config = load_rank_toponyms(checkout)
    assert (
        rank_toponyms(text="", toponyms=[], candidates_by_toponym={}, config=None) == {}
    )
    assert ranker_config.__name__ == "RankerConfig"
    # Importing must not leave a __pycache__ that the next verification refuses.
    assert not (checkout / "unitorank" / "__pycache__").exists()
    assert sys.dont_write_bytecode is False


def test_load_refuses_a_unitorank_imported_from_somewhere_else(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_upstream_modules
):
    checkout = _fake_checkout(tmp_path / "verified")
    elsewhere = t.cast(t.Any, types.ModuleType("unitorank"))
    elsewhere.__file__ = str(tmp_path / "elsewhere" / "unitorank" / "__init__.py")
    ranker = t.cast(t.Any, types.ModuleType("unitorank.ranker"))
    ranker.rank_toponyms = lambda **kwargs: {}
    ranker.RankerConfig = object
    # The clean_upstream_modules fixture removes these again at teardown.
    sys.modules["unitorank"] = elsewhere
    sys.modules["unitorank.ranker"] = ranker
    monkeypatch.setattr(pins, "verify_checkout", lambda _path: [])
    with pytest.raises(UpstreamMismatchError, match="imported from"):
        load_rank_toponyms(checkout)


def test_a_foreign_unitorank_earlier_on_the_path_never_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_upstream_modules
):
    """A copy earlier on sys.path must not run: the verified checkout is moved first."""
    checkout = _fake_checkout(tmp_path / "verified")
    foreign = _fake_checkout(tmp_path / "foreign")
    marker = tmp_path / "foreign-code-ran"
    (foreign / "unitorank" / "__init__.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\n"
    )
    monkeypatch.syspath_prepend(str(checkout.resolve()))
    monkeypatch.syspath_prepend(str(foreign.resolve()))
    monkeypatch.setattr(pins, "verify_checkout", lambda _path: [])
    load_rank_toponyms(checkout)
    assert not marker.exists()
    assert sys.path[0] == str(checkout.resolve())
    origin = sys.modules["unitorank"].__file__
    assert origin is not None
    assert origin.startswith(str(checkout.resolve()))


def test_a_checkout_behind_another_directory_is_moved_to_the_front(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_upstream_modules
):
    """An absolute import in the pinned code must resolve from the checkout."""
    checkout = _fake_checkout(tmp_path / "verified")
    (checkout / "thread_weight_rank_algorithm_3_beam.py").write_text(
        "SOURCE = 'verified'\n"
    )
    (checkout / "unitorank" / "ranker.py").write_text(
        "from thread_weight_rank_algorithm_3_beam import SOURCE\n"
        "class RankerConfig:\n    pass\n"
        "def rank_toponyms(**kwargs):\n    return {'source': SOURCE}\n"
    )
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "thread_weight_rank_algorithm_3_beam.py").write_text(
        "SOURCE = 'foreign'\n"
    )
    # The checkout is already on the path, but behind the foreign directory.
    monkeypatch.syspath_prepend(str(checkout.resolve()))
    monkeypatch.syspath_prepend(str(foreign.resolve()))
    monkeypatch.setattr(pins, "verify_checkout", lambda _path: [])
    try:
        rank_toponyms, _ = load_rank_toponyms(checkout)
        assert rank_toponyms() == {"source": "verified"}
        assert sys.path.count(str(checkout.resolve())) == 1
        assert sys.path[0] == str(checkout.resolve())
    finally:
        sys.modules.pop("thread_weight_rank_algorithm_3_beam", None)


@pytest.mark.skipif(sys.platform == "win32", reason="creating symlinks needs rights")
def test_a_symlinked_directory_in_the_checkout_is_refused(tmp_path):
    outside = tmp_path / "outside" / "rapidfuzz"
    outside.mkdir(parents=True)
    (outside / "__init__.py").write_text("RAISED = True\n", encoding="utf-8")
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "rapidfuzz").symlink_to(outside, target_is_directory=True)

    problems = verify_checkout(checkout, blobs={})

    assert "symlink: rapidfuzz" in problems
