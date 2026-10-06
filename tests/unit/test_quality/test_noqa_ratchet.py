"""Inline lint suppressions may only go down, never up."""

import json
import re
from importlib.metadata import distribution
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

import pytest

_HERE = Path(__file__).resolve().parents[3]


def _source_root() -> Path:
    """
    The checkout whose sources are counted.

    The mutation runner executes the suite from a ``mutants/`` copy whose
    sources are instrumented: each function is duplicated per mutant and its
    body moved, so counting there would not match the real code. That copy
    may also be a symlink to an artifact directory, so the checkout is found
    from the editable install's recorded location, not from this file's path.
    """
    if _HERE.name != "mutants":
        return _HERE
    direct_url = distribution("geoparser").read_text("direct_url.json") or "{}"
    return Path(url2pathname(urlparse(json.loads(direct_url)["url"]).path))


ROOT = _source_root()
# Line-level ``# noqa`` and the file-level ``# ruff: noqa`` / ``# flake8: noqa``
# forms, in any letter case and with or without spaces after the ``#``.
SUPPRESSION = re.compile(r"#\s*(?:(?:ruff|flake8)\s*:\s*)?noqa\b", re.IGNORECASE)

# What Ruff lints: the package, the developer scripts and the demo notebook.
SCANNED_DIRECTORIES = ("geoparser", "scripts", "demo")
SCANNED_SUFFIXES = ("*.py", "*.ipynb")

# Lower this when a suppression is removed. Raising it needs a reason in
# review: prefer fixing the code or one per-file-ignores entry in pyproject.toml.
MAX_INLINE_NOQA = 44


def _file_count(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    return len(SUPPRESSION.findall(text))


def _inline_noqa_count() -> int:
    return sum(
        _file_count(path)
        for directory in SCANNED_DIRECTORIES
        for suffix in SCANNED_SUFFIXES
        for path in (ROOT / directory).rglob(suffix)
    )


def test_inline_noqa_count_does_not_rise() -> None:
    assert _inline_noqa_count() <= MAX_INLINE_NOQA


def test_ratchet_is_tight() -> None:
    """A lower count must lower the ceiling so it keeps ratcheting."""
    assert _inline_noqa_count() == MAX_INLINE_NOQA


SUPPRESSION_FORMS = [
    "import a  # noqa: F401",
    "import a  # NOQA: F401",
    "import a  # NoQa",
    "import a  #noqa",
    "import a  # noqa:F401",
    "import a  # noqa: F401, E402",
    "import a  # type: ignore  # noqa: F401",
    "# ruff: noqa: F401",
    "# RUFF:NOQA",
    "# flake8: noqa: F401",
    "#flake8:noqa",
]


@pytest.mark.parametrize("line", SUPPRESSION_FORMS)
def test_every_suppression_form_is_counted(line: str, tmp_path: Path) -> None:
    source = tmp_path / "sample.py"
    source.write_text(f"{line}\nx = 1\n", encoding="utf-8")
    assert _file_count(source) == 1


def test_code_without_a_suppression_is_not_counted(tmp_path: Path) -> None:
    source = tmp_path / "sample.py"
    source.write_text("x = 1  # a note\n", encoding="utf-8")
    assert _file_count(source) == 0


def test_notebook_suppressions_are_counted(tmp_path: Path) -> None:
    notebook = tmp_path / "sample.ipynb"
    notebook.write_text('{"source": ["import a  # noqa: F401\\n"]}', encoding="utf-8")
    assert _file_count(notebook) == 1
