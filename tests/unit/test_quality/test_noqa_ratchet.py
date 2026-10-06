"""Inline lint suppressions may only go down, never up."""

import re
from pathlib import Path

_HERE = Path(__file__).resolve().parents[3]
# The mutation runner executes the suite from a ``mutants/`` copy whose
# sources are instrumented: each function is duplicated per mutant and its
# body moved, so counting there would not match the real code. Count the
# untouched originals one level up instead.
ROOT = _HERE.parent if _HERE.name == "mutants" else _HERE
SUPPRESSION = re.compile(r"#\s*noqa\b", re.IGNORECASE)

# Lower this when a suppression is removed. Raising it needs a reason in
# review: prefer fixing the code or one per-file-ignores entry in pyproject.toml.
MAX_INLINE_NOQA = 44


def _file_count(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    return len(SUPPRESSION.findall(text))


def _inline_noqa_count() -> int:
    return sum(
        _file_count(path)
        for directory in ("geoparser", "scripts")
        for path in (ROOT / directory).rglob("*.py")
    )


def test_inline_noqa_count_does_not_rise() -> None:
    assert _inline_noqa_count() <= MAX_INLINE_NOQA


def test_ratchet_is_tight() -> None:
    """A lower count must lower the ceiling so it keeps ratcheting."""
    assert _inline_noqa_count() == MAX_INLINE_NOQA


def test_suppressions_are_counted_in_any_letter_case(tmp_path: Path) -> None:
    source = tmp_path / "sample.py"
    source.write_text(
        "import a  # noqa: F401\nimport b  # NOQA: F401\nimport c  # NoQa\nx = 1\n",
        encoding="utf-8",
    )
    assert _file_count(source) == 3
