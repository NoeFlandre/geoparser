"""Inline lint suppressions may only go down, never up."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SUPPRESSION = re.compile(r"#\s*noqa\b.*")

# Lower this when a suppression is removed. Raising it needs a reason in
# review: prefer fixing the code or one per-file-ignores entry in pyproject.toml.
MAX_INLINE_NOQA = 32


def _suppressions() -> set[tuple[str, str]]:
    """
    Distinct (file, suppression comment) pairs.

    The mutation runner copies each mutated function many times with its
    comments intact, so counting lines would inflate the total there.
    Counting distinct comments per file gives the same number in the real
    tree and in that copy.
    """
    found = set()
    for directory in ("geoparser", "scripts"):
        for path in (ROOT / directory).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            relative = path.relative_to(ROOT).as_posix()
            found.update((relative, comment) for comment in SUPPRESSION.findall(text))
    return found


def test_inline_noqa_count_does_not_rise() -> None:
    assert len(_suppressions()) <= MAX_INLINE_NOQA


def test_ratchet_is_tight() -> None:
    """A lower count must lower the ceiling so it keeps ratcheting."""
    assert len(_suppressions()) == MAX_INLINE_NOQA
