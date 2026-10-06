"""Inline lint suppressions may only go down, never up."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SUPPRESSION = re.compile(r"#\s*noqa\b")

# Lower this when a suppression is removed. Raising it needs a reason in
# review: prefer fixing the code or one per-file-ignores entry in pyproject.toml.
MAX_INLINE_NOQA = 44


def _count_inline_noqa() -> int:
    sources = [
        path
        for directory in ("geoparser", "scripts")
        for path in (ROOT / directory).rglob("*.py")
    ]
    return sum(
        len(SUPPRESSION.findall(path.read_text(encoding="utf-8"))) for path in sources
    )


def test_inline_noqa_count_does_not_rise() -> None:
    assert _count_inline_noqa() <= MAX_INLINE_NOQA


def test_ratchet_is_tight() -> None:
    """A lower count must lower the ceiling so it keeps ratcheting."""
    assert _count_inline_noqa() == MAX_INLINE_NOQA
