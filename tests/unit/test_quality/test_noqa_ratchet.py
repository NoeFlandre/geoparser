"""Inline lint suppressions may only go down, never up."""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SUPPRESSION = re.compile(r"#\s*noqa\b", re.IGNORECASE)

# Lower this when a suppression is removed. Raising it needs a reason in
# review: prefer fixing the code or one per-file-ignores entry in pyproject.toml.
MAX_INLINE_NOQA = 44


def _generated_lines(tree: ast.AST) -> set[int]:
    """
    Lines inside functions the mutation runner generated.

    It copies every function once per mutant (``..__mutmut_N``) and keeps one
    more copy of the original (``..__mutmut_orig``), comments included. The
    public function stays in place, so skipping the copies counts each real
    suppression once, in the real tree and in the mutation runner's copy.
    """
    lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and "__mutmut_" in node.name:
            lines.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    return lines


def _file_count(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    generated = _generated_lines(ast.parse(text))
    return sum(
        1
        for number, line in enumerate(text.splitlines(), start=1)
        if number not in generated and SUPPRESSION.search(line)
    )


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
