"""
A guard that every ``# pragma: no mutate`` in the package is actually honoured.

mutmut recognizes these comments only in particular positions: trailing a
statement line, or on their own line before a statement as ``block`` /
between ``start`` and ``end``. A comment nested inside a call's argument list
or on a continuation line is silently ignored -- and ``ruff format`` moves
comments onto continuation lines whenever it wraps a long expression.

The failure is invisible: the mutants those lines were meant to exempt come
back as survivors, with a comment right beside them explaining why they were
not supposed to. This test asks mutmut itself which lines it will skip and
fails when a pragma is not among them.
"""

import ast
from pathlib import Path

import libcst as cst
import pytest
from libcst.metadata import MetadataWrapper
from mutmut.configuration import Config
from mutmut.mutation.pragma_handling import get_ignored_lines

HERE = Path(__file__).resolve()
PACKAGE = HERE.parents[3] / "geoparser"
MARKER = "pragma: no mutate"

# mutmut runs the suite from a rewritten copy of the tree under ``mutants/``,
# where each function has been expanded into numbered variants and a pragma
# region may straddle the split. Only the real source is worth checking, and
# checking the copy makes every mutation run fail on this test alone.
IN_MUTANT_TREE = "mutants" in HERE.parts

pytestmark = pytest.mark.skipif(
    IN_MUTANT_TREE, reason="the mutated copy of the tree is not the source"
)


def _pragma_files() -> list[Path]:
    """Every package file that carries at least one pragma."""
    return sorted(
        path for path in PACKAGE.rglob("*.py") if MARKER in path.read_text("utf-8")
    )


def _ignored(path: Path, source: str) -> set[int]:
    """The line numbers mutmut will not mutate in this file."""
    Config.ensure_loaded()
    wrapper = MetadataWrapper(cst.parse_module(source))
    return set(get_ignored_lines(str(path), source, wrapper).no_mutate_lines)


def _unhonoured_pragma_comments(path: Path) -> list[str]:
    """Describe pragma comments that mutmut does not actually honor."""
    source = path.read_text("utf-8")
    ignored = _ignored(path, source)
    return [
        f"{path.name}:{number} {line.strip()}"
        for number, line in enumerate(source.splitlines(), 1)
        if MARKER in line and number not in ignored
    ]


def _unhonoured_pragmas() -> list[str]:
    """Collect misleading mutation pragmas across the package."""
    findings = []
    for path in _pragma_files():
        findings.extend(_unhonoured_pragma_comments(path))
    return findings


def _class_definition(module: ast.Module, name: str) -> ast.ClassDef:
    return next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == name
    )


def _method_definition(class_node: ast.ClassDef, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _raises_named_message(node: ast.AST, exception_name: str) -> bool:
    if not isinstance(node, ast.Raise):
        return False
    if not isinstance(node.exc, ast.Call):
        return False
    return _calls_named_exception(node.exc, exception_name) and _passes_message_name(
        node.exc
    )


def _calls_named_exception(call: ast.Call, name: str) -> bool:
    return isinstance(call.func, ast.Name) and call.func.id == name


def _passes_message_name(call: ast.Call) -> bool:
    return (
        len(call.args) == 1
        and isinstance(call.args[0], ast.Name)
        and call.args[0].id == "msg"
    )


def _exception_message_lines(method: ast.FunctionDef, name: str) -> set[int]:
    lines = set()
    for node in ast.walk(method):
        if _raises_named_message(node, name):
            assert isinstance(node, ast.Raise)
            lines.add(node.lineno)
    return lines


def _region_state(line: str, inside: bool) -> tuple[bool, bool]:
    """Return the updated region state and whether a line is in the region."""
    if f"{MARKER} start" in line:
        return True, False
    if f"{MARKER} end" in line:
        return False, False
    return inside, inside


def _declared_pragma_lines(source: str) -> set[int]:
    """Find pragma comments and every line between explicit region markers."""
    declared = set()
    inside = False
    for number, line in enumerate(source.splitlines(), 1):
        inside, in_region = _region_state(line, inside)
        if in_region or MARKER in line:
            declared.add(number)
    return declared


def _stray_pragma_lines(path: Path, source: str) -> list[str]:
    """List mutmut exemptions with no trailing marker or enclosing region."""
    stray = sorted(_ignored(path, source) - _declared_pragma_lines(source))
    return [
        f"{path.name}:{number} {source.splitlines()[number - 1].strip()}"
        for number in stray
    ]


def _uncovered_region_lines(path: Path, source: str) -> list[str]:
    """List lines inside explicit pragma regions that mutmut still changes."""
    ignored = _ignored(path, source)
    inside = False
    uncovered = []
    for number, line in enumerate(source.splitlines(), 1):
        inside, in_region = _region_state(line, inside)
        if in_region and number not in ignored:
            uncovered.append(f"{path.name}:{number} {line.strip()}")
    return uncovered


@pytest.mark.unit
class TestPragmaPlacement:
    """Every pragma comment, and every line it claims to cover."""

    def test_the_package_uses_pragmas_at_all(self):
        """A sanity check: this guard is not passing because it found nothing."""
        # Act & Assert
        assert _pragma_files(), "expected some pragmas to guard"

    def test_every_pragma_comment_is_honoured(self):
        """No pragma sits somewhere mutmut ignores."""
        unhonoured = _unhonoured_pragmas()
        assert not unhonoured, "pragmas mutmut will not act on:\n" + "\n".join(
            unhonoured
        )

    def test_no_pragma_uses_the_unbounded_block_form(self):
        """
        ``block`` exempts to the end of the enclosing suite, which is a trap.

        It reads as "this statement", but mutmut applies it to every remaining
        statement in the block -- at module level, the rest of the file. A
        region that silently grows as code is added below it stops mutating
        code nobody meant to exempt, so the package uses explicit
        ``start``/``end`` regions and trailing pragmas instead.
        """
        # Act
        offenders = [
            f"{path.name}:{number}"
            for path in _pragma_files()
            for number, line in enumerate(path.read_text("utf-8").splitlines(), 1)
            if f"{MARKER} block" in line
        ]

        # Assert
        assert not offenders, (
            "use `# pragma: no mutate start` / `end`, or a trailing pragma, "
            "instead of the open-ended block form:\n" + "\n".join(offenders)
        )

    def test_import_validation_typeerror_messages_are_exempt_from_mutation(self):
        """Removing an optional exception message does not change behavior."""
        path = PACKAGE / "annotator/db/crud/session.py"
        source = path.read_text("utf-8")
        repository = _class_definition(ast.parse(source), "SessionRepository")
        method = _method_definition(repository, "_document_create_from_import")
        message_lines = _exception_message_lines(
            method,
            "TypeError",
        )

        assert len(message_lines) == 2
        assert message_lines <= _ignored(path, source)

    def test_no_line_is_exempt_without_a_marker_delimiting_it(self):
        """
        Every exempted line belongs to a region someone wrote deliberately.

        This is the check that catches an exemption widening by accident: a
        line mutmut skips that lies outside both a start/end region and a
        trailing pragma is one no comment in the file asked for.
        """
        # Act
        stray = []
        for path in _pragma_files():
            source = path.read_text("utf-8")
            stray.extend(_stray_pragma_lines(path, source))

        # Assert
        assert not stray, "lines exempt from mutation with no pragma around them:\n" + (
            "\n".join(stray)
        )

    def test_every_start_end_region_is_covered_end_to_end(self):
        """A start/end pair exempts every line between its markers."""
        # Act
        uncovered = []
        for path in _pragma_files():
            source = path.read_text("utf-8")
            uncovered.extend(_uncovered_region_lines(path, source))

        # Assert
        assert not uncovered, "lines inside a pragma region that still mutate:\n" + (
            "\n".join(uncovered)
        )
