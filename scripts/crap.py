"""
CRAP score gate.

CRAP (Change Risk Anti-Patterns) combines how branchy a function is with how
well it is tested::

    CRAP = complexity^2 * (1 - coverage)^3 + complexity

A fully covered function scores its own cyclomatic complexity, so the gate is
simultaneously a complexity ceiling for tested code and a much harsher one for
untested code. Run it after pytest has written a coverage data file.

    uv run python scripts/crap.py --max-crap 30

The supplied maximum is an exclusive upper bound: a score equal to the
threshold also fails. Comparisons use the calculated Python float directly,
with no tolerance that could let a boundary score through.
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

from coverage import Coverage
from coverage.exceptions import NoSource
from radon.complexity import cc_visit

# These are all first-party Python source trees. The collector deliberately has
# no omit option: uncovered helpers and tests remain visible to the gate.
SOURCE_DIRECTORIES = ("geoparser", "scripts", "tests")


@dataclass(frozen=True)
class Score:
    """A single function's complexity, coverage and resulting CRAP score."""

    path: str
    name: str
    lineno: int
    complexity: int
    coverage: float

    @property
    def crap(self) -> float:
        """The CRAP score for this function."""
        uncovered = 1.0 - self.coverage
        return self.complexity**2 * uncovered**3 + self.complexity

    def __str__(self) -> str:
        return (
            f"{self.crap:7.2f}  cc={self.complexity:<3} "
            f"cov={self.coverage * 100:5.1f}%  {self.path}:{self.lineno} {self.name}"
        )


def _analyse(coverage: Coverage, path: Path) -> tuple[set[int], set[int]] | None:
    """
    Return the (statements, missing) line numbers for a file.

    Args:
        coverage: A loaded coverage session
        path: The file to analyse

    Returns:
        A tuple of statement and missing line numbers, or None when the file
        has no coverage data at all.
    """
    try:
        _, statements, _, missing, _ = coverage.analysis2(str(path))
    except NoSource:
        return None
    return set(statements), set(missing)


def score_file(coverage: Coverage, path: Path, root: Path) -> list[Score]:
    """
    Score every function and method in one source file.

    Args:
        coverage: A loaded coverage session
        path: The file to score
        root: Repository root, used to render relative paths

    Returns:
        One Score per function, ordered as they appear in the file.
    """
    analysed = _analyse(coverage, path)
    if analysed is None:
        return []
    statements, missing = analysed

    relative = path.relative_to(root).as_posix()
    source = path.read_text(encoding="utf-8")
    functions = _functions(source)
    owners = _statement_owners(statements, functions)
    return [
        Score(
            path=relative,
            name=function.name,
            lineno=function.lineno,
            complexity=function.complexity,
            coverage=_function_coverage(index, owners, missing),
        )
        for index, function in enumerate(functions)
    ]


def _statement_owners(
    statements: set[int], functions: list[_Function]
) -> dict[int, int]:
    """Assign each statement to the innermost function that contains it."""
    # Give every executable statement to its innermost function. In particular,
    # lines in a nested function must not inflate its parent's coverage.
    owners: dict[int, int] = {}
    for line in statements:
        candidates = [
            index
            for index, function in enumerate(functions)
            if function.body_start <= line <= function.endline
        ]
        if candidates:
            owners[line] = min(
                candidates,
                key=lambda index: (
                    functions[index].endline - functions[index].body_start,
                    -functions[index].depth,
                ),
            )
    return owners


def _function_coverage(index: int, owners: dict[int, int], missing: set[int]) -> float:
    """Calculate a function's owned statement coverage."""
    owned = {line for line, owner in owners.items() if owner == index}
    if not owned:
        return 1.0
    return 1.0 - len(owned & missing) / len(owned)


@dataclass(frozen=True)
class _Function:
    """A function's own executable range and cyclomatic complexity."""

    name: str
    lineno: int
    body_start: int
    endline: int
    complexity: int
    depth: int


def _functions(source: str) -> list[_Function]:
    """Find top-level and nested functions without counting class totals."""
    collector = _FunctionCollector(source)
    collector.visit(ast.parse(source))
    return collector.functions


class _FunctionCollector(ast.NodeVisitor):
    """Collect function blocks with their enclosing class and function names."""

    def __init__(self, source: str) -> None:
        self.source = source
        self.functions: list[_Function] = []
        self.parents: list[str] = []
        self.depth = 0

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.parents.append(node.name)
        self._visit_body(node.body)
        self.parents.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._record_function(node)
        self.parents.append(node.name)
        self.depth += 1
        self._visit_body(node.body)
        self.depth -= 1
        self.parents.pop()

    def _visit_body(self, body: list[ast.stmt]) -> None:
        for child in body:
            self.visit(child)

    def _record_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        code = dedent(ast.get_source_segment(self.source, node) or "")
        self.functions.append(
            _Function(
                name=".".join((*self.parents, node.name)),
                lineno=node.lineno,
                body_start=min(child.lineno for child in node.body),
                endline=node.end_lineno or node.lineno,
                complexity=_function_complexity(code, node.name),
                depth=self.depth,
            )
        )


def _function_complexity(code: str, name: str) -> int:
    """Return Radon's complexity for one function block."""
    for block in cc_visit(code):
        if block.letter == "F":
            return block.complexity
    message = f"Radon found no function block for {name}"
    raise ValueError(message)


def collect(root: Path, data_file: Path) -> list[Score]:
    """
    Score every function in the package.

    Args:
        root: Repository root
        data_file: Path to the coverage data file written by pytest

    Returns:
        Every function in the package, scripts and tests, sorted worst first.
    """
    coverage = Coverage(data_file=str(data_file))
    coverage.load()

    scores: list[Score] = []
    for directory in SOURCE_DIRECTORIES:
        source_root = root / directory
        if not source_root.is_dir():
            message = f"CRAP source directory is missing: {source_root}"
            raise FileNotFoundError(message)
        for path in sorted(source_root.rglob("*.py")):
            scores.extend(score_file(coverage, path, root))
    return sorted(scores, key=lambda score: score.crap, reverse=True)


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the CRAP threshold and report size."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-crap", type=float, required=True)
    parser.add_argument("--data-file", type=Path, default=Path(".coverage"))
    parser.add_argument("--top", type=int, default=15)
    return parser


def main(argv: list[str] | None = None) -> int:
    """
    Report the worst CRAP scores and fail if any breaches the threshold.

    Args:
        argv: Command line arguments, defaulting to sys.argv

    Returns:
        Process exit code: 0 when every function is within the threshold.
    """
    args = build_parser().parse_args(argv)

    root = Path.cwd().resolve()
    if not args.data_file.exists():
        print(
            f"No coverage data at {args.data_file}; run pytest first.", file=sys.stderr
        )
        return 2

    scores = collect(root, args.data_file)
    return _report_scores(scores, args.max_crap, args.top)


def _print_ranked_scores(scores: list[Score], top: int) -> None:
    """Print the worst scores first, capped to the requested count."""
    print(f"Worst {min(top, len(scores))} CRAP scores of {len(scores)} functions:")
    for score in scores[:top]:
        print(f"  {score}")


def _crap_breaches(scores: list[Score], threshold: float) -> list[Score]:
    """Select scores that meet or exceed the exclusive limit."""
    return [score for score in scores if score.crap >= threshold]


def _print_crap_breaches(
    scores: list[Score], breaches: list[Score], threshold: float
) -> int:
    """Print all threshold breaches or confirm that every score passed."""
    if breaches:
        print(
            f"\n{len(breaches)} function(s) meet or exceed the CRAP threshold "
            f"of {threshold:g}:",
            file=sys.stderr,
        )
        for score in breaches:
            print(f"  {score}", file=sys.stderr)
        return 1
    print(
        f"\nAll {len(scores)} functions are within the CRAP threshold of {threshold:g}."
    )
    return 0


def _report_scores(scores: list[Score], threshold: float, top: int) -> int:
    """Print the ranked report and return whether its threshold was breached."""
    if not scores:
        print("No functions measured.", file=sys.stderr)
        return 2
    _print_ranked_scores(scores, top)
    return _print_crap_breaches(scores, _crap_breaches(scores, threshold), threshold)


if __name__ == "__main__":
    raise SystemExit(main())
