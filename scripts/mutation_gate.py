"""
Mutation testing gate.

Reads the summary that ``mutmut export-cicd-stats`` writes and fails when more
mutants survived than the agreed baseline. A surviving mutant is a change to
the source that the test suite did not notice, so it marks a line that is
executed but not actually checked.

    uv run mutmut run
    uv run mutmut export-cicd-stats
    uv run python scripts/mutation_gate.py --max-survivors 0

Fix a survivor by strengthening the test that should have caught it, not by
raising the threshold.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

STATS_PATH = Path("mutants/mutmut-cicd-stats.json")
MUTATION_OUTCOMES = (
    "killed",
    "survived",
    "timeout",
    "suspicious",
    "no_tests",
    "skipped",
)


def summarize(stats: dict[str, int]) -> str:
    """
    Render a one-line summary of a mutation run.

    Args:
        stats: The counts mutmut exported

    Returns:
        A human-readable summary line
    """
    killed = stats.get("killed", 0)
    survived = stats.get("survived", 0)
    timed_out = stats.get("timeout", 0)
    judged = killed + survived
    score = f"{100 * killed / judged:.1f}%" if judged else "n/a"
    return (
        f"resolved-outcome kill rate {score} ({killed} killed / {judged} decided); "
        f"survived {survived}; timeout {timed_out} "
        "(inconclusive; not counted as killed); "
        f"suspicious {stats.get('suspicious', 0)}  "
        f"no tests {stats.get('no_tests', 0)}  skipped {stats.get('skipped', 0)}  "
        f"total {stats.get('total', 0)}"
    )


def unaccounted_mutants(stats: dict[str, int]) -> int:
    """
    How many generated mutants the run reached no verdict on.

    Every mutant mutmut generates should end in exactly one outcome. A run cut
    short -- most often because mutmut's own baseline test run failed -- still
    writes a stats file, but with those mutants in none of the categories.

    Args:
        stats: The counts mutmut exported

    Returns:
        The number of mutants with no recorded outcome, zero when all are
        accounted for
    """
    accounted = sum(
        stats.get(outcome, 0)
        for outcome in (
            "killed",
            "survived",
            "timeout",
            "suspicious",
            "no_tests",
            "skipped",
        )
    )
    return max(stats.get("total", 0) - accounted, 0)


def mutation_diagnostics() -> str:
    """Return actionable non-killed mutant lines from the current run."""
    result = subprocess.run(
        (sys.executable, "-m", "mutmut", "results"),
        check=False,
        capture_output=True,
        text=True,
    )
    error = _result_error(result.returncode, result.stderr, result.stdout)
    return error or _actionable_result_lines(result.stdout)


def _actionable_result_lines(output: str) -> str:
    """Keep only mutmut result records that deserve human attention."""
    return "\n".join(
        line for line in output.splitlines() if _actionable_result_line(line)
    )


def _actionable_result_line(line: str) -> bool:
    """Whether a mutmut result line describes an unresolved outcome."""
    statuses = (
        ": survived",
        ": timeout",
        ": suspicious",
        ": no tests",
        ": segfault",
        ": caught by type check",
        ": check was interrupted by user",
    )
    return any(status in line for status in statuses)


def scoped_mutants(output: str, patterns: list[str]) -> list[tuple[str, str]]:
    """Return mutant records whose names match one of the requested patterns."""
    mutants = []
    for line in output.splitlines():
        record = line.strip()
        if ": " not in record:
            continue
        name, status = record.rsplit(": ", maxsplit=1)
        if any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns):
            mutants.append((name, status.lower()))
    return mutants


def summarize_scoped_mutants(mutants: list[tuple[str, str]]) -> dict[str, int]:
    """Build mutation-gate counts from the selected mutmut result records."""
    status_keys = {"no tests": "no_tests"}
    status_keys.update({outcome: outcome for outcome in MUTATION_OUTCOMES})
    counts = Counter(
        status_keys[status] for _, status in mutants if status in status_keys
    )
    counts["total"] = len(mutants)
    return dict(counts)


def scoped_diagnostics(mutants: list[tuple[str, str]]) -> str:
    """Render selected mutants that do not have a killed result."""
    return "\n".join(
        f"    {name}: {status}" for name, status in mutants if status != "killed"
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the mutation gate's command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-survivors", type=int, required=True)
    parser.add_argument("--max-no-tests", type=int)
    parser.add_argument("--stats", type=Path, default=STATS_PATH)
    parser.add_argument("--patterns", nargs="+")
    return parser.parse_args(argv)


def _scope_stats(
    exported: dict[str, int], patterns: list[str] | None
) -> tuple[dict[str, int], str | None, str | None]:
    """Select changed-module mutants and verify their exported counts."""
    if not patterns:
        return exported, None, None

    result = subprocess.run(
        (sys.executable, "-m", "mutmut", "results", "--all", "true"),
        check=False,
        capture_output=True,
        text=True,
    )
    error = _result_error(result.returncode, result.stderr, result.stdout)
    if error:
        return exported, error, None

    mutants = scoped_mutants(result.stdout, patterns)
    if not mutants:
        return exported, "No mutmut results matched the changed-module patterns.", None

    stats = summarize_scoped_mutants(mutants)
    mismatch = _scope_count_mismatch(exported, stats)
    return stats, mismatch, scoped_diagnostics(mutants)


def _result_error(returncode: int, stderr: str, stdout: str) -> str | None:
    """Format a failed mutmut subprocess result, when one occurred."""
    if not returncode:
        return None
    detail = stderr.strip() or stdout.strip() or "no output"
    return f"mutmut results failed with exit code {returncode}: {detail}"


def _scope_count_mismatch(
    exported: dict[str, int], stats: dict[str, int]
) -> str | None:
    """Ensure detailed changed-module results match the exported run totals."""
    exported_checked = sum(exported.get(outcome, 0) for outcome in MUTATION_OUTCOMES)
    scoped_checked = sum(stats.get(outcome, 0) for outcome in MUTATION_OUTCOMES)
    if scoped_checked == exported_checked:
        return None
    return (
        "Changed-module results account for "
        f"{scoped_checked} mutant(s), but mutmut stats report "
        f"{exported_checked} checked mutant(s)."
    )


def _unchecked_failure(stats: dict[str, int]) -> str | None:
    """Describe mutants for which mutmut never recorded a test outcome."""
    unchecked = unaccounted_mutants(stats)
    if not unchecked:
        return None
    return (
        f"\n{unchecked} of {stats.get('total', 0)} mutant(s) were never "
        f"checked. mutmut writes a stats file and exits zero even when its "
        f"own baseline test run failed, so a survivor count of zero here "
        f"means nothing was verified rather than nothing was wrong. Run "
        f"'mutmut run' again and read its output."
    )


def _survivor_failure(
    stats: dict[str, int], max_survivors: int, diagnostics: str | None
) -> tuple[str, str] | None:
    """Describe survivors that exceed the configured maximum."""
    survived = stats.get("survived", 0)
    if survived <= max_survivors:
        return None
    message = (
        f"\n{survived} mutant(s) survived, more than the agreed "
        f"{max_survivors}. Strengthen the tests that should have killed them."
    )
    return message, diagnostics or mutation_diagnostics()


def _no_tests_failure(
    stats: dict[str, int], max_no_tests: int | None, diagnostics: str | None
) -> tuple[str, str] | None:
    """Describe no-tests mutants that exceed the configured allowance."""
    no_tests = stats.get("no_tests", 0)
    if max_no_tests is None or no_tests <= max_no_tests:
        return None
    message = (
        f"\n{no_tests} mutant(s) have no covering tests, more than the "
        f"agreed {max_no_tests}. Add a focused unit test or justify the "
        f"scope in MUTATION_TESTING.md."
    )
    return message, diagnostics or mutation_diagnostics()


def _gate_failure(
    stats: dict[str, int],
    max_survivors: int,
    max_no_tests: int | None,
    diagnostics: str | None,
) -> tuple[str, str | None] | None:
    """Return the gate failure and its diagnostics, if a baseline was breached."""
    unchecked = _unchecked_failure(stats)
    if unchecked:
        return unchecked, None
    return _survivor_failure(stats, max_survivors, diagnostics) or _no_tests_failure(
        stats, max_no_tests, diagnostics
    )


def _load_stats(path: Path) -> tuple[dict[str, int], str | None]:
    """Load the mutmut export or describe how to create it."""
    if not path.exists():
        message = (
            f"No mutation stats at {path}; run "
            f"'mutmut run' then 'mutmut export-cicd-stats' first."
        )
        return {}, message
    return json.loads(path.read_text(encoding="utf-8")), None


def _print_gate_failure(failure: tuple[str, str | None]) -> None:
    """Print a failed baseline and any actionable mutant diagnostics."""
    message, details = failure
    print(message, file=sys.stderr)
    if details is not None:
        print(
            "\nMutation diagnostics:\n"
            + (details or "No actionable mutant details were returned."),
            file=sys.stderr,
        )


def _run_gate(args: argparse.Namespace, exported: dict[str, int]) -> int:
    """Apply scoped selection and the configured mutation baselines."""
    stats, scoped_error, diagnostics = _scope_stats(exported, args.patterns)
    if scoped_error:
        print(scoped_error, file=sys.stderr)
        return 1

    print(f"Mutation testing: {summarize(stats)}")
    failure = _gate_failure(stats, args.max_survivors, args.max_no_tests, diagnostics)
    if failure:
        _print_gate_failure(failure)
        return 1

    no_tests_baseline = (
        str(args.max_no_tests) if args.max_no_tests is not None else "unconfigured"
    )
    print(
        "Within the agreed baselines of "
        f"{args.max_survivors} surviving and {no_tests_baseline} untested mutant(s)."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Report the mutation score and fail when either baseline is breached."""
    args = _parse_args(argv)
    exported, error = _load_stats(args.stats)
    if error:
        print(error, file=sys.stderr)
        return 2
    return _run_gate(args, exported)


if __name__ == "__main__":
    raise SystemExit(main())
