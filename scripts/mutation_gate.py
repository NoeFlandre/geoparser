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
    judged = killed + survived
    score = f"{100 * killed / judged:.1f}%" if judged else "n/a"
    return (
        f"score {score}  killed {killed}  survived {survived}  "
        f"timeout {stats.get('timeout', 0)}  suspicious {stats.get('suspicious', 0)}  "
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
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip() or "no output"
        return f"mutmut results failed with exit code {result.returncode}: {detail}"

    actionable_statuses = (
        ": survived",
        ": timeout",
        ": suspicious",
        ": no tests",
        ": segfault",
        ": caught by type check",
        ": check was interrupted by user",
    )
    return "\n".join(
        line
        for line in result.stdout.splitlines()
        if any(status in line for status in actionable_statuses)
    )


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


def main(argv: list[str] | None = None) -> int:
    """
    Report the mutation score and fail if too many mutants survived.

    Args:
        argv: Command line arguments, defaulting to sys.argv

    Returns:
        Process exit code: 0 when survivors are within the threshold.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-survivors", type=int, required=True)
    parser.add_argument("--max-no-tests", type=int)
    parser.add_argument("--stats", type=Path, default=STATS_PATH)
    parser.add_argument("--patterns", nargs="+")
    args = parser.parse_args(argv)

    if not args.stats.exists():
        print(
            f"No mutation stats at {args.stats}; run "
            f"'mutmut run' then 'mutmut export-cicd-stats' first.",
            file=sys.stderr,
        )
        return 2

    exported_stats = json.loads(args.stats.read_text(encoding="utf-8"))
    stats = exported_stats
    scoped_error: str | None = None
    scoped_diagnostic_text: str | None = None
    if args.patterns:
        result = subprocess.run(
            (sys.executable, "-m", "mutmut", "results", "--all", "true"),
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode:
            detail = result.stderr.strip() or result.stdout.strip() or "no output"
            scoped_error = (
                f"mutmut results failed with exit code {result.returncode}: {detail}"
            )
        else:
            mutants = scoped_mutants(result.stdout, args.patterns)
            if not mutants:
                scoped_error = "No mutmut results matched the changed-module patterns."
            else:
                stats = summarize_scoped_mutants(mutants)
                exported_checked = sum(
                    exported_stats.get(outcome, 0) for outcome in MUTATION_OUTCOMES
                )
                scoped_checked = sum(
                    stats.get(outcome, 0) for outcome in MUTATION_OUTCOMES
                )
                if scoped_checked != exported_checked:
                    scoped_error = (
                        "Changed-module results account for "
                        f"{scoped_checked} mutant(s), but mutmut stats report "
                        f"{exported_checked} checked mutant(s)."
                    )
                scoped_diagnostic_text = scoped_diagnostics(mutants)

    if scoped_error:
        print(scoped_error, file=sys.stderr)
        return 1

    print(f"Mutation testing: {summarize(stats)}")

    unchecked = unaccounted_mutants(stats)
    if unchecked:
        print(
            f"\n{unchecked} of {stats.get('total', 0)} mutant(s) were never "
            f"checked. mutmut writes a stats file and exits zero even when its "
            f"own baseline test run failed, so a survivor count of zero here "
            f"means nothing was verified rather than nothing was wrong. Run "
            f"'mutmut run' again and read its output.",
            file=sys.stderr,
        )
        return 1

    survived = stats.get("survived", 0)
    if survived > args.max_survivors:
        diagnostic_text = scoped_diagnostic_text or mutation_diagnostics()
        print(
            f"\n{survived} mutant(s) survived, more than the agreed "
            f"{args.max_survivors}. Strengthen the tests that should have "
            f"killed them.",
            file=sys.stderr,
        )
        print(
            "\nMutation diagnostics:\n"
            + (diagnostic_text or "No actionable mutant details were returned."),
            file=sys.stderr,
        )
        return 1

    no_tests = stats.get("no_tests", 0)
    if args.max_no_tests is not None and no_tests > args.max_no_tests:
        diagnostic_text = scoped_diagnostic_text or mutation_diagnostics()
        print(
            f"\n{no_tests} mutant(s) have no covering tests, more than the "
            f"agreed {args.max_no_tests}. Add a focused unit test or justify "
            f"the scope in MUTATION_TESTING.md.",
            file=sys.stderr,
        )
        print(
            "\nMutation diagnostics:\n"
            + (diagnostic_text or "No actionable mutant details were returned."),
            file=sys.stderr,
        )
        return 1

    no_tests_baseline = (
        str(args.max_no_tests) if args.max_no_tests is not None else "unconfigured"
    )
    print(
        "Within the agreed baselines of "
        f"{args.max_survivors} surviving and {no_tests_baseline} untested mutant(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
