"""Replay a small, exact allowlist of previously timed-out mutants in CI."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.mutation_evidence import parse_results

_ALLOWLIST = Path(__file__).with_name("mutation_replay_allowlist.json")
_SHA = re.compile(r"[0-9a-f]{40}")
_ALLOWED_ID_CHARACTERS = frozenset("._ǁ")
_FAILURE_OUTCOMES = frozenset({"survived", "suspicious", "no tests", "unreported"})


class ReplaySelectionError(ValueError):
    """The requested CI replay selection does not satisfy its allowlist."""


def _require(message: str, *, condition: bool) -> None:
    """Raise a concise validation error when one selection contract fails."""
    if not condition:
        raise ReplaySelectionError(message)


def _is_literal_mutant_id(mutant_id: str) -> bool:
    """Require a mutmut ID made from safe literal-name characters."""
    return (
        mutant_id.startswith("geoparser.")
        and "__mutmut_" in mutant_id
        and all(
            character.isalnum() or character in _ALLOWED_ID_CHARACTERS
            for character in mutant_id
        )
    )


def _load_allowlist(path: Path) -> dict[str, Any]:
    """Load and validate the versioned timeout evidence allowlist."""
    allowlist = json.loads(path.read_text(encoding="utf-8"))
    _require(
        "Replay allowlist must be a JSON object", condition=isinstance(allowlist, dict)
    )
    _require(
        "Unsupported replay allowlist schema",
        condition=allowlist.get("schema_version") == 1,
    )
    ids = allowlist.get("mutant_ids")
    _require(
        "Replay allowlist has no mutant IDs",
        condition=isinstance(ids, list) and bool(ids),
    )
    _require(
        "Replay allowlist IDs must be strings",
        condition=all(isinstance(item, str) for item in ids),
    )
    _require(
        "Replay allowlist contains duplicate mutant IDs",
        condition=len(ids) == len(set(ids)),
    )
    _require(
        "Replay allowlist contains an invalid literal mutant ID",
        condition=all(_is_literal_mutant_id(item) for item in ids),
    )
    source = allowlist.get("source")
    _require(
        "Replay allowlist has no source evidence", condition=isinstance(source, dict)
    )
    _require(
        "Allowlist size differs from its source timeout count",
        condition=source.get("timeout_count") == len(ids),
    )
    maximum = allowlist.get("max_selected_per_dispatch")
    _require(
        "Replay selection cap must be between 1 and 8",
        condition=type(maximum) is int and 0 < maximum <= 8,
    )
    return allowlist


def _selection_lines(selection_text: str, maximum: int) -> list[str]:
    """Parse exact newline-separated IDs and enforce the dispatch bound."""
    lines = selection_text.splitlines()
    _require(
        "Mutant IDs cannot have surrounding whitespace",
        condition=all(line == line.strip() for line in lines),
    )
    selected = [line for line in lines if line]
    _require("Select at least one exact mutant ID", condition=bool(selected))
    _require(
        f"Select at most {maximum} mutants per dispatch",
        condition=len(selected) <= maximum,
    )
    _require(
        "Selection contains duplicate mutant IDs",
        condition=len(selected) == len(set(selected)),
    )
    return selected


def validate_selection(
    selection_text: str,
    allowlist_path: Path,
    expected_sha: str,
    actual_sha: str,
) -> tuple[list[str], dict[str, Any]]:
    """Check exact commit identity and membership in the finite allowlist."""
    _require(
        "Expected SHA must be a full lowercase 40-character commit SHA",
        condition=bool(_SHA.fullmatch(expected_sha)),
    )
    _require(
        "Selected ref SHA does not match expected_sha",
        condition=expected_sha == actual_sha,
    )
    allowlist = _load_allowlist(allowlist_path)
    selected = _selection_lines(selection_text, allowlist["max_selected_per_dispatch"])
    allowed = set(allowlist["mutant_ids"])
    _require(
        "Selection contains an ID outside the timeout allowlist",
        condition=all(mutant_id in allowed for mutant_id in selected),
    )
    return selected, allowlist


def _command(*arguments: str) -> list[str]:
    """Build a shell-free mutmut command for the current locked environment."""
    return [sys.executable, "-m", "mutmut", *arguments]


def _outcomes_by_id(output: str) -> dict[str, str]:
    """Return the most recent mutmut result for each exact mutant ID."""
    records, _unparsed = parse_results(output)
    return {record["id"]: record["outcome"] for record in records}


def _write_text(path: Path, contents: str) -> None:
    """Write one UTF-8 evidence file with a stable trailing newline."""
    path.write_text(contents.rstrip() + "\n", encoding="utf-8")


def _run_selected(
    selected: list[str],
    evidence_dir: Path,
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> tuple[list[dict[str, Any]], float]:
    """Run the exact selection serially and capture its terminal results."""
    started = time.monotonic()
    command = _command("run", "--max-children", "1", *selected)
    run_result = runner(command, check=False, capture_output=True, text=True)
    results_command = _command("results", "--all", "true")
    results_result = runner(
        results_command, check=False, capture_output=True, text=True
    )
    duration = time.monotonic() - started
    outcomes = _outcomes_by_id(results_result.stdout)
    _write_text(
        evidence_dir / "mutmut-run.log",
        "\n".join(
            (
                f"run_exit_code: {run_result.returncode}",
                run_result.stdout,
                run_result.stderr,
            )
        ),
    )
    _write_text(
        evidence_dir / "mutmut-results.txt",
        "\n".join(
            (
                f"results_exit_code: {results_result.returncode}",
                results_result.stdout,
                results_result.stderr,
            )
        ),
    )
    records = [
        {
            "mutant_id": mutant_id,
            "outcome": outcomes.get(mutant_id, "unreported"),
            "run_exit_code": run_result.returncode,
            "results_exit_code": results_result.returncode,
        }
        for mutant_id in selected
    ]
    return records, round(duration, 3)


def _summary(records: list[dict[str, Any]]) -> str:
    """Report diagnostic outcomes without counting timeouts as kills."""
    counts = Counter(record["outcome"] for record in records)
    return (
        f"Diagnostic replay: {counts['killed']} killed, {counts['survived']} survived, "
        f"{counts['timeout']} timeout (inconclusive), {counts['no tests']} no-tests, "
        f"{counts['unreported']} unreported across {len(records)} selected mutant(s)."
    )


def _replay_failed(records: list[dict[str, Any]]) -> bool:
    """Fail diagnostic execution on survivors or missing/error results."""
    return any(
        record["outcome"] in _FAILURE_OUTCOMES
        or record["run_exit_code"] != 0
        or record["results_exit_code"] != 0
        for record in records
    )


def execute_replay(
    selection_text: str,
    allowlist_path: Path,
    expected_sha: str,
    actual_sha: str,
    evidence_dir: Path,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> int:
    """Run the bounded selection and write reproducible CI replay evidence."""
    selected, allowlist = validate_selection(
        selection_text, allowlist_path, expected_sha, actual_sha
    )
    evidence_dir.mkdir(parents=True, exist_ok=True)
    records, elapsed = _run_selected(selected, evidence_dir, runner)
    summary = _summary(records)
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "actual_sha": actual_sha,
        "expected_sha": expected_sha,
        "allowlist_source": allowlist["source"],
        "selected_mutants": records,
        "elapsed_seconds": elapsed,
        "counts_by_outcome": dict(Counter(record["outcome"] for record in records)),
        "diagnostic_summary": summary,
        "timeouts_are_inconclusive": True,
    }
    (evidence_dir / "replay-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_text(evidence_dir / "summary.txt", summary)
    print(summary)
    return int(_replay_failed(records))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the explicit CI dispatch arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-file", type=Path, required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--actual-sha", required=True)
    parser.add_argument("--evidence-dir", type=Path, default=Path("mutation-replay"))
    parser.add_argument("--allowlist", type=Path, default=_ALLOWLIST)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the allowlisted replay and convert validation errors to CI errors."""
    args = _parse_args(argv)
    try:
        selection = args.selection_file.read_text(encoding="utf-8")
        if args.validate_only:
            selected, _allowlist = validate_selection(
                selection, args.allowlist, args.expected_sha, args.actual_sha
            )
            print("\n".join(selected))
            return 0
        return execute_replay(
            selection,
            args.allowlist,
            args.expected_sha,
            args.actual_sha,
            args.evidence_dir,
        )
    except ReplaySelectionError as error:
        print(f"Mutation replay selection rejected: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
