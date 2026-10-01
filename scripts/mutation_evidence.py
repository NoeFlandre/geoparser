"""Capture per-mutant results and diagnostics for an exact CI run."""

from __future__ import annotations

import argparse
import fnmatch
import importlib.metadata
import json
import os
import platform
import re
import shlex
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_NO_DIAGNOSTIC_OUTCOMES = {
    "killed",
    "not checked",
    "not run",
    "unchecked",
}


def _mutmut_command(*arguments: str) -> tuple[str, ...]:
    """Return a mutmut invocation using the active Python environment."""
    return (sys.executable, "-m", "mutmut", *arguments)


def parse_results(output: str) -> tuple[list[dict[str, str]], list[str]]:
    """Parse result records and retain any lines the parser cannot classify."""
    records = []
    unparsed = []
    for line in output.splitlines():
        if not line.strip():
            continue
        record = _parse_result_line(line)
        if record is None:
            unparsed.append(line)
        else:
            records.append(record)
    return records, unparsed


def _parse_result_line(line: str) -> dict[str, str] | None:
    """Parse one mutmut line or return ``None`` when it is not a result."""
    clean_line = _ANSI_ESCAPE.sub("", line).strip()
    if ": " not in clean_line:
        return None
    mutant_id, outcome = clean_line.rsplit(": ", maxsplit=1)
    if not mutant_id.startswith("geoparser.") or not outcome:
        return None
    return {"id": mutant_id, "outcome": outcome.lower()}


def select_results(
    records: list[dict[str, str]], patterns: list[str]
) -> list[dict[str, str]]:
    """Apply the same mutmut globs used to select changed package modules."""
    if not patterns:
        return records
    return [
        record
        for record in records
        if any(fnmatch.fnmatchcase(record["id"], pattern) for pattern in patterns)
    ]


def _read_stats(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Read exported counts while preserving missing or malformed-file errors."""
    try:
        stats = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return None, f"{type(error).__name__}: {error}"
    return stats, None


def _unaccounted_count(stats: dict[str, Any] | None) -> int | None:
    """Count generated mutants that did not receive any exported outcome."""
    if stats is None or not isinstance(stats.get("total"), int):
        return None
    outcomes = ("killed", "survived", "timeout", "suspicious", "no_tests", "skipped")
    accounted = sum(int(stats.get(outcome, 0)) for outcome in outcomes)
    return max(stats["total"] - accounted, 0)


def _needs_diagnostic(outcome: str) -> bool:
    """Whether mutmut should show the source change for an outcome."""
    return outcome not in _NO_DIAGNOSTIC_OUTCOMES


def _with_diagnostics(records: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Attach a reproducible mutant command and unresolved-result diff."""
    enriched = []
    for record in records:
        mutant_id = record["id"]
        replay = (
            "uv",
            "run",
            "--no-sync",
            "mutmut",
            "run",
            "--max-children",
            "1",
            mutant_id,
        )
        item: dict[str, Any] = {
            **record,
            "replay_command": shlex.join(replay),
            "diagnostic": None,
        }
        if _needs_diagnostic(record["outcome"]):
            command = _mutmut_command("show", mutant_id)
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
            item["diagnostic"] = {
                "command": list(command),
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        enriched.append(item)
    return enriched


def _revision() -> str:
    """Return the checked-out git commit or an explicit unknown marker."""
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _mutmut_version() -> str | None:
    """Return the installed mutmut version when package metadata is available."""
    try:
        return importlib.metadata.version("mutmut")
    except importlib.metadata.PackageNotFoundError:
        return None


def _metadata(patterns: list[str]) -> dict[str, Any]:
    """Record the exact source and runtime context for the evidence files."""
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkout_revision": _revision(),
        "github_sha": os.environ.get("GITHUB_SHA"),
        "pull_request_head_sha": os.environ.get("GITHUB_HEAD_SHA"),
        "workflow_run_id": os.environ.get("GITHUB_RUN_ID"),
        "workflow_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "python_version": sys.version,
        "platform": platform.platform(),
        "mutmut_version": _mutmut_version(),
        "selected_patterns": patterns,
        "selection_mode": "full" if not patterns else "changed_modules",
    }


def _write_json(path: Path, value: Any) -> None:
    """Write formatted UTF-8 JSON with a final newline."""
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_evidence(
    evidence_dir: Path,
    stats_path: Path,
    patterns: list[str],
    run_exit_code: int,
    export_exit_code: int,
    gate_exit_code: int,
) -> int:
    """Persist raw results, stats, metadata, mutant outcomes, and diagnostics."""
    evidence_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        _mutmut_command("results", "--all", "true"),
        check=False,
        capture_output=True,
        text=True,
    )
    (evidence_dir / "mutmut-results.txt").write_text(result.stdout, encoding="utf-8")
    (evidence_dir / "mutmut-results.stderr.txt").write_text(
        result.stderr, encoding="utf-8"
    )
    all_records, unparsed = parse_results(result.stdout)
    records = select_results(all_records, patterns)
    stats, stats_error = _read_stats(stats_path)
    if stats is not None:
        _write_json(evidence_dir / "mutmut-cicd-stats.json", stats)
    report = {
        "schema_version": 1,
        "metadata": _metadata(patterns),
        "exit_codes": {
            "mutation_run": run_exit_code,
            "stats_export": export_exit_code,
            "mutation_gate": gate_exit_code,
            "results_capture": result.returncode,
        },
        "stats": stats,
        "stats_error": stats_error,
        "unaccounted_mutants": _unaccounted_count(stats),
        "counts_by_outcome": dict(Counter(item["outcome"] for item in records)),
        "mutant_count": len(records),
        "mutants": _with_diagnostics(records),
        "unparsed_result_lines": unparsed,
    }
    _write_json(evidence_dir / "report.json", report)
    print(
        "Mutation evidence: "
        f"{len(records)} mutant record(s), "
        f"{report['counts_by_outcome'].get('timeout', 0)} timeout(s), "
        f"{report['counts_by_outcome'].get('survived', 0)} survivor(s)."
    )
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse paths, run outcomes, and changed-module selection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--run-exit-code", type=int, required=True)
    parser.add_argument("--export-exit-code", type=int, required=True)
    parser.add_argument("--gate-exit-code", type=int, required=True)
    parser.add_argument("--patterns", nargs="*", default=[])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Write evidence for the mutation run that just finished."""
    args = _parse_args(argv)
    return write_evidence(
        args.evidence_dir,
        args.stats,
        args.patterns,
        args.run_exit_code,
        args.export_exit_code,
        args.gate_exit_code,
    )


if __name__ == "__main__":
    raise SystemExit(main())
