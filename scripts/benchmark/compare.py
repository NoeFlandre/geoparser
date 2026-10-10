"""Compare opt-in pytest-benchmark reports against a pull request base."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _medians(report: dict[str, Any]) -> dict[str, float]:
    """Return finite medians keyed by pytest-benchmark's stable full test name."""
    benchmarks = report.get("benchmarks")
    if not isinstance(benchmarks, list):
        msg = "benchmark report must contain a benchmarks list"
        raise TypeError(msg)

    medians = {}
    for benchmark in benchmarks:
        name = _benchmark_name(benchmark)
        if name in medians:
            msg = f"duplicate benchmark fullname: {name}"
            raise ValueError(msg)
        medians[name] = _finite_median(name, benchmark.get("stats", {}).get("median"))
    return medians


def _benchmark_name(benchmark: dict[str, Any]) -> str:
    """Validate the stable identifier used to pair benchmark measurements."""
    name = benchmark.get("fullname")
    if not isinstance(name, str) or not name:
        msg = "every benchmark must have a fullname"
        raise ValueError(msg)
    return name


def _finite_median(name: str, median: Any) -> float:
    """Validate a timing value before using it in relative comparisons."""
    if isinstance(median, bool):
        msg = f"benchmark {name} must have a finite median"
        raise ValueError(msg)  # noqa: TRY004 - malformed reports share one error type.
    if not isinstance(median, int | float):
        msg = f"benchmark {name} must have a finite median"
        raise ValueError(msg)  # noqa: TRY004 - preserve the existing ValueError contract.
    if not math.isfinite(median):
        msg = f"benchmark {name} must have a finite median"
        raise ValueError(msg)
    if median < 0:
        msg = f"benchmark {name} median must not be negative"
        raise ValueError(msg)
    return float(median)


def compare_reports(
    baseline_report: dict[str, Any],
    candidate_report: dict[str, Any],
    maximum_regression: float = 0.25,
) -> list[str]:
    """Describe missing measurements and medians exceeding the allowed increase."""
    if not 0 <= maximum_regression < 1:
        msg = "maximum_regression must be in [0, 1)"
        raise ValueError(msg)

    baseline = _medians(baseline_report)
    candidate = _medians(candidate_report)
    failures = [
        f"candidate is missing benchmark {name}"
        for name in sorted(baseline.keys() - candidate.keys())
    ]
    failures.extend(
        f"base is missing benchmark {name}"
        for name in sorted(candidate.keys() - baseline.keys())
    )
    failures.extend(_regressions(baseline, candidate, maximum_regression))
    return failures


def _regressions(
    baseline: dict[str, float], candidate: dict[str, float], maximum_regression: float
) -> list[str]:
    """Describe common benchmarks whose candidate median exceeds the limit."""
    failures = []
    limit = 1 + maximum_regression
    for name in sorted(baseline.keys() & candidate.keys()):
        before = baseline[name]
        after = candidate[name]
        if after > before * limit:
            change = math.inf if before == 0 else (after / before) - 1
            failures.append(
                f"{name}: median increased by {change:.1%} "
                f"({before:.6g}s -> {after:.6g}s; limit {maximum_regression:.1%})"
            )
    return failures


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the median regression check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--maximum-regression", type=float, default=0.25)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)

    baseline_report = json.loads(arguments.baseline.read_text())
    candidate_report = json.loads(arguments.candidate.read_text())
    failures = compare_reports(
        baseline_report, candidate_report, arguments.maximum_regression
    )
    if failures:
        print("Benchmark regression check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    compared = len(_medians(candidate_report))
    print(
        f"Compared {compared} benchmark medians; no regression exceeded "
        f"{arguments.maximum_regression:.0%}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
