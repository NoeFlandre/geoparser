"""Compare opt-in pytest-benchmark reports against a pull request base."""

from __future__ import annotations

import argparse
import json
import math
import statistics
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


def _median_report(reports: list[dict[str, Any]], label: str) -> dict[str, float]:
    """Return medians across repeated report medians with stable test inventory."""
    if not reports:
        msg = f"{label} reports must not be empty"
        raise ValueError(msg)

    values = [_medians(report) for report in reports]
    names = set(values[0])
    for index, report_values in enumerate(values[1:], start=2):
        if set(report_values) != names:
            msg = f"{label} report {index} has a different benchmark inventory"
            raise ValueError(msg)
    return {
        name: statistics.median(report_values[name] for report_values in values)
        for name in sorted(names)
    }


def compare_repeated_reports(
    baseline_reports: list[dict[str, Any]],
    candidate_reports: list[dict[str, Any]],
    maximum_regression: float = 0.25,
) -> list[str]:
    """Compare the median of repeated per-run medians against the same gate."""
    if not baseline_reports or len(baseline_reports) != len(candidate_reports):
        msg = "baseline and candidate reports must have the same non-zero count"
        raise ValueError(msg)

    baseline = _median_report(baseline_reports, "baseline")
    candidate = _median_report(candidate_reports, "candidate")
    return compare_reports(
        {
            "benchmarks": [
                {"fullname": name, "stats": {"median": value}}
                for name, value in baseline.items()
            ]
        },
        {
            "benchmarks": [
                {"fullname": name, "stats": {"median": value}}
                for name, value in candidate.items()
            ]
        },
        maximum_regression,
    )


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
    parser.add_argument("--additional-baseline", type=Path, action="append", default=[])
    parser.add_argument(
        "--additional-candidate", type=Path, action="append", default=[]
    )
    parser.add_argument("--maximum-regression", type=float, default=0.25)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)

    baseline_reports = [
        json.loads(path.read_text())
        for path in [arguments.baseline, *arguments.additional_baseline]
    ]
    candidate_reports = [
        json.loads(path.read_text())
        for path in [arguments.candidate, *arguments.additional_candidate]
    ]
    if len(baseline_reports) == 1 and len(candidate_reports) == 1:
        baseline_report, candidate_report = baseline_reports[0], candidate_reports[0]
        failures = compare_reports(
            baseline_report, candidate_report, arguments.maximum_regression
        )
    else:
        failures = compare_repeated_reports(
            baseline_reports, candidate_reports, arguments.maximum_regression
        )
        baseline_report = {
            "benchmarks": [
                {"fullname": name, "stats": {"median": value}}
                for name, value in _median_report(baseline_reports, "baseline").items()
            ]
        }
        candidate_report = {
            "benchmarks": [
                {"fullname": name, "stats": {"median": value}}
                for name, value in _median_report(
                    candidate_reports, "candidate"
                ).items()
            ]
        }
    if failures:
        print("Benchmark regression check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    compared = len(_medians(candidate_report))
    repeated = len(baseline_reports)
    suffix = f" from {repeated} runs per revision" if repeated > 1 else ""
    print(
        f"Compared {compared} benchmark medians{suffix}; no regression exceeded "
        f"{arguments.maximum_regression:.0%}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
