"""Unit checks for the pull request benchmark gate."""

import pytest

from scripts.benchmark.compare import compare_reports

pytestmark = pytest.mark.benchmark


def _report(median: float) -> dict:
    return {
        "benchmarks": [
            {
                "fullname": "test_sample",
                "stats": {"median": median},
            }
        ]
    }


def test_benchmark_comparison_allows_regressions_up_to_25_percent():
    assert compare_reports(_report(1.0), _report(1.25)) == []


def test_benchmark_comparison_rejects_regressions_above_25_percent():
    failures = compare_reports(_report(1.0), _report(1.251))

    assert len(failures) == 1
    assert "increased by 25.1%" in failures[0]


def test_benchmark_comparison_rejects_missing_measurements():
    failures = compare_reports(_report(1.0), {"benchmarks": []})

    assert failures == ["candidate is missing benchmark test_sample"]
