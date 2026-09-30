"""Tests for validating and comparing saved pytest-benchmark reports."""

import json
import math
import sys

import pytest

from scripts.benchmark.compare import _finite_median, _medians, compare_reports, main


def report(median: float) -> dict:
    return {"benchmarks": [{"fullname": "test_sample", "stats": {"median": median}}]}


def test_comparison_allows_the_limit_and_reports_both_missing_sides() -> None:
    assert compare_reports(report(1.0), report(1.25)) == []
    assert compare_reports(
        {"benchmarks": [{"fullname": "old", "stats": {"median": 1}}]},
        {"benchmarks": [{"fullname": "new", "stats": {"median": 1}}]},
    ) == ["candidate is missing benchmark old", "base is missing benchmark new"]


def test_comparison_reports_regressions_and_rejects_bad_limit() -> None:
    assert "increased by 25.1%" in compare_reports(report(1.0), report(1.251))[0]
    assert "increased by inf%" in compare_reports(report(0.0), report(0.1))[0]
    with pytest.raises(ValueError, match="maximum_regression"):
        compare_reports(report(1.0), report(1.0), 1.0)


@pytest.mark.parametrize(
    ("median", "message"),
    [
        (True, "finite median"),
        ("1.0", "finite median"),
        (math.inf, "finite median"),
        (-0.1, "must not be negative"),
    ],
)
def test_median_validation_rejects_non_numeric_non_finite_and_negative_values(
    median, message
):
    with pytest.raises(ValueError, match=message):
        _finite_median("test_sample", median)


@pytest.mark.parametrize(
    "invalid_report",
    [
        {},
        {"benchmarks": "not a list"},
        {"benchmarks": [{"stats": {"median": 1.0}}]},
        {"benchmarks": [{"fullname": "", "stats": {"median": 1.0}}]},
        {
            "benchmarks": [
                {"fullname": "same", "stats": {"median": 1.0}},
                {"fullname": "same", "stats": {"median": 1.1}},
            ]
        },
    ],
)
def test_medians_reject_malformed_or_duplicate_benchmarks(invalid_report):
    with pytest.raises((TypeError, ValueError)):
        _medians(invalid_report)


@pytest.mark.parametrize(("candidate", "expected"), [(1.1, 0), (1.5, 1)])
def test_comparison_cli_reads_reports_and_returns_the_gate_result(
    monkeypatch, tmp_path, capsys, candidate, expected
):
    base_path = tmp_path / "base.json"
    candidate_path = tmp_path / "candidate.json"
    base_path.write_text(json.dumps(report(1.0)), encoding="utf-8")
    candidate_path.write_text(json.dumps(report(candidate)), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["compare", str(base_path), str(candidate_path)])

    exit_code = main()

    output = capsys.readouterr().out
    assert exit_code == expected
    expected_message = (
        "Compared 1 benchmark medians" if expected == 0 else "increased by 50.0%"
    )
    assert expected_message in output
