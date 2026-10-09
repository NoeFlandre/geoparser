import copy

import pytest

from scripts.masakhaner_benchmark.coverage import (
    coverage_rows,
    coverage_summary,
)
from scripts.masakhaner_benchmark.manifest import read_manifest


@pytest.fixture(scope="module")
def manifest():
    return read_manifest()


@pytest.fixture(scope="module")
def rows(manifest):
    return coverage_rows(manifest)


@pytest.fixture(scope="module")
def summary(manifest, rows):
    return coverage_summary(manifest, rows)


def test_the_target_table_has_one_row_per_canonical_code(rows):
    assert len(rows) == 85
    assert len({row["code"] for row in rows}) == 85


def test_only_hausa_igbo_xhosa_yoruba_and_zulu_are_covered(summary):
    assert summary["covered_codes"] == ["ha", "ig", "xh", "yo", "zu"]
    assert summary["covered_targets"] == 5
    assert summary["missing_targets"] == 80


def test_the_included_configurations_are_the_five_matching_iso_codes(summary):
    assert summary["included_configurations"] == ["hau", "ibo", "xho", "yor", "zul"]


def test_amharic_is_a_target_that_masakhaner_2_0_does_not_cover(rows):
    amharic = next(row for row in rows if row["code"] == "am")

    assert amharic["status"] == "missing"
    assert amharic["masakhaner_config"] is None
    assert amharic["readme_counts"] is None


def test_hausa_xhosa_and_zulu_fill_the_wikiann_gap(summary):
    assert summary["wikiann_missing_targets"] == ["ha", "xh", "zu"]
    assert summary["wikiann_gaps_filled"] == ["ha", "xh", "zu"]


def test_a_covered_row_carries_its_readme_counts(rows):
    hausa = next(row for row in rows if row["code"] == "ha")

    assert hausa["status"] == "covered"
    assert hausa["masakhaner_config"] == "hau"
    assert hausa["fills_wikiann_gap"] is True
    assert hausa["readme_counts"] == {"train": 5716, "validation": 816, "test": 1633}


def test_covered_example_totals_are_summed_per_split(summary):
    assert summary["examples_by_split"] == {
        "train": 31793,
        "validation": 4542,
        "test": 9081,
    }


def test_fifteen_configurations_are_excluded_with_a_reason(summary):
    excluded = {row["config"]: row for row in summary["excluded_configurations"]}

    assert len(excluded) == 15
    assert (
        excluded["bbj"]["reason"]
        == "no ISO 639-1 code, so it cannot match the target list"
    )
    assert excluded["swa"]["reason"] == (
        "its ISO 639-1 code is not in the 85-language target list"
    )
    assert "hau" not in excluded


def test_removing_a_configuration_moves_its_target_to_missing(manifest):
    changed = copy.deepcopy(manifest)
    changed["languages"] = [
        row for row in changed["languages"] if row["config"] != "hau"
    ]

    rows = coverage_rows(changed)
    summary = coverage_summary(changed, rows)

    assert summary["covered_codes"] == ["ig", "xh", "yo", "zu"]
    assert "ha" in summary["missing_codes"]
