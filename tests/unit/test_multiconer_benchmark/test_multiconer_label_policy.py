"""Hand-checked label policy for the MultiCoNER II place recognition comparison."""

import pytest

from scripts.multiconer_benchmark.label_policy import (
    DOCUMENTED_LABELS,
    KNOWN_LABELS,
    LOADER_ONLY_LABELS,
    PLACE_LABELS,
    UnknownLabelError,
    label_for,
    label_mapping,
)


def test_place_labels_are_the_four_coarse_location_types():
    assert {"Facility", "OtherLOC", "HumanSettlement", "Station"} == PLACE_LABELS


def test_documented_tagset_has_thirty_three_fine_types():
    assert len(DOCUMENTED_LABELS) == 33
    assert {"SportsGRP", "PublicCORP", "ORG", "Disease"} <= set(DOCUMENTED_LABELS)


def test_loader_only_labels_are_kept_apart_from_the_documented_tagset():
    assert {
        "OtherCW",
        "OtherCorp",
        "TechCORP",
        "PublicCorp",
        "PrivateCorp",
    } == LOADER_ONLY_LABELS
    assert not LOADER_ONLY_LABELS & set(DOCUMENTED_LABELS)
    assert len(KNOWN_LABELS) == 38


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("HumanSettlement", "LOC"),
        ("Facility", "LOC"),
        ("Station", "LOC"),
        ("OtherLOC", "LOC"),
        ("SportsGRP", "ignore"),
        ("PublicCORP", "ignore"),
        ("PublicCorp", "ignore"),
        ("Politician", "ignore"),
        ("Disease", "ignore"),
    ],
)
def test_place_versus_non_place_policy(label, expected):
    assert label_for(label) == expected


def test_unknown_or_case_altered_labels_are_rejected_not_guessed():
    with pytest.raises(UnknownLabelError):
        label_for("Planet")
    with pytest.raises(UnknownLabelError):
        label_for("humansettlement")


def test_label_mapping_covers_every_known_label_and_nothing_else():
    mapping = label_mapping()
    assert set(mapping) == KNOWN_LABELS
    assert set(mapping.values()) == {"LOC", "ignore"}
    assert sum(value == "LOC" for value in mapping.values()) == 4
