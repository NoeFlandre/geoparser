"""Candidate mapping, against values worked out by hand from the fixtures below."""

import math

from scripts.unitoprank_benchmark.candidates import (
    Candidate,
    build_candidate_set,
    normalize_surface,
)

# Two real Parises as GeoNames records would describe them. Identifiers are
# ordered as strings: "2988507" sorts before "4717560".
PARIS_FR = Candidate(
    identifier="2988507",
    name="Paris",
    latitude=48.8566,
    longitude=2.3522,
    feature_code="PPLC",
    population=2140526,
    admin_path=("Ile-de-France", "France", "Europe"),
    alternate_names=("Parigi", "París", "Parijs"),
)
PARIS_TX = Candidate(
    identifier="4717560",
    name="Paris",
    latitude=33.6609,
    longitude=-95.5555,
    feature_code="PPLA2",
    population=24719,
    admin_path=("Lamar County", "Texas", "United States", "North America"),
)


def test_output_does_not_depend_on_the_order_the_gazetteer_returned():
    forward = build_candidate_set({"Paris": [PARIS_FR, PARIS_TX]})
    backward = build_candidate_set({"Paris": [PARIS_TX, PARIS_FR]})
    assert forward.by_surface == backward.by_surface
    assert forward.identifiers == backward.identifiers
    assert [entry["address"] for entry in forward.by_surface["paris"]] == [
        "Paris, Ile-de-France, France, Europe",
        "Paris, Lamar County, Texas, United States, North America",
    ]


def test_entry_has_the_fields_the_ranker_reads():
    entry = build_candidate_set({"Paris": [PARIS_FR]}).by_surface["paris"][0]
    assert entry == {
        "address": "Paris, Ile-de-France, France, Europe",
        "lat": 48.8566,
        "lon": 2.3522,
        "name": "Paris",
        "alt_names": ["Parigi", "Parijs", "París"],
        "population": 2140526,
        "admin_level": "PPLC",
    }


def test_surface_forms_that_normalize_alike_share_one_list():
    result = build_candidate_set({"Paris": [PARIS_TX], " paris ": [PARIS_FR]})
    assert list(result.by_surface) == ["paris"]
    assert [entry["name"] for entry in result.by_surface["paris"]] == [
        "Paris",
        "Paris",
    ]
    assert result.identifiers["paris"] == {
        "Paris, Ile-de-France, France, Europe": "2988507",
        "Paris, Lamar County, Texas, United States, North America": "4717560",
    }


def test_normalize_surface_strips_and_lowercases():
    assert normalize_surface("  Bâle ") == "bâle"


def test_repeated_identifier_is_kept_once_and_counted():
    result = build_candidate_set({"Paris": [PARIS_FR, PARIS_FR]})
    assert len(result.by_surface["paris"]) == 1
    assert result.dropped["duplicate_identifier"] == 1


def test_missing_population_becomes_zero_and_is_counted():
    none = Candidate("1", "Anywhere", 10.0, 10.0, "PPL", population=None)
    negative = Candidate("2", "Elsewhere", 10.0, 10.0, "PPL", population=-5)
    not_a_number = Candidate("3", "Nowhere", 10.0, 10.0, "PPL", population=math.nan)
    result = build_candidate_set({"x": [none, negative, not_a_number]})
    assert [entry["population"] for entry in result.by_surface["x"]] == [0, 0, 0]
    assert result.missing["population"] == 3


def test_missing_admin_levels_are_left_out_of_the_address_and_counted():
    candidate = Candidate(
        "7",
        "Paris",
        33.6609,
        -95.5555,
        "PPLA2",
        population=24719,
        admin_path=("Lamar County", None, "", "United States"),
    )
    result = build_candidate_set({"Paris": [candidate]})
    assert result.by_surface["paris"][0]["address"] == (
        "Paris, Lamar County, United States"
    )
    assert result.missing["admin_level"] == 2
    assert "admin_path" not in result.missing


def test_no_admin_path_at_all_is_counted_separately():
    candidate = Candidate("8", "Paris", 33.0, -95.0, "PPLA2", population=1)
    result = build_candidate_set({"Paris": [candidate]})
    assert result.by_surface["paris"][0]["address"] == "Paris"
    assert result.missing["admin_path"] == 1
    assert result.missing["admin_level"] == 0


def test_missing_feature_code_becomes_empty_and_is_counted():
    candidate = Candidate("9", "Paris", 33.0, -95.0, None, population=1)
    result = build_candidate_set({"Paris": [candidate]})
    assert result.by_surface["paris"][0]["admin_level"] == ""
    assert result.missing["feature_code"] == 1


def test_candidates_sharing_an_address_keep_the_lower_identifier():
    later = Candidate("11", "Springfield", 39.8, -89.6, "PPLA2", 100, ("Illinois",))
    earlier = Candidate("10", "Springfield", 39.8, -89.6, "PPLA2", 200, ("Illinois",))
    result = build_candidate_set({"Springfield": [later, earlier]})
    assert [entry["population"] for entry in result.by_surface["springfield"]] == [200]
    assert result.identifiers["springfield"] == {"Springfield, Illinois": "10"}
    assert result.dropped["address_collision"] == 1


def test_candidates_without_usable_coordinates_are_dropped():
    no_latitude = Candidate("1", "Paris", None, 2.35, "PPLC", 1)
    too_far_north = Candidate("2", "Paris", 91.0, 2.35, "PPLC", 1)
    not_finite = Candidate("3", "Paris", 48.85, math.nan, "PPLC", 1)
    usable = Candidate("4", "Paris", 48.85, 2.35, "PPLC", 1)
    result = build_candidate_set(
        {"Paris": [no_latitude, too_far_north, not_finite, usable]}
    )
    assert [entry["address"] for entry in result.by_surface["paris"]] == ["Paris"]
    assert result.dropped["no_coordinates"] == 3


def test_blank_name_is_dropped():
    blank = Candidate("5", "   ", 48.85, 2.35, "PPLC", 1)
    result = build_candidate_set({"Paris": [blank]})
    assert result.by_surface["paris"] == []
    assert result.dropped["no_name"] == 1


def test_alternate_names_are_stripped_deduplicated_and_sorted():
    candidate = Candidate(
        "6",
        "Paris",
        48.85,
        2.35,
        "PPLC",
        1,
        alternate_names=("París", " ", "Parigi", "Parigi"),
    )
    entry = build_candidate_set({"Paris": [candidate]}).by_surface["paris"][0]
    assert entry["alt_names"] == ["Parigi", "París"]


def test_surface_with_no_candidates_has_an_empty_list():
    result = build_candidate_set({"Nowhere": []})
    assert result.by_surface == {"nowhere": []}
    assert result.identifiers == {"nowhere": {}}
