"""
Tests for looking up Wikidata coordinates.

The lookup is cached to a file that is committed alongside the harness, so a
benchmark run on a node without Wikidata access still scores the same spans.
"""

import json

from scripts.benchmark.wikidata import coordinates_from_entities, load_coordinates


def entity(qid, latitude=None, longitude=None):
    """Return a minimal wbgetentities entity."""
    claims = {}
    if latitude is not None:
        claims["P625"] = [
            {
                "mainsnak": {
                    "datavalue": {
                        "value": {"latitude": latitude, "longitude": longitude}
                    }
                }
            }
        ]
    return {"id": qid, "claims": claims}


class TestCoordinatesFromEntities:
    """Reading P625 out of an API response."""

    def test_reads_the_first_coordinate_claim(self):
        """Test that a located item yields its coordinates."""
        response = {"entities": {"Q90": entity("Q90", 48.85, 2.35)}}

        assert coordinates_from_entities(response) == {"Q90": (48.85, 2.35)}

    def test_records_an_unlocated_item_as_none(self):
        """Test that an item without P625 is cached as known-absent."""
        response = {"entities": {"Q5": entity("Q5")}}

        assert coordinates_from_entities(response) == {"Q5": None}

    def test_skips_a_claim_without_a_value(self):
        """Test that a 'somevalue' claim is not a crash."""
        response = {
            "entities": {"Q7": {"id": "Q7", "claims": {"P625": [{"mainsnak": {}}]}}}
        }

        assert coordinates_from_entities(response) == {"Q7": None}


class TestLoadCoordinates:
    """Cache-first lookup."""

    def test_fetches_only_what_the_cache_lacks(self, tmp_path):
        """Test that cached items are not asked for again."""
        cache = tmp_path / "coords.json"
        cache.write_text(json.dumps({"Q90": [48.85, 2.35]}))
        asked = []

        def fetch(qids):
            asked.append(sorted(qids))
            return dict.fromkeys(qids, (1.0, 2.0))

        result = load_coordinates({"Q90", "Q71"}, cache, fetch=fetch)

        assert asked == [["Q71"]]
        assert result == {"Q90": (48.85, 2.35), "Q71": (1.0, 2.0)}

    def test_persists_fetched_items_including_absent_ones(self, tmp_path):
        """Test that a second run needs no network at all."""
        cache = tmp_path / "coords.json"
        load_coordinates(
            {"Q1", "Q2"}, cache, fetch=lambda qids: {"Q1": (1.0, 2.0), "Q2": None}
        )

        def fail(qids):
            raise AssertionError("should not fetch")

        assert load_coordinates({"Q1", "Q2"}, cache, fetch=fail) == {"Q1": (1.0, 2.0)}

    def test_batches_requests(self, tmp_path):
        """Test that the API's 50-item limit is respected."""
        sizes = []

        def fetch(qids):
            sizes.append(len(qids))
            return dict.fromkeys(qids)

        load_coordinates(
            {f"Q{i}" for i in range(120)}, tmp_path / "c.json", fetch=fetch
        )

        assert sorted(sizes) == [20, 50, 50]
