"""
Integration tests for the gazetteer build pipeline.

Builds real artifacts from the Andorra fixture data and from small inline
configs (covering duplicate-identifier merging), and inspects the resulting
SQLite files through the runtime layer.
"""

import sqlite3
import textwrap

import pytest

from geoparser.gazetteer.artifact import GazetteerArtifact, artifact_path
from geoparser.gazetteer.build import GazetteerBuilder
from geoparser.gazetteer.build.builder import uninstall
from geoparser.gazetteer.gazetteer import Gazetteer


@pytest.mark.integration
class TestAndorraBuild:
    """Test the artifact built from the Andorra fixture config."""

    def test_artifact_is_installed(self, andorra_gazetteer):
        """The build produces an artifact in the gazetteers directory."""
        assert artifact_path("andorranames").exists()

    def test_artifact_metadata(self, andorra_gazetteer):
        """The artifact carries its name, CRS and counts as metadata."""
        artifact = GazetteerArtifact(artifact_path("andorranames"))

        assert artifact.name == "andorranames"
        assert artifact.crs == "EPSG:4326"
        assert int(artifact.metadata["feature_count"]) == artifact.count_features()
        assert int(artifact.metadata["name_count"]) == artifact.count_names()

    def test_features_and_names_are_populated(self, andorra_gazetteer):
        """The Andorra fixture produces a meaningful number of rows."""
        artifact = GazetteerArtifact(artifact_path("andorranames"))

        assert artifact.count_features() > 100
        assert artifact.count_names() > artifact.count_features()

    def test_identifiers_are_unique(self, andorra_gazetteer):
        """Every feature has a unique identifier."""
        connection = sqlite3.connect(artifact_path("andorranames"))
        try:
            total, distinct = connection.execute(
                "SELECT count(*), count(DISTINCT identifier) FROM feature"
            ).fetchone()
        finally:
            connection.close()

        assert total == distinct

    def test_split_names_are_registered(self, andorra_gazetteer):
        """Alternate names from the comma-separated column are searchable."""
        gazetteer = Gazetteer("andorranames")

        # "Andorre-la-Vieille" is an alternate name of Andorra la Vella
        results = gazetteer.search("Andorre-la-Vieille", method="exact")

        assert any(feature.identifier == "3041563" for feature in results)

    def test_expression_names_are_registered(self, andorra_gazetteer):
        """Names derived via expressions (parenthesis stripping) are searchable."""
        gazetteer = Gazetteer("andorranames")

        features_with_parens = _expression_features(gazetteer)
        assert features_with_parens
        for feature in features_with_parens:
            stripped = feature.data["name"].split("(")[0].strip()
            assert stripped in feature.names

    def test_spatial_lookup_assigns_shape(self, andorra_gazetteer):
        """The spatial lookup tags features inside the Andorra boundary."""
        gazetteer = Gazetteer("andorranames")

        feature = gazetteer.find("3041563")  # Andorra la Vella
        assert feature is not None

        assert feature.data["shape_fid"] is not None

    def test_uninstall_removes_artifact(
        self, andorra_config_path, tmp_path, monkeypatch
    ):
        """Uninstalling deletes the artifact file."""
        monkeypatch.setenv("GEOPARSER_GAZETTEERS_DIR", str(tmp_path))
        GazetteerBuilder().build(andorra_config_path)
        assert artifact_path("andorranames").exists()

        assert uninstall("andorranames") is True
        assert not artifact_path("andorranames").exists()
        assert uninstall("andorranames") is False


def _expression_features(gazetteer):
    """Select search results whose original source names contain expressions."""
    return [
        feature
        for feature in gazetteer.search("General", method="partial", tiers=3)
        if feature is not None and "(" in feature.data["name"]
    ]


@pytest.mark.integration
class TestDuplicateIdentifierMerge:
    """Test merging of rows sharing an identifier."""

    @pytest.fixture
    def duplicates_config(self, tmp_path):
        """A config whose source repeats identifiers across rows."""
        data_file = tmp_path / "peaks.csv"
        data_file.write_text(
            "p1\tNorth Summit\t800\np1\tSouth Summit\t1200\np2\tLone Hill\t300\n"
        )
        config_file = tmp_path / "peaks.yaml"
        config_file.write_text(
            textwrap.dedent(
                """
                name: peaks
                sources:
                  - name: peaks
                    path: peaks.csv
                    file: peaks.csv
                    delimiter: "\\t"
                    quote: ""
                    attributes:
                      - name: pid
                        type: text
                      - name: name
                        type: text
                      - name: height
                        type: integer
                features:
                  - source: peaks
                    identifier: "pid"
                    names:
                      - "name"
                    data:
                      - "name"
                      - "height"
                """
            )
        )
        return config_file

    @pytest.fixture
    def merged_duplicate_identifier(self, duplicates_config, tmp_path, monkeypatch):
        """Build the duplicate rows once and expose their merged feature."""
        monkeypatch.setenv("GEOPARSER_GAZETTEERS_DIR", str(tmp_path / "gazetteers"))
        GazetteerBuilder().build(duplicates_config)
        gazetteer = Gazetteer("peaks")
        merged = gazetteer.find("p1")
        assert merged is not None
        return gazetteer, merged

    def test_duplicate_identifiers_collect_their_names(
        self, merged_duplicate_identifier
    ):
        _, merged = merged_duplicate_identifier
        assert set(merged.names) == {"North Summit", "South Summit"}

    @pytest.mark.parametrize(
        ("field", "expected"), [("name", "North Summit"), ("height", 800)]
    )
    def test_duplicate_identifiers_keep_first_row_data(
        self, merged_duplicate_identifier, field, expected
    ):
        _, merged = merged_duplicate_identifier
        assert merged.data[field] == expected

    @pytest.mark.parametrize("name", ("North Summit", "South Summit"))
    def test_duplicate_names_both_search_the_merged_feature(
        self, merged_duplicate_identifier, name
    ):
        gazetteer, _ = merged_duplicate_identifier
        matches = gazetteer.search(name, method="exact")
        assert [feature.identifier for feature in matches] == ["p1"]

    def test_artifact_has_one_row_per_identifier(
        self, duplicates_config, tmp_path, monkeypatch
    ):
        """The artifact stores exactly one feature row per identifier."""
        monkeypatch.setenv("GEOPARSER_GAZETTEERS_DIR", str(tmp_path / "gazetteers"))
        GazetteerBuilder().build(duplicates_config)

        artifact = GazetteerArtifact(artifact_path("peaks"))

        assert artifact.count_features() == 2


@pytest.mark.integration
class TestDuplicateGeometryMerge:
    """Test that duplicate identifiers with geometry get unioned geometries."""

    @pytest.fixture
    def duplicate_geometry_config(self, tmp_path):
        """A config whose source repeats an identifier with different points."""
        data_file = tmp_path / "points.csv"
        data_file.write_text(
            "p1\tNorth Point\t1.0\t1.0\n"
            "p1\tSouth Point\t2.0\t2.0\n"
            "p2\tLone Point\t3.0\t3.0\n"
        )
        config_file = tmp_path / "points.yaml"
        config_file.write_text(
            textwrap.dedent(
                """
                name: points
                sources:
                  - name: points
                    path: points.csv
                    file: points.csv
                    delimiter: "\\t"
                    quote: ""
                    attributes:
                      - name: pid
                        type: text
                      - name: name
                        type: text
                      - name: lon
                        type: real
                      - name: lat
                        type: real
                features:
                  - source: points
                    identifier: "pid"
                    geometry: "ST_Point(lon, lat)"
                    names:
                      - "name"
                """
            )
        )
        return config_file

    @pytest.fixture
    def built_point_features(self, duplicate_geometry_config, tmp_path, monkeypatch):
        """Build two identifiers whose geometry cardinalities differ."""
        monkeypatch.setenv("GEOPARSER_GAZETTEERS_DIR", str(tmp_path / "gazetteers"))
        GazetteerBuilder().build(duplicate_geometry_config)
        gazetteer = Gazetteer("points")
        merged = gazetteer.find("p1")
        assert merged is not None
        single = gazetteer.find("p2")
        assert single is not None
        return merged, single

    def test_duplicate_geometries_are_unioned(self, built_point_features):
        """Duplicate rows' points are merged into one multi-point geometry."""
        merged, _ = built_point_features
        assert merged.geometry is not None
        assert merged.geometry.geom_type == "MultiPoint"
        assert {(point.x, point.y) for point in merged.geometry.geoms} == {
            (1.0, 1.0),
            (2.0, 2.0),
        }

    def test_single_geometry_remains_a_point(self, built_point_features):
        _, single = built_point_features
        assert single.geometry is not None
        assert single.geometry.geom_type == "Point"
        assert (single.geometry.x, single.geometry.y) == (3.0, 3.0)


@pytest.mark.integration
class TestCrossBlockIdentifierCollision:
    """Test that identifier collisions across feature blocks fail the build."""

    def test_build_fails_with_clear_error(self, tmp_path, monkeypatch):
        """The same identifier across two feature blocks aborts the build."""
        data_file = tmp_path / "rows.csv"
        data_file.write_text("x1\tSomething\n")
        config_file = tmp_path / "clash.yaml"
        config_file.write_text(
            textwrap.dedent(
                """
                name: clash
                sources:
                  - name: rows_a
                    path: rows.csv
                    file: rows.csv
                    delimiter: "\\t"
                    quote: ""
                    attributes:
                      - name: rid
                        type: text
                      - name: name
                        type: text
                  - name: rows_b
                    path: rows.csv
                    file: rows.csv
                    delimiter: "\\t"
                    quote: ""
                    attributes:
                      - name: rid
                        type: text
                      - name: name
                        type: text
                features:
                  - source: rows_a
                    identifier: "rid"
                    names:
                      - "name"
                  - source: rows_b
                    identifier: "rid"
                    names:
                      - "name"
                """
            )
        )
        monkeypatch.setenv("GEOPARSER_GAZETTEERS_DIR", str(tmp_path / "gazetteers"))

        with pytest.raises(ValueError, match="unique across the whole gazetteer"):
            GazetteerBuilder().build(config_file)

        assert not artifact_path("clash").exists()
