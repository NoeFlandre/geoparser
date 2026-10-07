"""
Unit tests for geoparser/gazetteer/artifact.py

Tests the artifact-level helpers and the GazetteerArtifact class directly,
independent of the higher-level Gazetteer query interface.
"""

import pytest

from geoparser.gazetteer import artifact as artifact_module
from geoparser.gazetteer.artifact import GazetteerArtifact, artifact_path


@pytest.mark.unit
class TestGazetteersDir:
    """Test the gazetteers_dir() helper."""

    def test_uses_override_env_var(self, tmp_path, monkeypatch):
        """The GEOPARSER_GAZETTEERS_DIR env var overrides the default location."""
        monkeypatch.setenv("GEOPARSER_GAZETTEERS_DIR", str(tmp_path))

        assert artifact_module.gazetteers_dir() == tmp_path

    def test_defaults_to_user_data_dir(self, monkeypatch):
        """Without an override, the user data directory is used."""
        monkeypatch.delenv("GEOPARSER_GAZETTEERS_DIR", raising=False)

        result = artifact_module.gazetteers_dir()

        assert result.name == "gazetteers"
        assert "geoparser" in str(result)


@pytest.mark.unit
class TestListArtifacts:
    """Test the list_artifacts() helper."""

    def test_returns_empty_list_when_directory_missing(self, tmp_path, monkeypatch):
        """A missing gazetteers directory yields no artifacts."""
        monkeypatch.setenv("GEOPARSER_GAZETTEERS_DIR", str(tmp_path / "missing"))

        assert artifact_module.list_artifacts() == []

    def test_returns_sorted_artifact_names(self, tmp_path, monkeypatch):
        """Installed artifacts are returned sorted by name, without extension."""
        monkeypatch.setenv("GEOPARSER_GAZETTEERS_DIR", str(tmp_path))
        (tmp_path / f"zebra{artifact_module.ARTIFACT_SUFFIX}").touch()
        (tmp_path / f"alpha{artifact_module.ARTIFACT_SUFFIX}").touch()
        (tmp_path / "not-an-artifact.txt").touch()
        (tmp_path / "subdirectory").mkdir()

        assert artifact_module.list_artifacts() == ["alpha", "zebra"]


@pytest.mark.unit
class TestGazetteerArtifactInitialization:
    """Test GazetteerArtifact.__init__()."""

    def test_raises_when_file_missing(self, tmp_path):
        """Opening a non-existent artifact file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            GazetteerArtifact(tmp_path / "missing.db")

    def test_raises_when_file_is_not_a_database(self, tmp_path):
        """A file that isn't a valid SQLite database raises a clear RuntimeError."""
        path = tmp_path / "broken.db"
        path.write_text("not a sqlite database")

        with pytest.raises(RuntimeError, match="not a valid gazetteer artifact"):
            GazetteerArtifact(path)


@pytest.mark.unit
class TestGazetteerArtifactClose:
    """Test GazetteerArtifact.close()."""

    def test_close_after_use_releases_connection(self, make_artifact):
        """Closing after querying releases the thread-local connection."""
        make_artifact(name="testgaz")
        artifact = GazetteerArtifact(artifact_path("testgaz"))
        artifact.find("1")

        artifact.close()

        assert getattr(artifact._local, "connection", None) is None

    def test_close_without_prior_use_is_a_no_op(self, make_artifact):
        """Closing an artifact that never opened a connection does nothing."""
        make_artifact(name="testgaz")
        artifact = GazetteerArtifact(artifact_path("testgaz"))

        artifact.close()

        assert getattr(artifact._local, "connection", None) is None

    def test_reopens_a_new_connection_after_close(self, make_artifact):
        """Querying again after close() transparently opens a new connection."""
        make_artifact(name="testgaz")
        artifact = GazetteerArtifact(artifact_path("testgaz"))
        artifact.find("1")
        artifact.close()

        feature = artifact.find("1")

        assert feature is not None


@pytest.mark.unit
class TestGazetteerArtifactCounts:
    """Test artifact-level row counts."""

    def test_counts_names(self, make_artifact):
        """The count reflects all searchable names, including alternate names."""
        make_artifact(name="counted")
        artifact = GazetteerArtifact(artifact_path("counted"))
        try:
            assert artifact.count_names() == 4
        finally:
            artifact.close()


@pytest.mark.unit
class TestGazetteerNameBoundary:
    @pytest.mark.parametrize(
        "name",
        [
            "",
            ".",
            "..",
            "../../notes",
            r"..\..\notes",
            "/tmp/notes",
            r"C:\notes",
            "C:notes",
            r"\\server\share\notes",
            "nested/name",
            r"nested\name",
            "name\n",
            "two words",
            "café",
            "name\x00",
        ],
    )
    def test_rejects_names_outside_the_supported_grammar(self, name):
        with pytest.raises(ValueError, match="must contain only"):
            artifact_path(name)

    @pytest.mark.parametrize(
        "name", ["geonames", "swissnames3d", "AZaz09_-", "0", "_", "-"]
    )
    def test_preserves_valid_names(self, name, tmp_path, monkeypatch):
        monkeypatch.setenv("GEOPARSER_GAZETTEERS_DIR", str(tmp_path))

        assert artifact_path(name) == tmp_path / f"{name}.db"

    @pytest.mark.parametrize("absolute", [False, True])
    def test_uninstall_keeps_outside_files(self, absolute, tmp_path, monkeypatch):
        from geoparser.gazetteer.build.builder import uninstall

        directory = tmp_path / "gazetteers"
        directory.mkdir()
        outside = tmp_path / "notes.db"
        outside.write_bytes(b"private notes")
        monkeypatch.setenv("GEOPARSER_GAZETTEERS_DIR", str(directory))
        name = str(outside.with_suffix("")) if absolute else "../notes"

        with pytest.raises(ValueError, match="must contain only"):
            uninstall(name)

        assert outside.read_bytes() == b"private notes"

    def test_gazetteer_rejects_outside_artifacts(self, tmp_path, monkeypatch):
        from geoparser.gazetteer import Gazetteer

        directory = tmp_path / "gazetteers"
        directory.mkdir()
        (tmp_path / "notes.db").write_bytes(b"private notes")
        monkeypatch.setenv("GEOPARSER_GAZETTEERS_DIR", str(directory))

        with pytest.raises(ValueError, match="must contain only"):
            Gazetteer("../notes")


@pytest.mark.unit
def test_list_artifacts_ignores_invalid_gazetteer_names(tmp_path, monkeypatch):
    monkeypatch.setenv("GEOPARSER_GAZETTEERS_DIR", str(tmp_path))
    for filename in ["valid-1.db", "bad.name.db", "two words.db", ".db"]:
        (tmp_path / filename).write_bytes(b"artifact")

    assert artifact_module.list_artifacts() == ["valid-1"]
