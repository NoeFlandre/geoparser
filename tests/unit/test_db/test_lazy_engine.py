"""The project and annotator databases create their engines on first use."""

from pathlib import Path

import pytest

import geoparser.annotator.db.db as annotator_db
import geoparser.db.db as project_db


@pytest.mark.unit
class TestProjectDatabaseConfiguration:
    def test_database_url_prefers_the_environment(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "sqlite:///override.db")

        assert project_db._database_url() == "sqlite:///override.db"

    def test_database_url_defaults_to_the_data_directory(self, monkeypatch, tmp_path):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.setenv("GEOPARSER_DATA_DIR", str(tmp_path))

        assert project_db._database_url() == f"sqlite:///{tmp_path / 'geoparser.db'}"

    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("postgresql://localhost/geoparser", None),
            ("sqlite://", None),
            ("sqlite:///:memory:", None),
            ("sqlite:///file:shared?mode=memory", None),
            ("sqlite:////data/geoparser.db", Path("/data/geoparser.db")),
        ],
    )
    def test_only_file_backed_sqlite_urls_have_a_path(self, url, expected):
        assert project_db._sqlite_file_path(url) == expected

    def test_database_path_without_an_engine_reads_the_configured_url(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(project_db, "_engine", None)
        monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'p.db'}")

        assert project_db.get_database_path() == tmp_path / "p.db"

    def test_engine_is_created_once_with_its_parent_directory(
        self, monkeypatch, tmp_path
    ):
        database_file = tmp_path / "nested" / "p.db"
        monkeypatch.setattr(project_db, "_engine", None)
        monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database_file}")

        engine = project_db.get_engine()

        try:
            assert project_db.get_engine() is engine
            assert Path(str(engine.url.database)) == database_file
            assert database_file.parent.is_dir()
        finally:
            engine.dispose()

    def test_unknown_module_attributes_raise(self):
        with pytest.raises(AttributeError, match="no_such_name"):
            project_db.__getattr__("no_such_name")


@pytest.mark.unit
class TestAnnotatorDatabaseConfiguration:
    def test_a_patched_engine_takes_precedence(self, monkeypatch):
        sentinel = object()
        monkeypatch.setitem(vars(annotator_db), "engine", sentinel)

        assert annotator_db.get_engine() is sentinel

    def test_former_module_names_resolve_lazily(self, monkeypatch, tmp_path):
        monkeypatch.setenv("GEOPARSER_DATA_DIR", str(tmp_path))
        location = tmp_path / "annotator" / "annotator.db"
        sentinel = object()
        monkeypatch.setattr(annotator_db, "get_engine", lambda: sentinel)

        assert annotator_db.__getattr__("engine") is sentinel
        assert annotator_db.__getattr__("db_location") == location
        assert annotator_db.__getattr__("sqlite_url") == f"sqlite:///{location}"
        with pytest.raises(AttributeError, match="no_such_name"):
            annotator_db.__getattr__("no_such_name")
