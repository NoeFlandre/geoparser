"""The project and annotator databases create their engines on first use."""

from pathlib import Path
from types import SimpleNamespace

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
        database_file = tmp_path / "nested" / "deeper" / "p.db"
        monkeypatch.setattr(project_db, "_engine", None)
        monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database_file}")

        engine = project_db.get_engine()

        try:
            assert project_db.get_engine() is engine
            assert Path(str(engine.url.database)) == database_file
            assert database_file.parent.is_dir()
        finally:
            engine.dispose()

    def test_former_database_url_name_resolves_lazily(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "sqlite:///lazy.db")

        assert project_db.__getattr__("DATABASE_URL") == "sqlite:///lazy.db"

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


class _RacingLock:
    """A lock whose holder finds another thread already created the engine."""

    def __init__(self, module, engine):
        self.module = module
        self.engine = engine

    def __enter__(self):
        self.module._engine = self.engine

    def __exit__(self, *args):
        return None


@pytest.mark.unit
class TestProjectEngineCreation:
    def test_engine_is_configured_for_sqlite_threads_without_pooling(
        self, monkeypatch, tmp_path
    ):
        from sqlalchemy.pool import NullPool

        created = object()
        calls = []
        database_file = tmp_path / "existing" / "p.db"
        database_file.parent.mkdir()
        monkeypatch.setattr(project_db, "_engine", None)
        monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database_file}")
        monkeypatch.setattr(
            project_db,
            "create_engine",
            lambda *args, **kwargs: calls.append((args, kwargs)) or created,
        )

        assert project_db.get_engine() is created
        assert project_db.get_engine() is created
        assert calls == [
            (
                (f"sqlite:///{database_file}",),
                {
                    "echo": False,
                    "connect_args": {"check_same_thread": False},
                    "poolclass": NullPool,
                },
            )
        ]

    def test_non_file_urls_create_no_directory(self, monkeypatch, tmp_path):
        monkeypatch.setattr(project_db, "_engine", None)
        monkeypatch.setenv("DATABASE_URL", "sqlite://")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(project_db, "create_engine", lambda *a, **k: object())

        project_db.get_engine()

        assert list(tmp_path.iterdir()) == []

    def test_a_concurrently_created_engine_is_reused(self, monkeypatch):
        winner = object()
        monkeypatch.setattr(project_db, "_engine", None)
        monkeypatch.setattr(project_db, "_engine_lock", _RacingLock(project_db, winner))
        monkeypatch.setattr(
            project_db,
            "create_engine",
            lambda *a, **k: pytest.fail("the engine must not be created twice"),
        )

        assert project_db.get_engine() is winner


@pytest.mark.unit
class TestAnnotatorEngineCreation:
    def test_engine_lives_under_the_data_directory_with_foreign_keys(
        self, monkeypatch, tmp_path
    ):
        from sqlalchemy import text

        data_dir = tmp_path / "not-yet-created"
        monkeypatch.setattr(annotator_db, "_engine", None)
        monkeypatch.setenv("GEOPARSER_DATA_DIR", str(data_dir))

        engine = annotator_db.get_engine()
        try:
            assert annotator_db.get_engine() is engine
            assert Path(str(engine.url.database)) == (
                data_dir / "annotator" / "annotator.db"
            )
            assert engine.echo is False
            with engine.connect() as connection:
                assert connection.execute(text("PRAGMA foreign_keys")).scalar() == 1
        finally:
            engine.dispose()

    def test_an_existing_directory_is_reused(self, monkeypatch, tmp_path):
        (tmp_path / "annotator").mkdir()
        monkeypatch.setattr(annotator_db, "_engine", None)
        monkeypatch.setenv("GEOPARSER_DATA_DIR", str(tmp_path))

        annotator_db.get_engine().dispose()

    def test_a_concurrently_created_engine_is_reused(self, monkeypatch):
        winner = object()
        monkeypatch.setattr(annotator_db, "_engine", None)
        monkeypatch.setattr(
            annotator_db, "_engine_lock", _RacingLock(annotator_db, winner)
        )
        monkeypatch.setattr(
            annotator_db,
            "create_engine",
            lambda *a, **k: pytest.fail("the engine must not be created twice"),
        )

        assert annotator_db.get_engine() is winner


@pytest.mark.unit
class TestDatabaseLookups:
    def test_database_path_prefers_a_patched_engine(self, monkeypatch):
        from sqlalchemy.engine import make_url

        monkeypatch.setitem(
            vars(project_db),
            "engine",
            SimpleNamespace(url=make_url("sqlite:////patched/geoparser.db")),
        )

        assert project_db.get_database_path() == Path("/patched/geoparser.db")

    def test_database_path_reads_the_created_engine(self, monkeypatch):
        from sqlalchemy.engine import make_url

        monkeypatch.setattr(
            project_db,
            "_engine",
            SimpleNamespace(url=make_url("sqlite:////created/geoparser.db")),
        )
        monkeypatch.setenv("DATABASE_URL", "sqlite:////configured/other.db")

        assert project_db.get_database_path() == Path("/created/geoparser.db")

    def test_db_path_alias_is_a_string_or_none(self, monkeypatch):
        from sqlalchemy.engine import make_url

        engine = SimpleNamespace(url=make_url("sqlite:////x/geoparser.db"))
        monkeypatch.setattr(project_db, "_engine", engine)
        assert project_db.__getattr__("db_path") == str(Path("/x/geoparser.db"))

        engine.url = make_url("sqlite://")
        assert project_db.__getattr__("db_path") is None

    def test_annotator_sessions_are_bound_to_its_engine(self, monkeypatch):
        engine = object()
        monkeypatch.setattr(annotator_db, "get_engine", lambda: engine)

        bound = [session.bind for session in annotator_db.get_db()]

        assert bound == [engine]
