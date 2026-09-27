"""Tests for the shared application data directory."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_NAMESPACE_PACKAGE = textwrap.dedent(
    """
import sys
import types
from pathlib import Path

package = types.ModuleType("geoparser")
package.__path__ = [str(Path.cwd() / "geoparser")]
package.__package__ = "geoparser"
sys.modules["geoparser"] = package
"""
)


def _run_python(
    source: str, *, env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    """Run a short script against this checkout in an isolated interpreter."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        check=False,
        capture_output=True,
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        text=True,
    )


@pytest.mark.unit
def test_data_directory_defaults_to_platform_application_data():
    """The override can be absent and preserve the platformdirs default."""
    env = os.environ.copy()
    env.pop("GEOPARSER_DATA_DIR", None)

    result = _run_python(
        _NAMESPACE_PACKAGE
        + textwrap.dedent(
            """
            from pathlib import Path

            from geoparser import paths

            paths.user_data_dir = lambda appname, appauthor: "/default/geoparser"
            assert paths.geoparser_data_dir() == Path("/default/geoparser")
            """
        ),
        env=env,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.unit
def test_data_directory_override_is_shared_by_gazetteers_and_databases(tmp_path):
    """The configured root controls artifact and both database paths."""
    env = os.environ.copy()
    data_dir = tmp_path / "geoparser-data"
    env["GEOPARSER_DATA_DIR"] = str(data_dir)
    env.pop("GEOPARSER_GAZETTEERS_DIR", None)
    env.pop("DATABASE_URL", None)

    result = _run_python(
        _NAMESPACE_PACKAGE
        + textwrap.dedent(
            """
        import importlib
        import os
        from pathlib import Path

        data_dir = Path(os.environ["GEOPARSER_DATA_DIR"])
        from geoparser.paths import geoparser_data_dir
        from geoparser.gazetteer.artifact import gazetteers_dir

        assert geoparser_data_dir() == data_dir
        assert gazetteers_dir() == data_dir / "gazetteers"

        project_db = importlib.import_module("geoparser.db.db")
        project_db.create_db_and_tables()
        assert Path(project_db.get_engine().url.database) == data_dir / "geoparser.db"

        importlib.import_module("geoparser.annotator.db.models")
        annotator_db = importlib.import_module("geoparser.annotator.db.db")
        annotator_db.create_db_and_tables()
        assert (
            Path(annotator_db.get_engine().url.database)
            == data_dir / "annotator" / "annotator.db"
        )

        assert (data_dir / "geoparser.db").is_file()
        assert (data_dir / "annotator" / "annotator.db").is_file()
        """
        ),
        env=env,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.unit
@pytest.mark.parametrize(
    "module_name",
    ["geoparser.db.db", "geoparser.annotator.db.db"],
)
def test_database_modules_do_not_read_data_configuration_at_import(
    module_name: str, tmp_path: Path
):
    """Importing either DB module does not read env vars or resolve a path."""
    env = os.environ.copy()
    env["GEOPARSER_DATA_DIR"] = str(tmp_path / "unused-data")
    env["DATABASE_URL"] = f"sqlite:///{tmp_path / 'unused.db'}"
    env["GEOPARSER_DB_MODULE"] = module_name

    result = _run_python(
        _NAMESPACE_PACKAGE
        + textwrap.dedent(
            """
        import importlib
        import os
        import sys

        from geoparser import paths

        def reject_path_resolution():
            raise AssertionError("data directory resolved during DB module import")

        paths.geoparser_data_dir = reject_path_resolution
        original_getenv = os.getenv

        def reject_database_configuration(name, *args, **kwargs):
            if name in {"DATABASE_URL", "GEOPARSER_DATA_DIR"}:
                raise AssertionError(f"{name} read during DB module import")
            return original_getenv(name, *args, **kwargs)

        os.getenv = reject_database_configuration
        importlib.import_module(os.environ["GEOPARSER_DB_MODULE"])
        """
        ),
        env=env,
    )

    assert result.returncode == 0, result.stderr
