"""
Unit tests for geoparser/gazetteer/build/builder.py

Exercises GazetteerBuilder's internal helpers directly (memory sizing,
temp-directory handling, spatial extension loading) and runs small,
hand-crafted builds to trigger error paths (no features produced, a stale
leftover build directory) that are impractical to reach through the
public build() entry point alone.
"""

import os
import shutil
import textwrap
from contextlib import nullcontext
from unittest.mock import MagicMock

import duckdb
import pytest

from geoparser.gazetteer import artifact
from geoparser.gazetteer.build import builder as builder_module
from geoparser.gazetteer.build.builder import (
    GazetteerBuilder,
    _format_bytes,
    _sqlite_temp_env_names,
    _sqlite_tmpdir,
    uninstall,
)
from geoparser.gazetteer.build.schema import FeatureConfig, GazetteerConfig


@pytest.mark.unit
class TestSqliteTmpdir:
    """Test the _sqlite_tmpdir() context manager."""

    def test_unix_sets_and_restores_sqlite_tmpdir(self, monkeypatch, tmp_path):
        """On Unix, SQLITE_TMPDIR is set for the block and restored after."""
        monkeypatch.setattr(os, "name", "posix")
        monkeypatch.setenv("SQLITE_TMPDIR", "/original/tmpdir")

        with _sqlite_tmpdir(tmp_path):
            assert os.environ["SQLITE_TMPDIR"] == str(tmp_path)

        assert os.environ["SQLITE_TMPDIR"] == "/original/tmpdir"

    def test_unix_removes_var_when_none_was_set_before(self, monkeypatch, tmp_path):
        """On Unix, SQLITE_TMPDIR is removed again if it wasn't set beforehand."""
        monkeypatch.setattr(os, "name", "posix")
        monkeypatch.delenv("SQLITE_TMPDIR", raising=False)

        with _sqlite_tmpdir(tmp_path):
            assert os.environ["SQLITE_TMPDIR"] == str(tmp_path)

        assert "SQLITE_TMPDIR" not in os.environ

    def test_windows_sets_and_restores_tmp_and_temp(self, monkeypatch, tmp_path):
        """On Windows, TMP and TEMP are set (GetTempPath) and restored after."""
        monkeypatch.setattr(os, "name", "nt")
        monkeypatch.setenv("TMP", "C:\\original\\tmp")
        monkeypatch.setenv("TEMP", "C:\\original\\temp")
        monkeypatch.delenv("SQLITE_TMPDIR", raising=False)

        with _sqlite_tmpdir(tmp_path):
            assert (
                os.environ.get("TMP"),
                os.environ.get("TEMP"),
                "SQLITE_TMPDIR" in os.environ,
            ) == (str(tmp_path), str(tmp_path), False)

        assert (os.environ.get("TMP"), os.environ.get("TEMP")) == (
            "C:\\original\\tmp",
            "C:\\original\\temp",
        )

    def test_windows_removes_tmp_vars_when_none_were_set_before(
        self, monkeypatch, tmp_path
    ):
        """On Windows, TMP/TEMP are removed again if they weren't set beforehand."""
        monkeypatch.setattr(os, "name", "nt")
        monkeypatch.delenv("TMP", raising=False)
        monkeypatch.delenv("TEMP", raising=False)

        with _sqlite_tmpdir(tmp_path):
            assert os.environ["TMP"] == str(tmp_path)
            assert os.environ["TEMP"] == str(tmp_path)

        assert "TMP" not in os.environ
        assert "TEMP" not in os.environ

    def test_env_names_match_platform(self, monkeypatch):
        """Unix steers SQLITE_TMPDIR; Windows steers TMP/TEMP."""
        monkeypatch.setattr(os, "name", "posix")
        assert _sqlite_temp_env_names() == ("SQLITE_TMPDIR",)

        monkeypatch.setattr(os, "name", "nt")
        assert _sqlite_temp_env_names() == ("TMP", "TEMP")


@pytest.mark.unit
class TestMemoryLimit:
    """Test memory/thread sizing helpers on GazetteerBuilder."""

    def test_returns_fallback_when_physical_memory_is_unknown(self, monkeypatch):
        """An undetectable RAM size still yields an explicit fallback limit."""
        builder = GazetteerBuilder()
        monkeypatch.setattr(builder, "_physical_memory_bytes", lambda: None)

        assert builder._memory_limit_mb() == GazetteerBuilder._FALLBACK_MEMORY_MB

    def test_returns_a_bounded_value_when_memory_is_known(self, monkeypatch):
        """A known RAM size yields a positive, bounded MiB limit."""
        builder = GazetteerBuilder()
        monkeypatch.setattr(
            builder, "_physical_memory_bytes", lambda: 8 * 1024 * 1024 * 1024
        )

        limit = builder._memory_limit_mb()
        assert limit > 0
        assert limit < 8 * 1024

    def test_thread_count_is_capped_by_memory_and_cpu(self, monkeypatch):
        """Thread count never exceeds CPU count or the memory budget."""
        builder = GazetteerBuilder()
        monkeypatch.setattr(
            GazetteerBuilder, "_available_cpus", staticmethod(lambda: 8)
        )

        assert builder._thread_count(512) == 1
        assert builder._thread_count(4096) == 4
        assert builder._thread_count(32_768) == 8

    def test_configure_staging_always_sets_memory_limit_and_threads(
        self, monkeypatch, tmp_path
    ):
        """Staging always applies an explicit memory_limit and threads value."""
        builder = GazetteerBuilder()
        monkeypatch.setattr(
            builder, "_physical_memory_bytes", lambda: 8 * 1024 * 1024 * 1024
        )
        monkeypatch.setattr(
            GazetteerBuilder, "_available_cpus", staticmethod(lambda: 4)
        )
        expected_limit = builder._memory_limit_mb()
        connection = duckdb.connect()
        try:
            builder._configure_staging(connection, tmp_path)
            memory_setting = connection.execute(
                "SELECT value FROM duckdb_settings() WHERE name = 'memory_limit'"
            ).fetchone()
            threads_setting = connection.execute(
                "SELECT value FROM duckdb_settings() WHERE name = 'threads'"
            ).fetchone()
            assert memory_setting is not None
            assert threads_setting is not None
            memory_limit = memory_setting[0]
            threads = threads_setting[0]
        finally:
            connection.close()

        # DuckDB may render/round the limit (e.g. ``4.5 GiB``); check we are
        # near the explicit budget rather than on DuckDB's unbounded default.
        actual_bytes = _setting_to_bytes(memory_limit)
        expected_bytes = expected_limit * 1024 * 1024
        assert abs(actual_bytes - expected_bytes) / expected_bytes < 0.15
        assert int(threads) == 4

    def test_physical_memory_bytes_falls_back_to_windows_api(self, monkeypatch):
        """When sysconf is unavailable, the Windows reader is used."""

        def _no_sysconf():
            raise AttributeError("no sysconf")

        monkeypatch.setattr(
            GazetteerBuilder,
            "_physical_memory_bytes_sysconf",
            staticmethod(_no_sysconf),
        )
        monkeypatch.setattr(
            GazetteerBuilder,
            "_physical_memory_bytes_windows",
            staticmethod(lambda: 16 * 1024 * 1024 * 1024),
        )

        assert GazetteerBuilder._physical_memory_bytes() == 16 * 1024 * 1024 * 1024

    def test_physical_memory_bytes_returns_none_when_all_readers_fail(
        self, monkeypatch
    ):
        """All detection failures are treated as 'unknown', not a crash."""

        def _cgroup_none():
            return None

        def _sysconf_error():
            raise OSError("not supported")

        def _windows_error():
            raise AttributeError("no windll")

        monkeypatch.setattr(
            GazetteerBuilder,
            "_physical_memory_bytes_cgroup",
            staticmethod(_cgroup_none),
        )
        monkeypatch.setattr(
            GazetteerBuilder,
            "_physical_memory_bytes_sysconf",
            staticmethod(_sysconf_error),
        )
        monkeypatch.setattr(
            GazetteerBuilder,
            "_physical_memory_bytes_windows",
            staticmethod(_windows_error),
        )

        assert GazetteerBuilder._physical_memory_bytes() is None

    def test_physical_memory_bytes_prefers_lower_cgroup_limit(self, monkeypatch):
        """A finite cgroup limit wins over larger host RAM from sysconf."""
        monkeypatch.setattr(
            GazetteerBuilder,
            "_physical_memory_bytes_cgroup",
            staticmethod(lambda: 4 * 1024 * 1024 * 1024),
        )
        monkeypatch.setattr(
            GazetteerBuilder,
            "_physical_memory_bytes_sysconf",
            staticmethod(lambda: 16 * 1024 * 1024 * 1024),
        )
        monkeypatch.setattr(
            GazetteerBuilder,
            "_physical_memory_bytes_windows",
            staticmethod(lambda: None),
        )

        assert GazetteerBuilder._physical_memory_bytes() == 4 * 1024 * 1024 * 1024

    def test_available_cpus_falls_back_when_affinity_unsupported(self, monkeypatch):
        """Without sched_getaffinity, cpu_count() is used."""

        def _no_affinity(_pid):
            raise AttributeError("unsupported")

        monkeypatch.setattr(os, "sched_getaffinity", _no_affinity, raising=False)
        monkeypatch.setattr(os, "cpu_count", lambda: 6)

        assert GazetteerBuilder._available_cpus() == 6

    def test_available_cpus_falls_back_when_affinity_missing(self, monkeypatch):
        """Missing sched_getaffinity is handled on platforms without it."""
        monkeypatch.delattr(os, "sched_getaffinity", raising=False)
        monkeypatch.setattr(os, "cpu_count", lambda: 6)

        assert GazetteerBuilder._available_cpus() == 6

    def test_cgroup_reader_returns_none_without_unified_hierarchy(
        self, tmp_path, monkeypatch
    ):
        """Legacy cgroup lines without ``0::`` yield no limit."""
        cgroup = tmp_path / "cgroup"
        cgroup.write_text("1:cpu:/\n")
        real_open = open

        def _open(path, *args, **kwargs):
            if str(path) == "/proc/self/cgroup":
                return real_open(cgroup, *args, **kwargs)
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", _open)

        assert GazetteerBuilder._physical_memory_bytes_cgroup() is None

    def test_cgroup_reader_returns_finite_memory_max(self, tmp_path, monkeypatch):
        """A finite memory.max on the process cgroup is returned."""
        from pathlib import Path as PathType

        cgroup = tmp_path / "cgroup"
        cgroup.write_text("0::/docker/abc\n")
        cg_root = tmp_path / "cgroupfs"
        limit_dir = cg_root / "docker" / "abc"
        limit_dir.mkdir(parents=True)
        (limit_dir / "memory.max").write_text("2147483648\n")

        real_open = open

        def _open(path, *args, **kwargs):
            if str(path) == "/proc/self/cgroup":
                return real_open(cgroup, *args, **kwargs)
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", _open)

        real_path = PathType

        def _path(*args, **kwargs):
            if args and args[0] == "/sys/fs/cgroup":
                return real_path(cg_root, *args[1:], **kwargs)
            return real_path(*args, **kwargs)

        monkeypatch.setattr("geoparser.gazetteer.build.builder.Path", _path)

        assert GazetteerBuilder._physical_memory_bytes_cgroup() == 2147483648

    def test_cgroup_reader_walks_up_to_a_parent_that_sets_a_limit(
        self, tmp_path, monkeypatch
    ):
        """Docker sets memory.max on a parent while the leaf cgroup has none."""
        from pathlib import Path as PathType

        cgroup = tmp_path / "cgroup"
        cgroup.write_text("0::/docker/abc\n")
        cg_root = tmp_path / "cgroupfs"
        (cg_root / "docker" / "abc").mkdir(parents=True)
        # Only the parent carries the limit; the leaf has no memory.max at all.
        (cg_root / "docker" / "memory.max").write_text("1073741824\n")

        real_open = open

        def _open(path, *args, **kwargs):
            if str(path) == "/proc/self/cgroup":
                return real_open(cgroup, *args, **kwargs)
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", _open)

        real_path = PathType

        def _path(*args, **kwargs):
            if args and args[0] == "/sys/fs/cgroup":
                return real_path(cg_root, *args[1:], **kwargs)
            return real_path(*args, **kwargs)

        monkeypatch.setattr("geoparser.gazetteer.build.builder.Path", _path)

        assert GazetteerBuilder._physical_memory_bytes_cgroup() == 1073741824

    def test_cgroup_reader_returns_none_when_every_level_is_unlimited(
        self, tmp_path, monkeypatch
    ):
        """Walking to the root without a finite memory.max yields no limit."""
        from pathlib import Path as PathType

        cgroup = tmp_path / "cgroup"
        cgroup.write_text("0::/docker/abc\n")
        cg_root = tmp_path / "cgroupfs"
        (cg_root / "docker" / "abc").mkdir(parents=True)
        for directory in (cg_root, cg_root / "docker", cg_root / "docker" / "abc"):
            (directory / "memory.max").write_text("max\n")

        real_open = open

        def _open(path, *args, **kwargs):
            if str(path) == "/proc/self/cgroup":
                return real_open(cgroup, *args, **kwargs)
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", _open)

        real_path = PathType

        def _path(*args, **kwargs):
            if args and args[0] == "/sys/fs/cgroup":
                return real_path(cg_root, *args[1:], **kwargs)
            return real_path(*args, **kwargs)

        monkeypatch.setattr("geoparser.gazetteer.build.builder.Path", _path)

        assert GazetteerBuilder._physical_memory_bytes_cgroup() is None

    def test_windows_memory_reader_returns_total_phys(self, monkeypatch):
        """GlobalMemoryStatusEx success yields ullTotalPhys."""
        import ctypes
        import ctypes.wintypes
        import types

        class FakeKernel:
            @staticmethod
            def GlobalMemoryStatusEx(ref):
                ref._obj.ullTotalPhys = 8 * 1024 * 1024 * 1024
                return 1

        monkeypatch.setattr(
            ctypes,
            "windll",
            types.SimpleNamespace(kernel32=FakeKernel),
            raising=False,
        )

        assert (
            GazetteerBuilder._physical_memory_bytes_windows() == 8 * 1024 * 1024 * 1024
        )

    def test_windows_memory_reader_returns_none_on_api_failure(self, monkeypatch):
        """A failed GlobalMemoryStatusEx call is treated as unknown RAM."""
        import ctypes
        import ctypes.wintypes
        import types

        class FakeKernel:
            @staticmethod
            def GlobalMemoryStatusEx(_ref):
                return 0

        monkeypatch.setattr(
            ctypes,
            "windll",
            types.SimpleNamespace(kernel32=FakeKernel),
            raising=False,
        )

        assert GazetteerBuilder._physical_memory_bytes_windows() is None


@pytest.mark.unit
class TestDuplicateGeometryMerge:
    """Bound duplicate-geometry work without loading DuckDB's spatial module."""

    @pytest.mark.parametrize(
        ("query", "duplicate_count", "updates"),
        [
            (None, 0, 0),
            ("SELECT duplicate geometries", 0, 0),
            ("SELECT duplicate geometries", 2, 1),
        ],
    )
    def test_merges_only_when_the_compiler_finds_duplicate_geometries(
        self, monkeypatch, query, duplicate_count, updates
    ):
        connection = MagicMock()
        connection.execute.return_value.fetchone.return_value = (duplicate_count,)
        compiler = MagicMock()
        compiler.duplicate_geometry_query.return_value = query
        feature = FeatureConfig(
            source="places", identifier="id", names=["name"], geometry="geometry"
        )
        monkeypatch.setattr(
            builder_module, "item", lambda *args, **kwargs: nullcontext(None)
        )
        monkeypatch.setattr(
            builder_module, "track", lambda bar, progress, operation: operation()
        )
        advance = MagicMock()
        monkeypatch.setattr(builder_module, "advance", advance)

        GazetteerBuilder()._merge_duplicate_geometries(connection, compiler, feature)

        statements = [call.args[0] for call in connection.execute.call_args_list]
        assert (
            sum(statement.startswith("UPDATE _features") for statement in statements)
            == updates
        )
        assert ("DROP TABLE IF EXISTS _dup_geometry" in statements) is (
            query is not None
        )
        assert advance.call_count == (2 if updates else 1 if query is not None else 0)

    def test_drops_temporary_table_when_count_query_fails(self, monkeypatch):
        connection = MagicMock()

        def execute(sql):
            if sql.startswith("SELECT count(*)"):
                raise RuntimeError("count failed")
            return MagicMock()

        connection.execute.side_effect = execute
        compiler = MagicMock()
        compiler.duplicate_geometry_query.return_value = "SELECT duplicate geometries"
        feature = FeatureConfig(
            source="places", identifier="id", names=["name"], geometry="geometry"
        )
        monkeypatch.setattr(
            builder_module, "item", lambda *args, **kwargs: nullcontext(None)
        )
        monkeypatch.setattr(
            builder_module, "track", lambda bar, progress, operation: operation()
        )
        monkeypatch.setattr(builder_module, "advance", lambda: None)

        with pytest.raises(RuntimeError, match="count failed"):
            GazetteerBuilder()._merge_duplicate_geometries(
                connection, compiler, feature
            )

        assert connection.execute.call_args_list[-1].args[0] == (
            "DROP TABLE IF EXISTS _dup_geometry"
        )


@pytest.mark.unit
class TestFormatBytes:
    """Test human-readable byte formatting for disk errors."""

    def test_formats_gigabytes(self):
        assert _format_bytes(1_500_000_000) == "1.5 GB"

    def test_formats_megabytes(self):
        assert _format_bytes(2_500_000) == "2.5 MB"

    def test_formats_small_values_as_bytes(self):
        assert _format_bytes(512) == "512 bytes"


def _setting_to_bytes(value: str) -> int:
    """Parse a DuckDB memory setting such as ``4.7GB`` or ``4915MB`` to bytes."""
    units = {
        "B": 1,
        "KB": 1000,
        "MB": 1000**2,
        "GB": 1000**3,
        "TB": 1000**4,
        "KIB": 1024,
        "MIB": 1024**2,
        "GIB": 1024**3,
        "TIB": 1024**4,
    }
    cleaned = value.strip().upper().replace(" ", "")
    for suffix, factor in sorted(units.items(), key=lambda item: -len(item[0])):
        if cleaned.endswith(suffix):
            return int(float(cleaned[: -len(suffix)]) * factor)
    raise AssertionError(f"Unrecognized DuckDB memory setting: {value!r}")


@pytest.mark.unit
class TestDiskPreflight:
    """Test GazetteerBuilder._check_disk_space()."""

    def test_skips_when_disk_is_unset(self, tmp_path):
        """Configs without a measured disk budget are not preflight-checked."""
        config = GazetteerConfig.model_validate(
            {
                "name": "custom",
                "sources": [
                    {
                        "name": "rows",
                        "path": "unused.csv",
                        "file": "unused.csv",
                        "delimiter": "\t",
                        "attributes": [
                            {"name": "id", "type": "text"},
                            {"name": "name", "type": "text"},
                        ],
                    }
                ],
                "features": [{"source": "rows", "identifier": "id", "names": ["name"]}],
            }
        )

        GazetteerBuilder()._check_disk_space(config, tmp_path)

    def test_passes_when_free_space_meets_budget(self, monkeypatch, tmp_path):
        """Install proceeds when free space is at least config.disk."""
        config = GazetteerConfig.model_validate(
            {
                "name": "custom",
                "disk": 1_000_000_000,
                "sources": [
                    {
                        "name": "rows",
                        "path": "unused.csv",
                        "file": "unused.csv",
                        "delimiter": "\t",
                        "attributes": [
                            {"name": "id", "type": "text"},
                            {"name": "name", "type": "text"},
                        ],
                    }
                ],
                "features": [{"source": "rows", "identifier": "id", "names": ["name"]}],
            }
        )
        monkeypatch.setattr(
            shutil,
            "disk_usage",
            lambda _path: type(
                "U", (), {"free": 2_000_000_000, "total": 0, "used": 0}
            )(),
        )

        GazetteerBuilder()._check_disk_space(config, tmp_path)

    def test_raises_when_free_space_is_insufficient(self, monkeypatch, tmp_path):
        """Install fails early with a clear OSError when free space is low."""
        config = GazetteerConfig.model_validate(
            {
                "name": "geonames",
                "disk": 30_700_000_000,
                "sources": [
                    {
                        "name": "rows",
                        "path": "unused.csv",
                        "file": "unused.csv",
                        "delimiter": "\t",
                        "attributes": [
                            {"name": "id", "type": "text"},
                            {"name": "name", "type": "text"},
                        ],
                    }
                ],
                "features": [{"source": "rows", "identifier": "id", "names": ["name"]}],
            }
        )
        monkeypatch.setattr(
            shutil,
            "disk_usage",
            lambda _path: type(
                "U", (), {"free": 1_000_000_000, "total": 0, "used": 0}
            )(),
        )

        with pytest.raises(OSError, match=r"Not enough free disk space.*geonames"):
            GazetteerBuilder()._check_disk_space(config, tmp_path)


@pytest.mark.unit
class TestLoadSpatialExtension:
    """Test GazetteerBuilder._load_spatial_extension()."""

    def test_installs_then_loads_spatial(self):
        """A successful install is followed by loading the extension."""
        connection = MagicMock()

        GazetteerBuilder()._load_spatial_extension(connection)

        connection.install_extension.assert_called_once_with("spatial")
        connection.load_extension.assert_called_once_with("spatial")

    def test_wraps_duckdb_errors_in_a_clear_runtime_error(self):
        """A failure to install/load the extension raises a clear RuntimeError."""
        connection = MagicMock()
        connection.install_extension.side_effect = duckdb.Error("network unreachable")
        builder = GazetteerBuilder()

        with pytest.raises(
            RuntimeError, match="Failed to load the DuckDB spatial extension"
        ):
            builder._load_spatial_extension(connection)


@pytest.mark.unit
class TestSpatialPreflight:
    """Test spatial requirements and progress estimates without an extension."""

    def test_needs_spatial_for_non_tabular_or_geometry_features(self):
        """Spatial files and geometry projections require the extension."""
        tabular_source = {
            "name": "places",
            "path": "places.csv",
            "file": "places.csv",
            "delimiter": ",",
            "attributes": [
                {"name": "id", "type": "integer"},
                {"name": "name", "type": "text"},
            ],
        }
        tabular_feature = {
            "source": "places",
            "identifier": "id",
            "names": ["name"],
        }
        tabular_config = GazetteerConfig.model_validate(
            {
                "name": "tabular",
                "sources": [tabular_source],
                "features": [tabular_feature],
            }
        )
        geometry_config = GazetteerConfig.model_validate(
            {
                "name": "geometry",
                "sources": [tabular_source],
                "features": [{**tabular_feature, "geometry": "ST_Point(0, 0)"}],
            }
        )
        spatial_config = GazetteerConfig.model_validate(
            {
                "name": "spatial",
                "sources": [
                    {
                        "name": "places",
                        "path": "places.geojson",
                        "file": "places.geojson",
                        "attributes": [
                            *tabular_source["attributes"],
                            {"name": "geometry", "type": "geometry"},
                        ],
                    }
                ],
                "features": [tabular_feature],
            }
        )
        builder = GazetteerBuilder()

        assert builder._needs_spatial(tabular_config) is False
        assert builder._needs_spatial(geometry_config) is True
        assert builder._needs_spatial(spatial_config) is True

    def test_compile_estimate_counts_geometry_duplicate_checks(self):
        """The stage estimate reserves a check for each geometric feature."""
        config = GazetteerConfig.model_validate(
            {
                "name": "estimate",
                "sources": [
                    {
                        "name": "places",
                        "path": "places.csv",
                        "file": "places.csv",
                        "delimiter": ",",
                        "attributes": [
                            {"name": "id", "type": "integer"},
                            {"name": "name", "type": "text"},
                        ],
                    },
                    {
                        "name": "villages",
                        "path": "villages.csv",
                        "file": "villages.csv",
                        "delimiter": ",",
                        "attributes": [
                            {"name": "id", "type": "integer"},
                            {"name": "name", "type": "text"},
                            {"name": "lon", "type": "real"},
                            {"name": "lat", "type": "real"},
                        ],
                    },
                ],
                "features": [
                    {"source": "places", "identifier": "id", "names": ["name"]},
                    {
                        "source": "villages",
                        "identifier": "id",
                        "names": ["name", "upper(name)"],
                        "geometry": "ST_Point(lon, lat)",
                    },
                ],
            }
        )
        compiler = builder_module.ProjectionCompiler(
            config, GazetteerBuilder._source_catalog(config)
        )

        estimate = GazetteerBuilder._compile_item_estimate(config, compiler)

        assert estimate == 9

    def test_uninstall_reports_when_artifact_is_missing(self, monkeypatch, tmp_path):
        """Uninstalling an absent artifact reports that nothing was removed."""
        monkeypatch.setattr(
            builder_module.artifact,
            "artifact_path",
            lambda _name: tmp_path / "absent.db",
        )

        assert uninstall("absent") is False


@pytest.mark.unit
class TestBuildErrorPaths:
    """Test build() error paths that are hard to reach end-to-end."""

    def test_spatial_load_progress_completes(self, monkeypatch, tmp_path):
        """A successful spatial setup completes its visible progress item."""
        config_file = tmp_path / "places.yaml"
        config_file.write_text(
            textwrap.dedent(
                """
                name: places
                sources:
                  - name: places
                    path: places.csv
                    file: places.csv
                    delimiter: "\\t"
                    attributes:
                      - name: id
                        type: text
                      - name: lon
                        type: real
                      - name: lat
                        type: real
                features:
                  - source: places
                    identifier: id
                    names: [id]
                    geometry: ST_Point(lon, lat)
                """
            )
        )
        gazetteers_dir = tmp_path / "gazetteers"
        monkeypatch.setattr(
            artifact, "artifact_path", lambda name: gazetteers_dir / f"{name}.db"
        )
        connection = MagicMock()
        progress = MagicMock()
        acquirer = MagicMock()
        monkeypatch.setattr(builder_module.duckdb, "connect", lambda _path: connection)
        monkeypatch.setattr(builder_module, "Acquirer", lambda _path: acquirer)
        monkeypatch.setattr(builder_module, "build_display", nullcontext)
        monkeypatch.setattr(
            builder_module, "item", lambda *args, **kwargs: nullcontext(progress)
        )
        builder = GazetteerBuilder()
        monkeypatch.setattr(builder, "_configure_staging", lambda *_args: None)
        monkeypatch.setattr(builder, "_needs_spatial", lambda _config: True)
        load_spatial = MagicMock()
        monkeypatch.setattr(builder, "_load_spatial_extension", load_spatial)
        monkeypatch.setattr(builder, "_prepare_sources", lambda *_args: None)
        monkeypatch.setattr(builder, "_compile_features", lambda *_args: None)
        monkeypatch.setattr(builder, "_build_artifact", lambda *_args: (2, 3))

        result = builder.build(config_file)

        assert result == gazetteers_dir / "places.db"
        load_spatial.assert_called_once_with(connection)
        progress.set_progress.assert_called_once_with(100)
        connection.close.assert_called_once()
        acquirer.cleanup.assert_called_once()

    def test_compile_features_raises_when_no_features_produced(self):
        """A source with zero rows produces zero features, which fails clearly.

        Builds the underlying DuckDB table directly (bypassing acquire/load)
        so the scenario doesn't depend on how the CSV reader handles an
        empty file.
        """
        connection = duckdb.connect()
        connection.execute("CREATE TABLE rows (rid VARCHAR, name VARCHAR)")
        config = GazetteerConfig.model_validate(
            {
                "name": "empty",
                "sources": [
                    {
                        "name": "rows",
                        "path": "unused.csv",
                        "file": "unused.csv",
                        "delimiter": "\t",
                        "attributes": [
                            {"name": "rid", "type": "text"},
                            {"name": "name", "type": "text"},
                        ],
                    }
                ],
                "features": [
                    {"source": "rows", "identifier": "rid", "names": ["name"]}
                ],
            }
        )
        builder = GazetteerBuilder()

        with pytest.raises(ValueError, match="produced no features"):
            builder._compile_features(connection, config)

    def test_rebuild_clears_a_stale_leftover_build_directory(
        self, tmp_path, monkeypatch
    ):
        """A leftover build directory from an earlier, interrupted build is
        cleared out rather than reused."""
        data_file = tmp_path / "peaks.csv"
        data_file.write_text("p1\tSummit\t800\n")
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
                """
            )
        )
        gazetteers_dir = tmp_path / "gazetteers"
        monkeypatch.setenv("GEOPARSER_GAZETTEERS_DIR", str(gazetteers_dir))
        target_path = artifact.artifact_path("peaks")
        stale_build_dir = target_path.parent / ".build-peaks"
        stale_build_dir.mkdir(parents=True)
        (stale_build_dir / "leftover.txt").write_text("stale")

        GazetteerBuilder().build(config_file)

        assert target_path.exists()
        assert not stale_build_dir.exists()
