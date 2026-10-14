"""
Gazetteer builder: compiles a declarative config into a SQLite artifact.

The build pipeline has three stages, each its own progress group:

1. Preparing sources: resolve (download/extract, cached across builds) and
   load each source's file into a transient DuckDB table
2. Compiling features: compile and run one projection per feature block,
   producing canonical (identifier, source, data, geometry) and
   (identifier, name) rows, then merge rows that share an identifier
3. Building artifact: copy the compiled rows into a temporary SQLite file,
   build its search structures and metadata, then atomically move it into
   place

The staging database and all intermediate tables are discarded after the
build; the artifact is the only output.
"""

import contextlib
import os
import re
import shutil
import typing as t
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import duckdb

from geoparser.gazetteer import artifact
from geoparser.gazetteer.build.progress import (
    advance,
    build_display,
    item,
    label_suffixes,
    print_build_header,
    print_build_memory_diagnostics,
    print_build_summary,
    stage,
    track,
)
from geoparser.gazetteer.build.schema import FeatureConfig, GazetteerConfig
from geoparser.gazetteer.build.stages.acquire import Acquirer
from geoparser.gazetteer.build.stages.compile import ProjectionCompiler
from geoparser.gazetteer.build.stages.emit import create_artifact_db, emit, finalize
from geoparser.gazetteer.build.stages.load import Loader, quote_literal, scalar_int

_MEMORY_LIMIT_ENV = "GEOPARSER_GAZETTEER_BUILD_MEMORY_LIMIT_MB"
_CGROUP_V1_UNLIMITED_BYTES = 1 << 60


@dataclass(frozen=True)
class _MemoryBudget:
    """Detected allocation memory and the staging limits derived from it."""

    sources: tuple[tuple[str, int], ...]
    detected_bytes: int | None
    detected_source: str
    duckdb_limit_mb: int
    threads: int
    configured_limit_mb: int | None


def _sqlite_temp_env_names() -> tuple[str, ...]:
    """
    Environment variables that steer SQLite's temporary-file location.

    On Unix, SQLite consults ``SQLITE_TMPDIR`` (then ``TMPDIR``, then
    ``/tmp``). On Windows it ignores those and uses ``GetTempPath()``, which
    reads ``TMP`` then ``TEMP``. Returning the platform-appropriate names
    keeps large FTS/VACUUM spills on the build volume everywhere.
    """
    if os.name == "nt":
        return ("TMP", "TEMP")
    return ("SQLITE_TMPDIR",)


def _format_bytes(nbytes: int) -> str:
    """Format a byte count for error messages (decimal GB/MB)."""
    if nbytes >= 1_000_000_000:
        return f"{nbytes / 1_000_000_000:.1f} GB"
    if nbytes >= 1_000_000:
        return f"{nbytes / 1_000_000:.1f} MB"
    return f"{nbytes} bytes"


@contextlib.contextmanager
def _sqlite_tmpdir(directory: Path) -> t.Iterator[None]:
    """
    Temporarily point SQLite's temporary storage at ``directory``.

    SQLite has no reliable cross-platform per-connection temp-directory
    setting, so the location is steered through environment variables (see
    :func:`_sqlite_temp_env_names`). The previous values are restored when
    the context exits.

    Args:
        directory: Directory SQLite should use for temporary files
    """
    directory_str = str(directory)
    names = _sqlite_temp_env_names()
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ[name] = directory_str
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class GazetteerBuilder:
    """
    Builds gazetteer artifacts from declarative configuration files.
    """

    # Fraction of physical memory the staging engine may use before it starts
    # spilling intermediate data to disk.
    _MEMORY_FRACTION = 0.6
    # Headroom always left for the OS, the Python process and the SQLite
    # writer, so the whole build never approaches the machine's memory cap.
    _MEMORY_HEADROOM_MB = 1536
    # Never bound the staging engine below this, or builds become impractically
    # slow.
    _MIN_MEMORY_MB = 512
    # Used when physical RAM cannot be detected, so the staging engine is
    # still explicitly bounded instead of sizing itself against unknown RAM.
    _FALLBACK_MEMORY_MB = 2048
    # Rough per-thread working-set budget used to cap parallelism so thread
    # count stays proportional to the memory limit on small machines.
    _MB_PER_THREAD = 1024

    def build(self, config_path: str | Path, keep_downloads: bool = False) -> Path:
        """
        Build and install a gazetteer from a configuration file.

        Args:
            config_path: Path to the YAML configuration file
            keep_downloads: Whether to keep downloaded files after the build

        Returns:
            Path of the installed artifact
        """
        config = GazetteerConfig.from_yaml(config_path)
        print_build_header(config.name)

        target_path = artifact.artifact_path(config.name)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        self._check_disk_space(config, target_path.parent)

        downloads_dir = target_path.parent / ".downloads" / config.name
        acquirer = Acquirer(downloads_dir)
        build_dir = target_path.parent / f".build-{config.name}"
        if build_dir.exists():
            shutil.rmtree(build_dir)
        build_dir.mkdir(parents=True)

        try:
            with build_display():
                connection = duckdb.connect(str(build_dir / "staging.duckdb"))
                try:
                    self._configure_staging(connection, build_dir)
                    if self._needs_spatial(config):
                        with item("Loading spatial extension", total=100) as bar:
                            self._load_spatial_extension(connection)
                            bar.set_progress(100)
                    self._prepare_sources(acquirer, connection, config)
                    self._compile_features(connection, config)
                    feature_count, name_count = self._build_artifact(
                        connection, config, build_dir, target_path
                    )
                finally:
                    connection.close()
        finally:
            shutil.rmtree(build_dir, ignore_errors=True)
            if not keep_downloads:
                acquirer.cleanup()

        print_build_summary(feature_count, name_count)
        return target_path

    def _check_disk_space(self, config: GazetteerConfig, directory: Path) -> None:
        """
        Fail early when the gazetteers volume lacks the configured free space.

        Skipped when ``config.disk`` is unset (unmeasured custom configs).
        """
        if config.disk is None:
            return
        free = shutil.disk_usage(directory).free
        if free >= config.disk:
            return
        msg = (
            f"Not enough free disk space to install gazetteer '{config.name}': "
            f"need {_format_bytes(config.disk)}, have {_format_bytes(free)} free "
            f"on {directory}"
        )
        raise OSError(msg)

    def _configure_staging(
        self, connection: duckdb.DuckDBPyConnection, build_dir: Path
    ) -> None:
        """
        Bound the staging engine's memory and threads, and make it spill to disk.

        Large gazetteers (GeoNames is ~13M rows) do not fit in memory. By
        default DuckDB sizes its memory limit to a large fraction of RAM and
        only spills relative to that limit, so on memory-constrained machines
        (or WSL, where the reported RAM is the VM's) the process can be killed
        by the OS before it decides to spill. Capping the limit with headroom,
        scaling threads to that budget, and pointing the temporary directory
        at on-disk build storage keeps peak memory bounded regardless of the
        machine's specs — including Windows, where physical RAM is detected
        via the Win32 API rather than ``os.sysconf``.

        Args:
            connection: DuckDB connection used for staging and projection
            build_dir: Directory holding the transient build files
        """
        # DuckDB tracks query progress internally only while its progress bar
        # feature is enabled. By default it also prints that progress straight
        # to the terminal, outside Rich's control, which would desync the live
        # build display (stray bars flashing in, stage rows appearing to
        # repeat); ``enable_progress_bar_print`` keeps the tracking without
        # the printing, so `connection.query_progress()` (used to drive our
        # own item bars, see ``Loader`` and ``_compile_features``) works while
        # the terminal stays under our control. ``progress_bar_time = 0``
        # makes tracking start immediately rather than after DuckDB's default
        # delay for what it guesses will be a short query.
        connection.execute("SET enable_progress_bar = true")
        connection.execute("SET enable_progress_bar_print = false")
        connection.execute("SET progress_bar_time = 0")

        temp_dir = build_dir / "duckdb-temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        connection.execute(f"SET temp_directory = {quote_literal(str(temp_dir))}")
        # Ordering is established explicitly where it matters (row ids, emit),
        # so let DuckDB avoid buffering results just to preserve input order.
        connection.execute("SET preserve_insertion_order = false")
        # Always set an explicit memory limit and thread cap. Leaving either
        # at DuckDB's defaults lets it size itself against reported RAM and
        # use every core, which on Windows (where RAM detection used to fail)
        # and on small machines can exhaust memory before spilling kicks in.
        budget = self._memory_budget()
        print_build_memory_diagnostics(self._memory_budget_message(budget))
        memory_limit_mb = budget.duckdb_limit_mb
        connection.execute(f"SET memory_limit = '{memory_limit_mb}MiB'")
        connection.execute(f"SET threads = {budget.threads}")

    def _memory_limit_mb(self) -> int:
        """
        Return a conservative staging memory limit in MiB.

        When physical RAM can be detected, the limit is a fraction of that
        total with headroom left for the OS and the rest of the build. When
        it cannot, a fixed fallback is used so the staging engine is still
        explicitly bounded. ``GEOPARSER_GAZETTEER_BUILD_MEMORY_LIMIT_MB`` can
        lower this ceiling for a memory-constrained build.
        """
        return self._memory_budget().duckdb_limit_mb

    def _memory_budget(self) -> _MemoryBudget:
        """Detect available memory and derive the DuckDB cap and thread count."""
        sources = self._memory_sources()
        detected_source, detected_bytes = min(
            sources, key=lambda source: source[1], default=("fallback", None)
        )
        automatic_limit_mb = self._automatic_memory_limit_mb(
            detected_bytes, detected_source
        )
        configured_limit_mb = self._configured_memory_limit_mb()
        duckdb_limit_mb = self._apply_memory_ceiling(
            automatic_limit_mb, configured_limit_mb
        )
        return _MemoryBudget(
            sources=sources,
            detected_bytes=detected_bytes,
            detected_source=detected_source,
            duckdb_limit_mb=duckdb_limit_mb,
            threads=self._thread_count(duckdb_limit_mb),
            configured_limit_mb=configured_limit_mb,
        )

    def _automatic_memory_limit_mb(
        self, detected_bytes: int | None, detected_source: str
    ) -> int:
        """Apply the default bounded-memory policy to one detected limit."""
        if detected_bytes is None:
            return self._FALLBACK_MEMORY_MB
        total_mb = detected_bytes / (1024 * 1024)
        minimum_total_mb = self._MIN_MEMORY_MB + self._MEMORY_HEADROOM_MB
        if total_mb < minimum_total_mb:
            raise self._insufficient_memory_error(detected_source, total_mb)
        return int(
            max(
                min(
                    total_mb * self._MEMORY_FRACTION,
                    total_mb - self._MEMORY_HEADROOM_MB,
                ),
                self._MIN_MEMORY_MB,
            )
        )

    def _insufficient_memory_error(self, source: str, total_mb: float) -> MemoryError:
        """Build the clear error raised when the allocation is too small."""
        minimum_total_mb = self._MIN_MEMORY_MB + self._MEMORY_HEADROOM_MB
        message = (
            f"Detected {source} memory limit of {total_mb:,.1f} MiB "
            f"is below the {minimum_total_mb:,} MiB minimum needed to reserve "
            f"{self._MEMORY_HEADROOM_MB:,} MiB of headroom alongside a "
            f"{self._MIN_MEMORY_MB:,} MiB DuckDB limit."
        )
        return MemoryError(message)

    @staticmethod
    def _apply_memory_ceiling(
        automatic_limit_mb: int, configured_limit_mb: int | None
    ) -> int:
        """Apply a configured ceiling without raising the automatic cap."""
        if configured_limit_mb is None:
            return automatic_limit_mb
        return min(automatic_limit_mb, configured_limit_mb)

    @staticmethod
    def _configured_memory_limit_mb() -> int | None:
        """Read an optional positive MiB ceiling without changing defaults."""
        value = os.environ.get(_MEMORY_LIMIT_ENV)
        return GazetteerBuilder._parse_configured_memory_limit(value)

    @staticmethod
    def _parse_configured_memory_limit(value: str | None) -> int | None:
        """Parse the optional decimal MiB ceiling."""
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            return None
        if GazetteerBuilder._valid_configured_memory_limit(cleaned):
            return int(cleaned)
        msg = f"{_MEMORY_LIMIT_ENV} must be a positive integer number of MiB"
        raise ValueError(msg)

    @staticmethod
    def _valid_configured_memory_limit(value: str) -> bool:
        """Whether text is a positive ASCII decimal integer."""
        if not value.isascii():
            return False
        if not value.isdecimal():
            return False
        return int(value) > 0

    @staticmethod
    def _memory_sources() -> tuple[tuple[str, int], ...]:
        """Return detected cgroup and host memory values with their sources."""
        readers = (
            ("cgroup v2", GazetteerBuilder._physical_memory_bytes_cgroup_v2),
            ("cgroup v1", GazetteerBuilder._physical_memory_bytes_cgroup_v1),
            ("sysconf", GazetteerBuilder._physical_memory_bytes_sysconf),
            ("Windows API", GazetteerBuilder._physical_memory_bytes_windows),
        )
        return tuple(
            (name, value)
            for name, reader in readers
            if (value := GazetteerBuilder._read_memory(reader)) is not None
        )

    @staticmethod
    def _memory_budget_message(budget: _MemoryBudget) -> str:
        """Describe the memory probes and limits in a job-captured log line."""
        cgroup_text = GazetteerBuilder._format_cgroup_memory_limits(budget.sources)
        detected_text = GazetteerBuilder._format_detected_memory(
            budget.detected_bytes, budget.detected_source
        )
        ceiling_text = GazetteerBuilder._format_configured_memory_ceiling(
            budget.configured_limit_mb
        )
        return (
            f"Installer memory: cgroup limits={cgroup_text}; detected={detected_text}; "
            f"{ceiling_text}; DuckDB cap={budget.duckdb_limit_mb:,} MiB; "
            f"threads={budget.threads}"
        )

    @staticmethod
    def _format_cgroup_memory_limits(sources: tuple[tuple[str, int], ...]) -> str:
        """Format available cgroup probes for concise diagnostics."""
        formatted = (
            f"{name}={value // (1024 * 1024):,} MiB"
            for name, value in sources
            if name.startswith("cgroup")
        )
        return ", ".join(formatted) or "unavailable"

    @staticmethod
    def _format_detected_memory(
        detected_bytes: int | None, detected_source: str
    ) -> str:
        """Format the selected source or describe fallback detection."""
        if detected_bytes is None:
            return "unknown; using bounded fallback"
        detected_mib = detected_bytes / (1024 * 1024)
        return f"{detected_mib:,.0f} MiB from {detected_source}"

    @staticmethod
    def _format_configured_memory_ceiling(limit_mb: int | None) -> str:
        """Format the optional configured ceiling for diagnostics."""
        if limit_mb is None:
            return "no configured ceiling"
        return f"configured ceiling={limit_mb:,} MiB"

    def _thread_count(self, memory_limit_mb: int) -> int:
        """
        Return how many DuckDB threads to use for the given memory limit.

        Caps parallelism by both the CPUs available to this process (honoring
        cpuset / affinity limits from Docker and similar) and the memory
        budget so small machines do not run one hungry worker per core.
        """
        cpu_count = self._available_cpus()
        by_memory = max(1, memory_limit_mb // self._MB_PER_THREAD)
        return max(1, min(cpu_count, by_memory))

    @staticmethod
    def _available_cpus() -> int:
        """Return CPUs usable by this process (affinity-aware when possible)."""
        affinity = os.__dict__.get("sched_getaffinity")
        if affinity is None:
            return os.cpu_count() or 1
        try:
            return len(affinity(0))
        except (AttributeError, OSError):
            return os.cpu_count() or 1

    @staticmethod
    def _physical_memory_bytes() -> int | None:
        """
        Return usable physical memory in bytes, if detectable.

        Prefers a cgroup memory limit when one is set (Docker, systemd
        scopes, ...), since ``os.sysconf`` reports host RAM and would
        otherwise let the staging engine size itself above the container
        budget and get OOM-killed.
        """
        candidates = [value for _, value in GazetteerBuilder._memory_sources()]
        return min(candidates) if candidates else None

    @staticmethod
    def _read_memory(reader: t.Callable[[], int | None]) -> int | None:
        """
        Run one memory probe, treating an unavailable source as no answer.

        Args:
            reader: A probe that returns a byte count or None

        Returns:
            A positive byte count, or None when this source cannot answer
        """
        try:
            value = reader()
        except (ValueError, OSError, AttributeError):
            return None
        return value if value is not None and value > 0 else None

    @staticmethod
    def _physical_memory_bytes_cgroup() -> int | None:
        """Return the smallest finite v1 or v2 cgroup memory limit."""
        candidates = [
            value
            for reader in (
                GazetteerBuilder._physical_memory_bytes_cgroup_v2,
                GazetteerBuilder._physical_memory_bytes_cgroup_v1,
            )
            if (value := GazetteerBuilder._read_memory(reader)) is not None
        ]
        return min(candidates) if candidates else None

    @staticmethod
    def _physical_memory_bytes_cgroup_v2() -> int | None:
        """Read this process's finite cgroup v2 ``memory.max`` limit."""
        return GazetteerBuilder._cgroup_memory_limit("v2", "memory.max")

    @staticmethod
    def _physical_memory_bytes_cgroup_v1() -> int | None:
        """Read this process's finite cgroup v1 ``memory.limit_in_bytes``."""
        return GazetteerBuilder._cgroup_memory_limit(
            "v1", "memory.limit_in_bytes", unlimited_bytes=_CGROUP_V1_UNLIMITED_BYTES
        )

    @classmethod
    def _cgroup_memory_limit(
        cls, version: str, filename: str, unlimited_bytes: int | None = None
    ) -> int | None:
        """Read the smallest finite memory cap from a cgroup and its parents."""
        membership = cls._cgroup_memberships().get(version)
        if membership is None:
            return None
        limits = cls._cgroup_limits_for_mounts(
            membership, cls._cgroup_mounts(), version, filename, unlimited_bytes
        )
        return min(limits, default=None)

    @classmethod
    def _cgroup_limits_for_mounts(
        cls,
        membership: str,
        mounts: tuple[tuple[str, str, str], ...],
        version: str,
        filename: str,
        unlimited_bytes: int | None,
    ) -> tuple[int, ...]:
        """Collect finite limits from mounts matching one cgroup version."""
        return tuple(
            limit
            for mount_version, mount_root, mountpoint in mounts
            if mount_version == version
            if (
                limit := cls._cgroup_limit_for_mount(
                    membership, mount_root, mountpoint, filename, unlimited_bytes
                )
            )
            is not None
        )

    @classmethod
    def _cgroup_limit_for_mount(
        cls,
        membership: str,
        mount_root: str,
        mountpoint: str,
        filename: str,
        unlimited_bytes: int | None,
    ) -> int | None:
        """Find the smallest finite cap from a mounted cgroup and its parents."""
        try:
            relative = PurePosixPath(membership).relative_to(PurePosixPath(mount_root))
        except ValueError:
            return None
        boundary = Path(mountpoint)
        directory = boundary.joinpath(*relative.parts)
        values = (
            cls._read_cgroup_limit_file(path / filename, unlimited_bytes)
            for path in cls._cgroup_ancestors(directory, boundary)
        )
        return min((value for value in values if value is not None), default=None)

    @staticmethod
    def _cgroup_ancestors(directory: Path, boundary: Path) -> t.Iterator[Path]:
        """Yield a cgroup directory and its parents up to the mount boundary."""
        while directory == boundary or boundary in directory.parents:
            yield directory
            if directory == boundary:
                return
            directory = directory.parent

    @staticmethod
    def _read_cgroup_limit_file(path: Path, unlimited_bytes: int | None) -> int | None:
        """Read one finite cgroup memory limit, ignoring missing/unlimited files."""
        try:
            value_text = path.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        return GazetteerBuilder._parse_cgroup_limit(value_text, unlimited_bytes)

    @staticmethod
    def _parse_cgroup_limit(value_text: str, unlimited_bytes: int | None) -> int | None:
        """Parse one cgroup limit and ignore malformed or unlimited values."""
        if not value_text or value_text == "max":
            return None
        try:
            value = int(value_text)
        except ValueError:
            return None
        if not GazetteerBuilder._is_finite_cgroup_limit(value, unlimited_bytes):
            return None
        return value

    @staticmethod
    def _is_finite_cgroup_limit(value: int, unlimited_bytes: int | None) -> bool:
        """Whether a parsed cgroup value is a usable finite positive cap."""
        if value <= 0:
            return False
        return unlimited_bytes is None or value < unlimited_bytes

    @staticmethod
    def _read_proc_file(path: str) -> str:
        """Read a small Linux procfs file used for cgroup discovery."""
        return Path(path).read_text(encoding="utf-8")

    @staticmethod
    def _cgroup_memberships() -> dict[str, str]:
        """Map cgroup v1/v2 hierarchies to this process's membership paths."""
        try:
            contents = GazetteerBuilder._read_proc_file("/proc/self/cgroup")
        except OSError:
            return {}
        entries = map(GazetteerBuilder._parse_cgroup_membership, contents.splitlines())
        return dict(filter(None, entries))

    @staticmethod
    def _parse_cgroup_membership(line: str) -> tuple[str, str] | None:
        """Parse one unified or memory-controller membership line."""
        hierarchy, controllers, path = line.split(":", maxsplit=2)
        if hierarchy == "0" and not controllers:
            return "v2", path
        if "memory" in controllers.split(","):
            return "v1", path
        return None

    @staticmethod
    def _cgroup_mounts() -> tuple[tuple[str, str, str], ...]:
        """Read cgroup mount version, root and mountpoint from mountinfo."""
        try:
            contents = GazetteerBuilder._read_proc_file("/proc/self/mountinfo")
        except OSError:
            return ()
        entries = map(
            GazetteerBuilder._parse_cgroup_mountinfo_line, contents.splitlines()
        )
        return tuple(filter(None, entries))

    @staticmethod
    def _parse_cgroup_mountinfo_line(line: str) -> tuple[str, str, str] | None:
        """Parse one mountinfo row when it describes a cgroup mount."""
        before, separator, after = line.partition(" - ")
        if not separator:
            return None
        mount_fields = before.split()
        if len(mount_fields) < 5:
            return None
        filesystem_fields = after.split(maxsplit=2)
        version = GazetteerBuilder._cgroup_mount_version(filesystem_fields)
        if version is None:
            return None
        root = GazetteerBuilder._decode_mountinfo_path(mount_fields[3])
        mountpoint = GazetteerBuilder._decode_mountinfo_path(mount_fields[4])
        return version, root, mountpoint

    @staticmethod
    def _cgroup_mount_version(filesystem_fields: list[str]) -> str | None:
        """Identify cgroup v1/v2 mount rows, ignoring unrelated filesystems."""
        if not filesystem_fields:
            return None
        filesystem = filesystem_fields[0]
        if filesystem == "cgroup2":
            return "v2"
        if filesystem == "cgroup":
            return GazetteerBuilder._cgroup_v1_mount_version(filesystem_fields)
        return None

    @staticmethod
    def _cgroup_v1_mount_version(filesystem_fields: list[str]) -> str | None:
        """Check whether a legacy cgroup mount exposes the memory controller."""
        if len(filesystem_fields) != 3:
            return None
        if "memory" in filesystem_fields[2].split(","):
            return "v1"
        return None

    @staticmethod
    def _decode_mountinfo_path(value: str) -> str:
        """Decode octal escapes used for whitespace in mountinfo paths."""
        return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), value)

    @staticmethod
    def _physical_memory_bytes_sysconf() -> int | None:
        """Read physical memory via ``os.sysconf`` (Linux, macOS, ...)."""
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")

    @staticmethod
    def _physical_memory_bytes_windows() -> int | None:
        """Read physical memory via ``GlobalMemoryStatusEx`` (Windows)."""
        import ctypes
        import ctypes.wintypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.wintypes.DWORD),
                ("dwMemoryLoad", ctypes.wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_uint64),
                ("ullAvailPhys", ctypes.c_uint64),
                ("ullTotalPageFile", ctypes.c_uint64),
                ("ullAvailPageFile", ctypes.c_uint64),
                ("ullTotalVirtual", ctypes.c_uint64),
                ("ullAvailVirtual", ctypes.c_uint64),
                ("ullAvailExtendedVirtual", ctypes.c_uint64),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(status)
        # ctypes only exposes windll on Windows. Off Windows this raises
        # AttributeError, which _physical_memory_bytes catches to skip this
        # reader; typeshed likewise only declares it for the checker's platform.
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(  # ty: ignore[unresolved-attribute]
            ctypes.byref(status)
        ):
            return None
        return int(status.ullTotalPhys)

    def _needs_spatial(self, config: GazetteerConfig) -> bool:
        """Whether the build requires DuckDB's spatial extension."""
        if any(not source_config.is_tabular for source_config in config.sources):
            return True
        return any(feature.geometry is not None for feature in config.features)

    def _load_spatial_extension(self, connection: duckdb.DuckDBPyConnection) -> None:
        """Install and load the DuckDB spatial extension."""
        try:
            connection.install_extension("spatial")
            connection.load_extension("spatial")
        except duckdb.Error as error:
            msg = (
                "Failed to load the DuckDB spatial extension, which is required "
                "to build gazetteers with geometries. The extension is downloaded "
                "on first use; make sure you have network access, then try again. "
                f"Original error: {error}"
            )
            raise RuntimeError(msg) from error

    def _prepare_sources(
        self,
        acquirer: Acquirer,
        connection: duckdb.DuckDBPyConnection,
        config: GazetteerConfig,
    ) -> None:
        """
        Resolve, then load, each source's data file in turn.

        Acquiring (see :mod:`stages.acquire`) and loading (see :mod:`stages.load`) each
        report their own items as they run their actual downloads,
        extractions and queries, and advance the stage themselves as each
        one finishes; how many a given source shows depends on whether it
        needed downloading/extracting and whether it is spatial, so the
        estimate below undercounts some sources and overcounts others, but
        the stage's total grows on the fly to stay ahead of it either way.
        """
        loader = Loader(connection, config.crs)
        total_estimate = sum(
            1 + (1 if source_config.is_tabular else 2)
            for source_config in config.sources
        )
        with stage("Preparing sources", "Prepared sources", total_estimate):
            for source_config in config.sources:
                file_path = acquirer.acquire(source_config)
                loader.load(source_config, file_path)

    @staticmethod
    def _source_catalog(config: GazetteerConfig) -> dict[str, list[str]]:
        """
        The attribute names of each staged source table.

        Every source declares its attributes, so the catalog comes straight
        from the config; the staged tables carry exactly this schema.

        Args:
            config: The gazetteer configuration

        Returns:
            Attribute names keyed by source name
        """
        return {
            source_config.name: [
                attribute.name for attribute in source_config.attributes
            ]
            for source_config in config.sources
        }

    @staticmethod
    def _compile_item_estimate(
        config: GazetteerConfig, compiler: ProjectionCompiler
    ) -> int:
        """
        How many progress items the compile stage expects to show.

        Each feature block shows one item for assembling its features, one per
        name it collects, and (only if it has a geometry) one for checking
        whether any identifier repeats. Three more cover the cross-block merge.
        The stage's total grows on the fly to cover the further "merging
        duplicates" item that shows up only when duplicates are found.

        Args:
            config: The gazetteer configuration
            compiler: The projection compiler for this build

        Returns:
            The initial item estimate for the stage
        """
        total = 3
        for feature in config.features:
            total += 1 + len(compiler.name_queries(feature))
            if compiler.duplicate_geometry_query(feature) is not None:
                total += 1
        return total

    def _compile_features(
        self, connection: duckdb.DuckDBPyConnection, config: GazetteerConfig
    ) -> None:
        """Run each feature block's projection, then merge across blocks."""
        # Every source declares its attributes, so the catalog is built
        # directly from the config; the staged tables carry exactly this schema.
        compiler = ProjectionCompiler(config, self._source_catalog(config))

        connection.execute(
            "CREATE TABLE _features (identifier VARCHAR, source VARCHAR, "
            "data VARCHAR, geometry BLOB)"
        )
        connection.execute("CREATE TABLE _names (identifier VARCHAR, text VARCHAR)")

        # Each feature block shows one item for assembling its features, one
        # per name it collects, and (only if it has a geometry) one for
        # checking whether any identifier repeats; the stage's total grows on
        # the fly to cover the further "merging duplicates" item that shows
        # up only when duplicates are actually found.
        total_estimate = self._compile_item_estimate(config, compiler)

        with stage("Compiling features", "Compiled features", total_estimate):
            for feature in config.features:
                self._compile_feature(connection, compiler, feature)

            # Merging rows across feature blocks (the checks and joins below)
            # means scanning the whole of _features and _names, which for a
            # large gazetteer is not instantaneous; each gets its own item
            # rather than leaving a silent gap after the last feature block.
            with item("Checking identifiers", total=100) as bar:
                track(
                    bar,
                    connection.query_progress,
                    lambda: self._check_duplicate_identifiers(connection),
                )
            advance()

            # Deterministic internal ids; names of unknown identifiers (e.g.
            # from a related names input covering more places) are dropped by
            # the join below.
            with item("Assigning feature ids", total=100) as bar:
                track(
                    bar,
                    connection.query_progress,
                    lambda: connection.execute(
                        "CREATE TABLE _features_final AS "
                        "SELECT row_number() OVER (ORDER BY identifier) AS id, "
                        "* FROM _features"
                    ),
                )
            advance()

            with item("Linking names", total=100) as bar:
                track(
                    bar,
                    connection.query_progress,
                    lambda: connection.execute(
                        "CREATE TABLE _names_final AS "
                        "SELECT DISTINCT f.id AS feature_id, n.text "
                        "FROM _names n JOIN _features_final f USING (identifier)"
                    ),
                )
            advance()

        total = scalar_int(connection, "SELECT count(*) FROM _features_final")
        if total == 0:
            msg = (
                "The build produced no features; check the configuration's "
                "'features' blocks and input files"
            )
            raise ValueError(msg)

    def _compile_feature(
        self,
        connection: duckdb.DuckDBPyConnection,
        compiler: ProjectionCompiler,
        feature: FeatureConfig,
    ) -> None:
        """
        Run one feature block's queries, each reported as its own item.

        Each item advances the active stage as soon as it finishes, rather
        than only once every query for this feature block has run.
        """
        with item(f"Assembling features from {feature.source}", total=100) as bar:
            track(
                bar,
                connection.query_progress,
                lambda: connection.execute(
                    f"INSERT INTO _features {compiler.feature_query(feature)}"
                ),
            )
        advance()
        name_queries = compiler.name_queries(feature)
        # A feature block commonly collects several names (a primary name,
        # alternate names, ...), all from the same source, so this loop is
        # the most common source of identical-looking, back-to-back items;
        # number them whenever there is more than one, so a repeat reads as
        # a distinct new item rather than the previous one's bar restarting.
        name_suffixes = label_suffixes(
            [f"Collecting names from {feature.source}"] * len(name_queries)
        )
        for name_query, name_suffix in zip(name_queries, name_suffixes, strict=True):
            with item(
                f"Collecting names from {feature.source}{name_suffix}", total=100
            ) as bar:
                track(
                    bar,
                    connection.query_progress,
                    lambda nq=name_query: connection.execute(
                        f"INSERT INTO _names {nq}"
                    ),
                )
            advance()
        self._merge_duplicate_geometries(connection, compiler, feature)

    def _merge_duplicate_geometries(
        self,
        connection: duckdb.DuckDBPyConnection,
        compiler: ProjectionCompiler,
        feature: FeatureConfig,
    ) -> None:
        """
        Union the geometries of identifiers that repeat within a source.

        The feature query keeps only the first row's geometry (a bounded
        aggregate); here we recompute the union for the few genuinely
        duplicated identifiers and patch those rows. Doing this only for
        duplicates keeps ``ST_Union_Agg``'s unmanaged memory bounded, so the
        build stays within limits even for very large sources.
        """
        query = compiler.duplicate_geometry_query(feature)
        if query is None:
            return
        with item(f"Checking duplicates in {feature.source}", total=100) as bar:
            track(
                bar,
                connection.query_progress,
                lambda: connection.execute(
                    f"CREATE OR REPLACE TEMP TABLE _dup_geometry AS {query}"
                ),
            )
        advance()
        try:
            has_duplicates = scalar_int(
                connection, "SELECT count(*) FROM _dup_geometry"
            )
            if has_duplicates:
                with item(f"Merging duplicates in {feature.source}", total=100) as bar:
                    track(
                        bar,
                        connection.query_progress,
                        lambda: connection.execute(
                            "UPDATE _features SET geometry = ("
                            "SELECT geometry FROM _dup_geometry d "
                            "WHERE d.identifier = _features.identifier"
                            ") WHERE identifier IN "
                            "(SELECT identifier FROM _dup_geometry)"
                        ),
                    )
                advance()
        finally:
            connection.execute("DROP TABLE IF EXISTS _dup_geometry")

    def _check_duplicate_identifiers(
        self, connection: duckdb.DuckDBPyConnection
    ) -> None:
        """Fail the build when an identifier appears in multiple feature blocks."""
        duplicates = connection.execute(
            "SELECT identifier, string_agg(DISTINCT source, ', ') "
            "FROM _features GROUP BY identifier HAVING count(*) > 1 LIMIT 5"
        ).fetchall()
        if duplicates:
            examples = "; ".join(
                f"'{identifier}' (sources: {sources})"
                for identifier, sources in duplicates
            )
            msg = (
                "Identifiers must be unique across the whole gazetteer, but the "
                f"following appear in multiple feature blocks: {examples}. "
                "Merge the blocks or disambiguate the identifiers with an "
                "expression (e.g. a source prefix)."
            )
            raise ValueError(msg)

    def _build_artifact(
        self,
        connection: duckdb.DuckDBPyConnection,
        config: GazetteerConfig,
        build_dir: Path,
        target_path: Path,
    ) -> tuple[int, int]:
        """Write the artifact file and atomically move it into place."""
        temporary_path = build_dir / target_path.name
        # Keep SQLite's temporary files (FTS rebuild, index sorts, VACUUM) on
        # the build volume rather than the default location, which on some
        # systems is a RAM-backed tmpfs (e.g. WSL's /tmp) or a small system
        # drive (Windows %TEMP%) and would defeat the point of spilling to
        # disk. See ``_sqlite_tmpdir`` for the platform-specific env vars.
        sqlite_temp_dir = build_dir / "sqlite-temp"
        sqlite_temp_dir.mkdir(parents=True, exist_ok=True)
        with _sqlite_tmpdir(sqlite_temp_dir):
            sqlite_connection = create_artifact_db(temporary_path)
            try:
                with stage("Building artifact", "Built artifact", 6):
                    feature_count, name_count = emit(connection, sqlite_connection)
                    finalize(sqlite_connection, config, feature_count, name_count)
            finally:
                sqlite_connection.close()
        Path(temporary_path).replace(target_path)
        return feature_count, name_count


def uninstall(gazetteer_name: str) -> bool:
    """
    Remove an installed gazetteer artifact.

    Args:
        gazetteer_name: Name of the gazetteer to remove

    Returns:
        True if an artifact was removed, False if none was installed
    """
    path = artifact.artifact_path(gazetteer_name)
    if path.exists():
        path.unlink()
        return True
    return False
