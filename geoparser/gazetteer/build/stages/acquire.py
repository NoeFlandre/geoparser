"""
Acquisition of gazetteer source files.

Handles downloading remote files (with validator-based caching), validating
local paths, extracting ZIP archives and locating the target file within
extracted contents or directories.
"""

import hashlib
import json
import os
import shutil
import typing as t
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import requests

from geoparser.gazetteer.build.progress import advance, item
from geoparser.gazetteer.build.schema import SourceConfig

# Network request timeout in seconds
REQUEST_TIMEOUT = 30

# Download chunk size in bytes (8KB)
DOWNLOAD_CHUNK_SIZE = 8192


class Acquirer:
    """
    Downloads and extracts gazetteer source files.

    Remote files are cached in the downloads directory and skipped when the
    local copy matches the server's ETag or Last-Modified validator. Downloads
    are streamed to a partial file and published only when complete. ZIP
    archives are extracted next to the archive, with extraction skipped when
    contents are up to date. Each download or extraction that actually runs
    shows its own item bar and advances the active stage (see :mod:`progress`)
    once it finishes; a cached, unzipped local file shows neither and advances
    nothing.
    """

    def __init__(self, downloads_directory: Path):
        """
        Initialize the acquirer.

        Args:
            downloads_directory: Directory to store downloaded files
        """
        self.downloads_directory = downloads_directory
        self.downloads_directory.mkdir(parents=True, exist_ok=True)

    def acquire(self, source_config: SourceConfig) -> Path:
        """
        Resolve the data file for a source, downloading and extracting as needed.

        Args:
            source_config: Source configuration

        Returns:
            Path to the source's target file
        """
        source_path = self._resolve_source_path(source_config)
        return self._resolve_file_path(source_config, source_path)

    def cleanup(self) -> None:
        """Remove all downloaded files and extracted contents."""
        if self.downloads_directory.exists():
            shutil.rmtree(self.downloads_directory)

    def _resolve_source_path(self, source_config: SourceConfig) -> Path:
        """Download the source's file or validate its local path."""
        if source_config.url:
            return self._download_file(source_config.url, source_config.sha256)
        if source_config.path is None:  # pragma: no cover - validate_source
            raise ValueError(f"Source '{source_config.name}' has neither url nor path")
        local_path = Path(source_config.path)
        if not local_path.exists():
            raise FileNotFoundError(f"Local path does not exist: {local_path}")
        if source_config.sha256 is not None:
            if not local_path.is_file():
                raise ValueError("sha256 can only be verified for a source file")
            self._verify_sha256(local_path, source_config.sha256)
        return local_path

    def _download_file(self, url: str, sha256: str | None = None) -> Path:
        """Download a file unless a matching local copy already exists."""
        filename = Path(urlparse(url).path).name or "download"
        download_path = self.downloads_directory / filename
        if self._should_skip_download(url, download_path) and (
            sha256 is None or self._matches_sha256(download_path, sha256)
        ):
            return download_path
        return self._stream_download(url, download_path, sha256)

    def _should_skip_download(self, url: str, local_path: Path) -> bool:
        """Check if a remote validator still identifies the cached file."""
        if not local_path.is_file():
            return False
        try:
            response = requests.head(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            etag = response.headers.get("ETag")
            last_modified = response.headers.get("Last-Modified")
            if etag:
                validator_key, validator_value = "etag", etag
            elif last_modified:
                validator_key, validator_value = "last_modified", last_modified
            else:
                return False

            metadata = self._read_download_metadata(local_path)
            if metadata is None or metadata.get(validator_key) != validator_value:
                return False

            local_size = local_path.stat().st_size
            recorded_size = metadata.get("content_length")
            if not isinstance(recorded_size, int) or recorded_size != local_size:
                return False

            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                remote_size = int(content_length)
                if remote_size < 0 or remote_size != local_size:
                    return False
            return True
        except (requests.RequestException, OSError, ValueError):
            # If the HEAD request fails, proceed with the download
            return False

    def _stream_download(
        self, url: str, download_path: Path, sha256: str | None = None
    ) -> Path:
        """Stream, validate, then atomically publish a file download."""
        partial_path = self._partial_path(download_path)
        digest = hashlib.sha256() if sha256 is not None else None
        received_size = 0

        try:
            with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT) as response:
                response.raise_for_status()
                content_length = response.headers.get("Content-Length")
                total_size = int(content_length) if content_length is not None else 0
                if total_size < 0:
                    raise ValueError("Content-Length cannot be negative")

                with (
                    partial_path.open("wb") as output_file,
                    item(
                        f"Downloading {download_path.name}", total=total_size or None
                    ) as progress_bar,
                ):
                    for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                        if chunk:
                            output_file.write(chunk)
                            received_size += len(chunk)
                            if digest is not None:
                                digest.update(chunk)
                            progress_bar.update(len(chunk))

                if content_length is not None and received_size != total_size:
                    raise ValueError(
                        f"Downloaded {received_size} bytes but Content-Length "
                        f"declared {total_size} bytes"
                    )

                if digest is not None and digest.hexdigest() != sha256:
                    raise ValueError(
                        f"SHA-256 mismatch for '{url}': expected {sha256}, "
                        f"got {digest.hexdigest()}"
                    )

                os.replace(partial_path, download_path)
                self._write_download_metadata(
                    download_path,
                    {
                        "etag": response.headers.get("ETag"),
                        "last_modified": response.headers.get("Last-Modified"),
                        "content_length": received_size,
                    },
                )
        finally:
            partial_path.unlink(missing_ok=True)

        advance()

        return download_path

    @staticmethod
    def _partial_path(path: Path) -> Path:
        """Return the sibling temporary path used before atomic replacement."""
        return path.with_name(f"{path.name}.part")

    @classmethod
    def _metadata_path(cls, path: Path) -> Path:
        """Return the validator sidecar path for a downloaded file."""
        return path.with_name(f"{path.name}.meta")

    @classmethod
    def _read_download_metadata(cls, path: Path) -> dict[str, t.Any] | None:
        """Read a download's validator sidecar, treating invalid data as stale."""
        try:
            metadata = json.loads(cls._metadata_path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return metadata if isinstance(metadata, dict) else None

    @classmethod
    def _write_download_metadata(
        cls, path: Path, metadata: dict[str, str | int | None]
    ) -> None:
        """Atomically persist the server validators associated with a download."""
        metadata_path = cls._metadata_path(path)
        partial_path = cls._partial_path(metadata_path)
        try:
            partial_path.write_text(
                json.dumps(metadata, sort_keys=True), encoding="utf-8"
            )
            os.replace(partial_path, metadata_path)
        finally:
            partial_path.unlink(missing_ok=True)

    @staticmethod
    def _sha256_file(path: Path) -> str:
        """Return the SHA-256 digest of a file, reading bounded chunks."""
        digest = hashlib.sha256()
        with path.open("rb") as source_file:
            for chunk in iter(lambda: source_file.read(DOWNLOAD_CHUNK_SIZE), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def _matches_sha256(cls, path: Path, expected: str) -> bool:
        """Whether a local file matches its optional expected SHA-256 digest."""
        try:
            return cls._sha256_file(path) == expected
        except OSError:
            return False

    @classmethod
    def _verify_sha256(cls, path: Path, expected: str) -> None:
        """Raise a clear error when a source file does not match its digest."""
        actual = cls._sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"SHA-256 mismatch for '{path}': expected {expected}, got {actual}"
            )

    def _resolve_file_path(
        self, source_config: SourceConfig, source_path: Path
    ) -> Path:
        """
        Locate the target file from a source path.

        Handles directories (recursive search), ZIP archives (extraction) and
        raw files (name check).
        """
        target_filename = source_config.file

        if source_path.is_dir():
            return self._find_target_file(source_path, target_filename)

        if not zipfile.is_zipfile(source_path):
            if source_path.name == target_filename:
                return source_path
            raise FileNotFoundError(
                f"Source '{source_config.name}': file '{target_filename}' not "
                f"found at {source_path}"
            )

        extraction_dir = source_path.parent / source_path.stem
        if self._should_skip_extraction(source_path, extraction_dir, target_filename):
            return self._find_target_file(extraction_dir, target_filename)
        return self._extract_zip(source_path, extraction_dir, target_filename)

    def _should_skip_extraction(
        self, archive_path: Path, extraction_dir: Path, target_filename: str
    ) -> bool:
        """Check if a previous extraction is still up to date."""
        if not extraction_dir.exists():
            return False

        # Local directories are already in their final form
        if not archive_path.is_file():
            return True

        if extraction_dir.name == target_filename:
            return self._is_fresh(archive_path, extraction_dir)

        target_path = self._find_target_file_quiet(extraction_dir, target_filename)
        return target_path is not None and self._is_fresh(archive_path, target_path)

    @staticmethod
    def _is_fresh(archive_path: Path, extracted: Path) -> bool:
        """
        Whether an extracted path is at least as new as its archive.

        Args:
            archive_path: The archive it came from
            extracted: The extracted file or directory

        Returns:
            True when the extraction does not need redoing
        """
        return archive_path.stat().st_mtime <= extracted.stat().st_mtime

    @staticmethod
    def _unpack_zip(archive_path: Path, extraction_dir: Path) -> None:
        """
        Extract every member of a ZIP, reporting progress by uncompressed size.

        Args:
            archive_path: The archive to unpack
            extraction_dir: Directory to unpack into
        """
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            total_size = sum(info.file_size for info in zip_ref.infolist())
            with item(
                f"Unpacking {archive_path.name}", total=total_size or None
            ) as progress_bar:
                for zip_info in zip_ref.infolist():
                    zip_ref.extract(zip_info, path=extraction_dir)
                    progress_bar.update(zip_info.file_size)

    def _extract_zip(
        self, archive_path: Path, extraction_dir: Path, target_filename: str
    ) -> Path:
        """Extract a ZIP archive and locate the target file."""
        if extraction_dir.exists():
            shutil.rmtree(extraction_dir)
        extraction_dir.mkdir(exist_ok=True)

        self._unpack_zip(archive_path, extraction_dir)
        advance()

        if extraction_dir.name == target_filename:
            return extraction_dir
        return self._find_target_file(extraction_dir, target_filename)

    def _find_target_file(self, directory: Path, filename: str) -> Path:
        """Find a file by name in a directory tree."""
        for path in directory.glob("**/*"):
            if path.name == filename:
                return path
        raise FileNotFoundError(f"File '{filename}' not found in {directory}")

    def _find_target_file_quiet(self, directory: Path, filename: str) -> Path | None:
        """Find a file by name, returning None if not found."""
        try:
            return self._find_target_file(directory, filename)
        except FileNotFoundError:
            return None
