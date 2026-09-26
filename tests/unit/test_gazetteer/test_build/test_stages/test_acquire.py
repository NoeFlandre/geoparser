"""
Unit tests for geoparser/gazetteer/build/stages/acquire.py

Tests source resolution, downloading (with validator-based caching) and ZIP
extraction directly against the Acquirer's methods, using tmp_path for local
files/archives and requests_mock for HTTP.
"""

import hashlib
import json
import os
import time
import zipfile

import pytest
import requests

from geoparser.gazetteer.build.schema import SourceConfig
from geoparser.gazetteer.build.stages.acquire import Acquirer


def make_source(**overrides) -> SourceConfig:
    """A minimal valid tabular SourceConfig, with fields overridable."""
    data = {
        "name": "places",
        "path": "/tmp/unused.csv",
        "file": "places.csv",
        "delimiter": ",",
        "attributes": [{"name": "id", "type": "integer"}],
    }
    data.update(overrides)
    return SourceConfig.model_validate(data)


@pytest.fixture
def acquirer(tmp_path) -> Acquirer:
    return Acquirer(tmp_path / "downloads")


@pytest.mark.unit
class TestResolveSourcePath:
    """Test Acquirer._resolve_source_path()."""

    def test_raises_when_local_path_missing(self, acquirer, tmp_path):
        """A local source whose path doesn't exist raises FileNotFoundError."""
        source = make_source(path=str(tmp_path / "missing.csv"))

        with pytest.raises(FileNotFoundError, match="Local path does not exist"):
            acquirer._resolve_source_path(source)

    def test_returns_existing_local_path(self, acquirer, tmp_path):
        """A local source whose path exists is returned as-is."""
        data_file = tmp_path / "places.csv"
        data_file.write_text("1,Paris\n")
        source = make_source(path=str(data_file))

        assert acquirer._resolve_source_path(source) == data_file

    def test_remote_source_downloads_the_file(self, acquirer, requests_mock):
        """A source with a URL is downloaded rather than read locally."""
        requests_mock.get("https://example.com/places.csv", content=b"1,Paris\n")
        source = make_source(url="https://example.com/places.csv", path=None)

        path = acquirer._resolve_source_path(source)

        assert path.read_bytes() == b"1,Paris\n"


@pytest.mark.unit
class TestDownload:
    """Test Acquirer._download_file() and its caching decision."""

    def test_downloads_when_no_local_copy_exists(self, acquirer, requests_mock):
        """A file with no cached local copy is downloaded."""
        requests_mock.get("https://example.com/places.csv", content=b"1,Paris\n")

        path = acquirer._download_file("https://example.com/places.csv")

        assert path.read_bytes() == b"1,Paris\n"

    def test_persists_response_validators_in_json_sidecar(
        self, acquirer, requests_mock
    ):
        """A completed download records validators and its received byte count."""
        url = "https://example.com/places.csv"
        content = b"1,Paris\n"
        last_modified = "Mon, 01 Jan 2024 00:00:00 GMT"
        requests_mock.get(
            url,
            content=content,
            headers={
                "ETag": '"v1"',
                "Last-Modified": last_modified,
                "Content-Length": str(len(content)),
            },
        )

        path = acquirer._download_file(url)

        metadata = json.loads(
            path.with_name(f"{path.name}.meta").read_text(encoding="utf-8")
        )
        assert metadata == {
            "content_length": len(content),
            "etag": '"v1"',
            "last_modified": last_modified,
        }

    def test_skips_download_when_validators_match(self, acquirer, requests_mock):
        """A cached file with a matching ETag is not re-downloaded."""
        content = b"1,Paris\n"
        download_path = acquirer.downloads_directory / "places.csv"
        download_path.write_bytes(content)
        download_path.with_name(f"{download_path.name}.meta").write_text(
            json.dumps(
                {
                    "etag": '"v1"',
                    "last_modified": None,
                    "content_length": len(content),
                }
            )
        )
        requests_mock.head(
            "https://example.com/places.csv",
            headers={"content-length": str(len(content)), "etag": '"v1"'},
        )
        get_mock = requests_mock.get(
            "https://example.com/places.csv", content=b"SHOULD NOT BE FETCHED"
        )

        path = acquirer._download_file("https://example.com/places.csv")

        assert path.read_bytes() == content
        assert not get_mock.called

    def test_redownloads_same_size_file_when_etag_changes(
        self, acquirer, requests_mock
    ):
        """A new ETag invalidates a cached file even when its size is unchanged."""
        url = "https://example.com/places.csv"
        content = b"OLD"
        download_path = acquirer.downloads_directory / "places.csv"
        download_path.write_bytes(content)
        download_path.with_name(f"{download_path.name}.meta").write_text(
            json.dumps(
                {
                    "etag": '"v1"',
                    "last_modified": None,
                    "content_length": len(content),
                }
            )
        )
        requests_mock.head(
            url,
            headers={"content-length": str(len(content)), "etag": '"v2"'},
        )
        get_mock = requests_mock.get(url, content=b"NEW", headers={"etag": '"v2"'})

        path = acquirer._download_file(url)

        assert get_mock.called
        assert path.read_bytes() == b"NEW"

    def test_redownloads_same_size_file_when_last_modified_changes(
        self, acquirer, requests_mock
    ):
        """Last-Modified invalidates a cache when the server has no ETag."""
        url = "https://example.com/places.csv"
        content = b"OLD"
        download_path = acquirer.downloads_directory / "places.csv"
        download_path.write_bytes(content)
        download_path.with_name(f"{download_path.name}.meta").write_text(
            json.dumps(
                {
                    "etag": None,
                    "last_modified": "Mon, 01 Jan 2024 00:00:00 GMT",
                    "content_length": len(content),
                }
            )
        )
        requests_mock.head(
            url,
            headers={
                "content-length": str(len(content)),
                "last-modified": "Tue, 02 Jan 2024 00:00:00 GMT",
            },
        )
        get_mock = requests_mock.get(url, content=b"NEW")

        path = acquirer._download_file(url)

        assert get_mock.called
        assert path.read_bytes() == b"NEW"

    def test_redownloads_when_cached_file_size_differs_from_sidecar(
        self, acquirer, requests_mock
    ):
        """A matching validator does not reuse a locally changed cache file."""
        url = "https://example.com/places.csv"
        download_path = acquirer.downloads_directory / "places.csv"
        download_path.write_bytes(b"ALTERED")
        download_path.with_name(f"{download_path.name}.meta").write_text(
            json.dumps(
                {
                    "etag": '"v1"',
                    "last_modified": None,
                    "content_length": len(b"OLD"),
                }
            )
        )
        requests_mock.head(url, headers={"etag": '"v1"'})
        get_mock = requests_mock.get(url, content=b"NEW", headers={"etag": '"v1"'})

        path = acquirer._download_file(url)

        assert get_mock.called
        assert path.read_bytes() == b"NEW"

    def test_query_string_does_not_become_part_of_download_filename(
        self, acquirer, requests_mock
    ):
        """The cache filename comes from the URL path without its query."""
        url = "https://example.com/data/places.csv?token=secret"
        requests_mock.get(url, content=b"1,Paris\n")

        path = acquirer._download_file(url)

        assert path.name == "places.csv"

    def test_interrupted_stream_does_not_publish_partial_file(
        self, acquirer, monkeypatch
    ):
        """A failed response leaves neither a final file nor a .part file."""
        download_path = acquirer.downloads_directory / "places.csv"

        class InterruptedResponse:
            def __init__(self):
                self.headers = {"Content-Length": "8", "ETag": '"v1"'}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def raise_for_status(self):
                return None

            def iter_content(self, chunk_size):
                yield b"part"
                raise requests.ConnectionError("connection dropped")

        monkeypatch.setattr(
            requests, "get", lambda *args, **kwargs: InterruptedResponse()
        )

        with pytest.raises(requests.ConnectionError, match="connection dropped"):
            acquirer._stream_download("https://example.com/places.csv", download_path)

        assert not download_path.exists()
        assert not download_path.with_name(f"{download_path.name}.part").exists()

    def test_content_length_mismatch_does_not_publish_file(self, acquirer, monkeypatch):
        """The received byte count must match a declared Content-Length."""
        download_path = acquirer.downloads_directory / "places.csv"

        class ShortResponse:
            def __init__(self):
                self.headers = {"Content-Length": "4"}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def raise_for_status(self):
                return None

            def iter_content(self, chunk_size):
                yield b"abc"

        monkeypatch.setattr(requests, "get", lambda *args, **kwargs: ShortResponse())

        with pytest.raises(ValueError, match="Content-Length"):
            acquirer._stream_download("https://example.com/places.csv", download_path)

        assert not download_path.exists()
        assert not download_path.with_name(f"{download_path.name}.part").exists()

    def test_sha256_mismatch_does_not_publish_download(self, acquirer, requests_mock):
        """A configured SHA-256 mismatch fails the acquisition atomically."""
        url = "https://example.com/places.csv"
        expected_sha256 = hashlib.sha256(b"right").hexdigest()
        requests_mock.get(url, content=b"wrong")
        source = make_source(url=url, path=None, sha256=expected_sha256)

        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            acquirer.acquire(source)

        download_path = acquirer.downloads_directory / "places.csv"
        assert not download_path.exists()
        assert not download_path.with_name(f"{download_path.name}.part").exists()

    def test_redownloads_when_size_differs(self, acquirer, requests_mock):
        """A cached file whose size no longer matches is re-downloaded."""
        download_path = acquirer.downloads_directory / "places.csv"
        download_path.write_bytes(b"OLD")
        requests_mock.head(
            "https://example.com/places.csv", headers={"content-length": "999"}
        )
        requests_mock.get("https://example.com/places.csv", content=b"NEW-CONTENT")

        path = acquirer._download_file("https://example.com/places.csv")

        assert path.read_bytes() == b"NEW-CONTENT"

    def test_redownloads_when_head_request_fails(self, acquirer, requests_mock):
        """If the HEAD request fails, the download proceeds anyway."""
        download_path = acquirer.downloads_directory / "places.csv"
        download_path.write_bytes(b"OLD")
        requests_mock.head(
            "https://example.com/places.csv", exc=requests.ConnectionError
        )
        requests_mock.get("https://example.com/places.csv", content=b"NEW-CONTENT")

        path = acquirer._download_file("https://example.com/places.csv")

        assert path.read_bytes() == b"NEW-CONTENT"

    def test_redownloads_when_content_length_missing(self, acquirer, requests_mock):
        """A HEAD response with no content-length is treated as size 0 (no match)."""
        download_path = acquirer.downloads_directory / "places.csv"
        download_path.write_bytes(b"OLD")
        requests_mock.head("https://example.com/places.csv")
        requests_mock.get("https://example.com/places.csv", content=b"NEW-CONTENT")

        path = acquirer._download_file("https://example.com/places.csv")

        assert path.read_bytes() == b"NEW-CONTENT"

    def test_streamed_download_reports_progress_without_content_length(
        self, acquirer, requests_mock
    ):
        """A response without content-length still downloads correctly."""
        requests_mock.get("https://example.com/places.csv", content=b"1,Paris\n")

        path = acquirer._stream_download(
            "https://example.com/places.csv",
            acquirer.downloads_directory / "places.csv",
        )

        assert path.read_bytes() == b"1,Paris\n"


@pytest.mark.unit
class TestResolveFilePath:
    """Test Acquirer._resolve_file_path()."""

    def test_finds_file_in_directory_tree(self, acquirer, tmp_path):
        """A directory source path is searched recursively for the target file."""
        directory = tmp_path / "extracted"
        (directory / "nested").mkdir(parents=True)
        target = directory / "nested" / "places.csv"
        target.write_text("data")
        source = make_source(file="places.csv")

        assert acquirer._resolve_file_path(source, directory) == target

    def test_returns_matching_plain_file(self, acquirer, tmp_path):
        """A plain (non-ZIP) file matching the target name is returned as-is."""
        file_path = tmp_path / "places.csv"
        file_path.write_text("data")
        source = make_source(file="places.csv")

        assert acquirer._resolve_file_path(source, file_path) == file_path

    def test_raises_when_plain_file_name_mismatches(self, acquirer, tmp_path):
        """A plain file with an unexpected name raises FileNotFoundError."""
        file_path = tmp_path / "other.csv"
        file_path.write_text("data")
        source = make_source(file="places.csv")

        with pytest.raises(FileNotFoundError, match="not found at"):
            acquirer._resolve_file_path(source, file_path)

    def test_extracts_zip_and_finds_target_file(self, acquirer, tmp_path):
        """A ZIP archive is extracted and its target file located."""
        archive_path = tmp_path / "places.zip"
        with zipfile.ZipFile(archive_path, "w") as zip_file:
            zip_file.writestr("places.csv", "1,Paris\n")
        source = make_source(file="places.csv")

        result = acquirer._resolve_file_path(source, archive_path)

        assert result.name == "places.csv"
        assert result.read_text() == "1,Paris\n"

    def test_skips_extraction_on_second_call(self, acquirer, tmp_path):
        """A second call reuses the already-extracted, up-to-date contents."""
        archive_path = tmp_path / "places.zip"
        with zipfile.ZipFile(archive_path, "w") as zip_file:
            zip_file.writestr("places.csv", "1,Paris\n")
        source = make_source(file="places.csv")
        first = acquirer._resolve_file_path(source, archive_path)

        second = acquirer._resolve_file_path(source, archive_path)

        assert second == first

    def test_extraction_dir_matching_target_name_is_returned_directly(
        self, acquirer, tmp_path
    ):
        """When the extraction dir's own name is the target file, it is returned."""
        archive_path = tmp_path / "places.csv.zip"
        with zipfile.ZipFile(archive_path, "w") as zip_file:
            zip_file.writestr("contents.txt", "irrelevant")
        source = make_source(file="places.csv")

        result = acquirer._resolve_file_path(source, archive_path)

        assert result == archive_path.parent / "places.csv"


@pytest.mark.unit
class TestShouldSkipExtraction:
    """Test Acquirer._should_skip_extraction()."""

    def test_false_when_extraction_dir_missing(self, acquirer, tmp_path):
        """No prior extraction means extraction cannot be skipped."""
        archive_path = tmp_path / "archive.zip"
        archive_path.touch()
        extraction_dir = tmp_path / "missing_dir"

        assert (
            acquirer._should_skip_extraction(archive_path, extraction_dir, "x.csv")
            is False
        )

    def test_true_when_archive_path_is_not_a_file(self, acquirer, tmp_path):
        """A local directory 'archive' is treated as already fully extracted."""
        archive_path = tmp_path / "already_a_directory"
        archive_path.mkdir()
        extraction_dir = tmp_path / "extraction_target"
        extraction_dir.mkdir()

        assert (
            acquirer._should_skip_extraction(archive_path, extraction_dir, "x.csv")
            is True
        )

    def test_false_when_target_file_is_absent(self, acquirer, tmp_path):
        """An extraction dir that doesn't contain the target file is stale."""
        archive_path = tmp_path / "archive.zip"
        archive_path.touch()
        extraction_dir = tmp_path / "extraction_target"
        extraction_dir.mkdir()

        assert (
            acquirer._should_skip_extraction(
                archive_path, extraction_dir, "missing.csv"
            )
            is False
        )

    def test_true_when_extraction_dir_itself_is_the_up_to_date_target(
        self, acquirer, tmp_path
    ):
        """An extraction dir named after the target file, newer than the
        archive, is treated as already up to date."""
        archive_path = tmp_path / "archive.zip"
        archive_path.touch()
        extraction_dir = tmp_path / "places.csv"
        extraction_dir.mkdir()
        future = time.time() + 10
        os.utime(extraction_dir, (future, future))

        assert (
            acquirer._should_skip_extraction(archive_path, extraction_dir, "places.csv")
            is True
        )


@pytest.mark.unit
class TestExtractZip:
    """Test Acquirer._extract_zip()."""

    def test_removes_stale_extraction_before_re_extracting(self, acquirer, tmp_path):
        """A pre-existing (stale) extraction directory is removed and redone."""
        archive_path = tmp_path / "places.zip"
        with zipfile.ZipFile(archive_path, "w") as zip_file:
            zip_file.writestr("places.csv", "1,Paris\n")
        source = make_source(file="places.csv")
        acquirer._resolve_file_path(source, archive_path)
        extraction_dir = archive_path.parent / archive_path.stem
        stale_marker = extraction_dir / "stale.txt"
        stale_marker.write_text("old")
        # Make the archive newer than the extraction, so it is seen as stale.
        future = time.time() + 10
        os.utime(archive_path, (future, future))

        acquirer._resolve_file_path(source, archive_path)

        assert not stale_marker.exists()


@pytest.mark.unit
class TestFindTargetFile:
    """Test Acquirer._find_target_file() and _find_target_file_quiet()."""

    def test_raises_when_file_not_found(self, acquirer, tmp_path):
        """_find_target_file raises FileNotFoundError when nothing matches."""
        directory = tmp_path / "empty"
        directory.mkdir()

        with pytest.raises(FileNotFoundError, match="not found in"):
            acquirer._find_target_file(directory, "missing.csv")

    def test_quiet_variant_returns_none_when_not_found(self, acquirer, tmp_path):
        """_find_target_file_quiet swallows the error and returns None."""
        directory = tmp_path / "empty"
        directory.mkdir()

        assert acquirer._find_target_file_quiet(directory, "missing.csv") is None

    def test_quiet_variant_returns_path_when_found(self, acquirer, tmp_path):
        """_find_target_file_quiet returns the path when the file exists."""
        directory = tmp_path / "dir"
        directory.mkdir()
        target = directory / "found.csv"
        target.write_text("data")

        assert acquirer._find_target_file_quiet(directory, "found.csv") == target
