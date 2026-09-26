"""
The user data directory must not move when the directory library changes.

Installed gazetteers live there, so a different path would silently orphan
them. These tests pin the directory appdirs used to resolve,
``user_data_dir("geoparser", "")``, on each platform.
"""

import ntpath
from pathlib import Path
from unittest.mock import patch

import pytest
from platformdirs.macos import MacOS
from platformdirs.unix import Unix

from geoparser.paths import geoparser_data_dir


@pytest.mark.unit
class TestGeoparserDataDir:
    def test_asks_for_geoparser_without_a_vendor_directory(self):
        with patch("geoparser.paths.user_data_dir", return_value="/x") as resolve:
            assert geoparser_data_dir() == Path("/x")

        resolve.assert_called_once_with("geoparser", appauthor=False)

    def test_linux_honours_xdg_data_home(self, monkeypatch):
        monkeypatch.setenv("XDG_DATA_HOME", "/data")

        assert Unix("geoparser", appauthor=False).user_data_dir == "/data/geoparser"

    def test_linux_defaults_to_local_share(self, monkeypatch):
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
        monkeypatch.setenv("HOME", "/home/u")

        assert (
            Unix("geoparser", appauthor=False).user_data_dir
            == "/home/u/.local/share/geoparser"
        )

    def test_macos_uses_application_support(self, monkeypatch):
        monkeypatch.setenv("HOME", "/Users/u")

        assert (
            MacOS("geoparser", appauthor=False).user_data_dir
            == "/Users/u/Library/Application Support/geoparser"
        )

    def test_windows_has_no_vendor_directory(self, monkeypatch):
        import platformdirs.windows as windows

        local = r"C:\Users\u\AppData\Local"
        monkeypatch.setattr(windows, "get_win_folder", lambda _csidl: local)
        monkeypatch.setattr(windows.os, "path", ntpath)

        resolved = windows.Windows("geoparser", appauthor=False).user_data_dir

        # appdirs gave <LOCALAPPDATA>\geoparser for appauthor="", not the
        # <LOCALAPPDATA>\geoparser\geoparser platformdirs gives for "".
        assert resolved == local + r"\geoparser"
