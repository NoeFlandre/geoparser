"""Filesystem locations shared across geoparser subsystems."""

from pathlib import Path

from appdirs import user_data_dir


def geoparser_data_dir() -> Path:
    """
    Return the per-user data directory for geoparser.

    Returns:
        Path to the geoparser user data directory
    """
    # The empty appauthor keeps the path free of a vendor directory on
    # Windows; on every other platform appdirs ignores it entirely, so
    # mutating it cannot change where the data lives here.
    return Path(user_data_dir("geoparser", ""))  # pragma: no mutate
