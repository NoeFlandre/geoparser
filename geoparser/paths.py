"""Filesystem locations shared across geoparser subsystems."""

import os
from pathlib import Path

from platformdirs import user_data_dir


def geoparser_data_dir() -> Path:
    """
    Return the per-user data directory for geoparser.

    Returns:
        Path to the geoparser user data directory
    """
    override = os.getenv("GEOPARSER_DATA_DIR")
    if override:
        return Path(override).expanduser()

    # appauthor=False keeps the path free of a vendor directory on Windows,
    # matching where the former appdirs call (with appauthor="") put the
    # data, so installed gazetteers are still found after the switch. Every
    # other platform ignores appauthor.
    return Path(user_data_dir("geoparser", appauthor=False))  # pragma: no mutate
