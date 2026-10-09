"""Crash-safe file writes shared by the benchmark scripts."""

from __future__ import annotations

import json
import os
import typing as t
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def write_json_atomic(
    path: Path, value: t.Any, *, pretty: bool = False, fsync: bool = True
) -> None:
    """
    Write JSON so that ``path`` only ever holds a complete document.

    The data goes to a sibling temporary file that is renamed over ``path``.
    With ``fsync`` the bytes are synced first, so a killed job cannot leave a
    renamed but empty file behind.

    Args:
        path: Destination, whose parent directory is created if missing
        value: The JSON-serializable value
        pretty: Human-readable output (2-space indent, sorted keys, UTF-8,
            trailing newline) instead of compact ASCII-escaped JSON
        fsync: Sync the data to disk before the rename
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if pretty:
        text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    else:
        text = json.dumps(value)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        if fsync:
            handle.flush()
            os.fsync(handle.fileno())
    temporary.replace(path)
