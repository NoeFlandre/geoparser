"""Check local MasakhaNER 2.0 files against their pinned git blob ids.

The check reads local files only. It never fetches anything, so it can run
offline. The local layout mirrors the upstream data directory: one folder per
configuration, holding train.txt, dev.txt and test.txt.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.masakhaner_benchmark.data import git_blob_sha1

OK = "ok"
MISSING = "missing"
MISMATCH = "mismatch"


def check_file(path: Path, expected_sha: str, expected_size: int) -> str:
    """Return ok, missing or mismatch for one local file."""
    if not path.is_file():
        return MISSING
    content = path.read_bytes()
    if len(content) != expected_size or git_blob_sha1(content) != expected_sha:
        return MISMATCH
    return OK


def check_directory(
    data_dir: Path, languages: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Check every pinned split file of every configuration under data_dir."""
    rows: list[dict[str, Any]] = []
    for language in languages:
        for split, pin in language["files"].items():
            local = data_dir / language["config"] / Path(pin["path"]).name
            status = check_file(local, pin["git_blob_sha"], pin["size_bytes"])
            rows.append(
                {
                    "config": language["config"],
                    "split": split,
                    "local_path": str(local),
                    "status": status,
                }
            )
    return rows
