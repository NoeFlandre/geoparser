"""Read the current Git commit and whether its working tree is dirty."""

from __future__ import annotations

import subprocess
from pathlib import Path


def commit_id(*, short: bool = False, cwd: Path | None = None) -> str:
    """Return the commit ID with a dirty suffix, or ``unknown`` on Git errors."""
    revision_arguments = ["git", "rev-parse"]
    if short:
        revision_arguments.append("--short")
    revision_arguments.append("HEAD")
    try:
        commit = subprocess.run(
            revision_arguments,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{commit}-dirty" if dirty else commit
