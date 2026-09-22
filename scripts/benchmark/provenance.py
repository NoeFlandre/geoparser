"""
Record what produced a set of numbers.

A benchmark result that cannot be traced back to a commit, a set of model
checkpoints and a corpus is not evidence, and on a shared cluster the run that
produced it is gone by the time anyone asks.
"""

from __future__ import annotations

import platform
import subprocess
import typing as t
from datetime import datetime, timezone
from pathlib import Path


def source_commit(root: Path) -> str:
    """
    Return the commit the code is running from, or a marker when unknown.

    Args:
        root: The repository root

    Returns:
        The short commit hash, suffixed ``-dirty`` when the tree has changes
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{commit}-dirty" if dirty else commit


def environment(job_id: str | None = None) -> dict[str, t.Any]:
    """
    Return the facts about this run worth keeping next to its numbers.

    Args:
        job_id: The scheduler job this ran under, when there is one

    Returns:
        A JSON-safe description of the machine and the moment
    """
    facts: dict[str, t.Any] = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if job_id:
        facts["job_id"] = job_id
    try:
        import torch

        facts["torch"] = torch.__version__
        facts["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            facts["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:  # pragma: no cover - torch is a hard dependency
        pass
    return facts
