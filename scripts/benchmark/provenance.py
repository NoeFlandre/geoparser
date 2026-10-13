"""
Record what produced a set of numbers.

A benchmark result that cannot be traced back to a commit, a set of model
checkpoints and a corpus is not evidence, and on a shared cluster the run that
produced it is gone by the time anyone asks.
"""

from __future__ import annotations

import hashlib
import platform
import typing as t
from datetime import datetime, timezone
from pathlib import Path

from scripts import git_provenance


def gazetteer_artifact_sha256(name: str) -> str:
    """Return the SHA-256 identity of an installed gazetteer artifact."""
    from geoparser.gazetteer.artifact import artifact_path

    path = artifact_path(name)
    if not path.is_file():
        message = f"Gazetteer artifact not found: {path}"
        raise FileNotFoundError(message)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def source_commit(root: Path) -> str:
    """
    Return the commit the code is running from, or a marker when unknown.

    Args:
        root: The repository root

    Returns:
        The short commit hash, suffixed ``-dirty`` when the tree has changes
    """
    return git_provenance.commit_id(short=True, cwd=root)


def source_commit_full(root: Path) -> str:
    """Return the complete Git object ID recorded in the run manifest."""
    return git_provenance.commit_id(cwd=root)


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
