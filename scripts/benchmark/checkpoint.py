"""
Resumable checkpoints for a long benchmark run.

A Grid'5000 job ends when its walltime expires, whether or not the work is
finished, so a run that cannot resume is a run that must fit in one
reservation. This module keeps enough state to continue in a later job, and
refuses to resume from state that no longer describes the same experiment.

What makes a checkpoint resumable is not only the predictions but the
provenance next to them: the source commit, the models, the corpus digest and
the settings. If any of those differ, continuing would silently blend two
experiments into one set of numbers, so it is rejected instead.

Writes are atomic -- a temporary file replaced into place -- because a job
killed mid-write would otherwise leave a truncated file that parses as a
shorter, wrong run.
"""

from __future__ import annotations

import json
import os
import typing as t
from dataclasses import asdict, dataclass, field
from pathlib import Path

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RunIdentity:
    """Everything that must match for two partial runs to be one run."""

    pipeline: str
    corpus_digest: str
    gazetteer: str
    min_similarity: float
    limit: int | None
    commit: str
    schema_version: int = SCHEMA_VERSION

    def differences(self, other: RunIdentity) -> list[str]:
        """
        Return the fields on which this identity differs from another.

        Args:
            other: The identity recorded in an existing checkpoint

        Returns:
            Field names that differ, empty when the two describe one run
        """
        return [
            name
            for name, value in asdict(self).items()
            if value != asdict(other).get(name)
        ]


@dataclass
class Checkpoint:
    """Partial results for one pipeline, plus the identity they belong to."""

    identity: RunIdentity
    # Document identifier -> the annotations predicted for it. Keyed by
    # document rather than appended, so a document processed twice after an
    # interrupted write cannot be counted twice.
    recognition: dict[str, list[dict[str, t.Any]]] = field(default_factory=dict)
    resolution: dict[str, list[dict[str, t.Any]]] = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    def completed(self, phase: str) -> set[str]:
        """
        Return the identifiers of documents already done in a phase.

        Args:
            phase: Either ``recognition`` or ``resolution``

        Returns:
            The document identifiers already recorded
        """
        return set(self._phase(phase))

    def record(
        self, phase: str, document_id: str, annotations: list[dict[str, t.Any]]
    ) -> None:
        """
        Store one document's predictions for a phase.

        Args:
            phase: Either ``recognition`` or ``resolution``
            document_id: The document these annotations belong to
            annotations: The serialized annotations
        """
        self._phase(phase)[document_id] = annotations

    def annotations(self, phase: str) -> list[dict[str, t.Any]]:
        """Return every recorded annotation for a phase, document order aside."""
        return [
            annotation
            for document_annotations in self._phase(phase).values()
            for annotation in document_annotations
        ]

    def _phase(self, phase: str) -> dict[str, list[dict[str, t.Any]]]:
        """Return the mapping for a phase, rejecting an unknown name."""
        if phase == "recognition":
            return self.recognition
        if phase == "resolution":
            return self.resolution
        raise ValueError(
            f"Unknown phase {phase!r}; expected 'recognition' or 'resolution'."
        )


def save(path: Path, checkpoint: Checkpoint) -> None:
    """
    Write a checkpoint atomically.

    The data is flushed and synced before the rename, so the file that appears
    at ``path`` is always a complete one even if the job is killed during the
    write.

    Args:
        path: Where the checkpoint lives
        checkpoint: The state to persist
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "identity": asdict(checkpoint.identity),
        "recognition": checkpoint.recognition,
        "resolution": checkpoint.resolution,
        "elapsed_seconds": checkpoint.elapsed_seconds,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def load(path: Path, identity: RunIdentity) -> tuple[Checkpoint, list[str]]:
    """
    Load a checkpoint to continue, or start a fresh one.

    A missing, unreadable or mismatched checkpoint yields a fresh one rather
    than an error: the run should proceed, just without resuming. The reasons
    are returned so the caller can say why it started over.

    Args:
        path: Where the checkpoint lives
        identity: The identity this run requires

    Returns:
        The checkpoint to use, and the reasons a previous one was not resumed
    """
    if not path.exists():
        return Checkpoint(identity), []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        stored = RunIdentity(**payload["identity"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        return Checkpoint(identity), [f"unreadable checkpoint ({error})"]

    differences = identity.differences(stored)
    if differences:
        return Checkpoint(identity), [f"changed: {', '.join(differences)}"]

    return (
        Checkpoint(
            identity,
            recognition=payload.get("recognition", {}),
            resolution=payload.get("resolution", {}),
            elapsed_seconds=payload.get("elapsed_seconds", 0.0),
        ),
        [],
    )
