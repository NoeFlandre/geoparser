"""Identity-checked, atomic per-model/language benchmark checkpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from scripts._io import write_json_atomic


def require_clean_commit(commit: str) -> str:
    """Reject benchmark provenance that cannot identify an immutable tree."""
    if commit == "unknown" or commit.endswith("-dirty"):
        message = "PAN-X evaluation requires a clean, identifiable repository commit"
        raise ValueError(message)
    return commit


def _canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class ModelLanguageCheckpoints:
    """Keep resumable language scores bound to one immutable run identity."""

    def __init__(self, base_directory: Path, identity: dict[str, Any]):
        self.snapshot_id = hashlib.sha256(
            _canonical_json(identity).encode()
        ).hexdigest()
        self.directory = base_directory / self.snapshot_id
        self.directory.mkdir(parents=True, exist_ok=True)
        self.manifest = {"snapshot_id": self.snapshot_id, **identity}
        manifest_path = self.directory / "snapshot.json"
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing != self.manifest:
                message = (
                    "Checkpoint manifest does not match the requested run snapshot"
                )
                raise ValueError(message)
        elif any(self.directory.iterdir()):
            message = "Checkpoint directory has data but no run snapshot manifest"
            raise ValueError(message)
        else:
            write_json_atomic(manifest_path, self.manifest, pretty=True, fsync=False)

    def _language_path(self, model_key: str, language: str) -> Path:
        return self.directory / "languages" / model_key / f"{language}.json"

    def _language_identity(
        self,
        model_key: str,
        model_id: str,
        model_revision: str | None,
        language: str,
        source_example_count: int,
    ) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "model_key": model_key,
            "model_id": model_id,
            "model_revision": model_revision,
            "language": language,
            "source_test_examples": source_example_count,
        }

    def load_language(
        self,
        model_key: str,
        model_id: str,
        model_revision: str | None,
        language: str,
        source_example_count: int,
    ) -> dict[str, Any] | None:
        """Load a completed language result, rejecting records from other runs."""
        path = self._language_path(model_key, language)
        if not path.exists():
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
        identity = self._language_identity(
            model_key, model_id, model_revision, language, source_example_count
        )
        if any(record.get(key) != value for key, value in identity.items()):
            message = f"Checkpoint identity mismatch for {model_key}/{language}"
            raise ValueError(message)
        return record["result"]

    def save_language(
        self,
        model_key: str,
        model_id: str,
        model_revision: str | None,
        language: str,
        source_example_count: int,
        result: dict[str, Any],
    ) -> None:
        """Atomically store one completed model/language result."""
        record = self._language_identity(
            model_key, model_id, model_revision, language, source_example_count
        )
        record["result"] = result
        write_json_atomic(
            self._language_path(model_key, language), record, pretty=True, fsync=False
        )
