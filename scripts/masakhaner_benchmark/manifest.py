"""Read and validate the pinned MasakhaNER 2.0 inventory."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from scripts.masakhaner_benchmark.data import LABEL_NAMES

PACKAGE_DIR = Path(__file__).parent
MANIFEST_PATH = PACKAGE_DIR / "manifest.json"
SPLIT_NAMES = ("train", "validation", "test")
_HEX_40 = re.compile(r"^[0-9a-f]{40}$")


def read_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    """Read the checked-in inventory and reject it if its pins are malformed."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        message = f"Expected a JSON object in {path}"
        raise TypeError(message)
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Check the structure and the pins that the rest of the package relies on."""
    if manifest.get("schema_version") != 1:
        message = "Unsupported MasakhaNER manifest schema version"
        raise ValueError(message)
    dataset = manifest["dataset"]
    if not _HEX_40.match(dataset["huggingface_revision"]):
        message = "The Hugging Face revision must be a 40-character commit id"
        raise ValueError(message)
    if not _HEX_40.match(dataset["github_commit"]):
        message = "The GitHub commit must be a 40-character commit id"
        raise ValueError(message)
    if manifest["label_names"] != list(LABEL_NAMES):
        message = "The manifest label names differ from the parser's label set"
        raise ValueError(message)
    languages = manifest["languages"]
    if len(languages) != 20:
        message = "The MasakhaNER 2.0 manifest must list 20 configurations"
        raise ValueError(message)
    _check_unique(languages, "config")
    _check_unique([row for row in languages if row["iso639_1"]], "iso639_1")
    for row in languages:
        _validate_language(row, dataset)


def _check_unique(rows: list[dict[str, Any]], key: str) -> None:
    """Reject duplicate identifiers under one key."""
    values = [row[key] for row in rows]
    if len(set(values)) != len(values):
        message = f"Duplicate {key} values in the MasakhaNER manifest"
        raise ValueError(message)


def _validate_language(row: dict[str, Any], dataset: dict[str, Any]) -> None:
    """Check one configuration's counts and file pins."""
    if sorted(row["readme_counts"]) != sorted(SPLIT_NAMES):
        message = f"{row['config']} must record train, validation and test counts"
        raise ValueError(message)
    if sorted(row["files"]) != sorted(SPLIT_NAMES):
        message = f"{row['config']} must pin train, validation and test files"
        raise ValueError(message)
    for split, pin in row["files"].items():
        expected_path = (
            f"{dataset['data_root']}/{row['config']}/{dataset['split_files'][split]}"
        )
        if pin["path"] != expected_path:
            message = f"{row['config']} {split} is pinned to an unexpected path"
            raise ValueError(message)
        if not _HEX_40.match(pin["git_blob_sha"]) or pin["size_bytes"] <= 0:
            message = f"{row['config']} {split} has an invalid blob pin"
            raise ValueError(message)
