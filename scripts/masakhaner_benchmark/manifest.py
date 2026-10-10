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
    _check_commit_id(
        dataset["huggingface_revision"],
        "The Hugging Face revision must be a 40-character commit id",
    )
    _check_commit_id(
        dataset["github_commit"],
        "The GitHub commit must be a 40-character commit id",
    )
    if manifest["label_names"] != list(LABEL_NAMES):
        message = "The manifest label names differ from the parser's label set"
        raise ValueError(message)
    languages = manifest["languages"]
    if len(languages) != 20:
        message = "The MasakhaNER 2.0 manifest must list 20 configurations"
        raise ValueError(message)
    _check_identifiers(languages)
    for row in languages:
        _validate_language(row, dataset)


def _check_commit_id(value: str, message: str) -> None:
    """Reject a revision or commit that is not a full 40-character git id."""
    if not _HEX_40.match(value):
        raise ValueError(message)


def _check_identifiers(languages: list[dict[str, Any]]) -> None:
    """Reject duplicate configurations and duplicate ISO 639-1 codes."""
    _check_unique(languages, "config")
    _check_unique([row for row in languages if row["iso639_1"]], "iso639_1")


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
        _validate_pin(row["config"], split, pin, dataset)


def _validate_pin(
    config: str, split: str, pin: dict[str, Any], dataset: dict[str, Any]
) -> None:
    """Check that one split file is pinned to its expected path and a git id."""
    expected_path = f"{dataset['data_root']}/{config}/{dataset['split_files'][split]}"
    if pin["path"] != expected_path:
        message = f"{config} {split} is pinned to an unexpected path"
        raise ValueError(message)
    if not _HEX_40.match(pin["git_blob_sha"]) or pin["size_bytes"] <= 0:
        message = f"{config} {split} has an invalid blob pin"
        raise ValueError(message)
