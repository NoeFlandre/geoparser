"""Load and validate the pinned MultiCoNER II manifest. Nothing is downloaded."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from scripts.panx_benchmark.data import target_languages

DEFAULT_MANIFEST_PATH = Path(__file__).with_name("multiconer_manifest.json")
DATASET_ID = "MultiCoNER/multiconer_v2"
PINNED_REVISION = "4be2d62c912977ee26ed14d2553a4fe17ca3d980"
# Split counts and viewer totals of PINNED_REVISION, one entry per language.
VERIFIED_SPLIT_COUNTS: dict[str, dict[str, int]] = {
    "bn": {"train": 9708, "dev": 507, "test": 19859, "viewer_num_rows": 30074},
    "de": {"train": 9785, "dev": 512, "test": 20145, "viewer_num_rows": 30442},
    "en": {"train": 16778, "dev": 871, "test": 249980, "viewer_num_rows": 267629},
    "es": {"train": 16453, "dev": 854, "test": 246900, "viewer_num_rows": 264207},
    "fa": {"train": 16321, "dev": 855, "test": 219168, "viewer_num_rows": 236344},
    "fr": {"train": 16548, "dev": 857, "test": 249786, "viewer_num_rows": 267191},
    "hi": {"train": 9632, "dev": 514, "test": 18399, "viewer_num_rows": 28545},
    "it": {"train": 16579, "dev": 858, "test": 247881, "viewer_num_rows": 265318},
    "pt": {"train": 16469, "dev": 854, "test": 229490, "viewer_num_rows": 246813},
    "sv": {"train": 16363, "dev": 856, "test": 231190, "viewer_num_rows": 248409},
    "uk": {"train": 16429, "dev": 851, "test": 238296, "viewer_num_rows": 255576},
    "zh": {"train": 9759, "dev": 506, "test": 20265, "viewer_num_rows": 30530},
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SPLITS = ("train", "dev", "test")


def load_manifest(path: Path = DEFAULT_MANIFEST_PATH) -> dict[str, Any]:
    """Read the checked-in pins and refuse any inconsistent value."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_manifest(payload)
    return payload


def intersection_languages(manifest: dict[str, Any]) -> tuple[str, ...]:
    """Return dataset languages in the 85-code inventory, in inventory order."""
    dataset_codes = set(manifest["languages"])
    return tuple(code for code in target_languages() if code in dataset_codes)


def validate_manifest(payload: dict[str, Any]) -> None:
    """Check pins, counts and digest policy without touching the network."""
    _validate_dataset(payload)
    _validate_languages(payload["languages"])


def _validate_dataset(payload: dict[str, Any]) -> None:
    """Check the dataset identity, its commit pin and the license."""
    dataset = payload["dataset"]
    if dataset["id"] != DATASET_ID:
        message = "the manifest names a different dataset"
        raise ValueError(message)
    if dataset["revision"] != PINNED_REVISION:
        message = "dataset revision must be the verified 40-character commit"
        raise ValueError(message)
    if payload["license"]["spdx"] != "CC-BY-4.0":
        message = "the pinned license is not CC-BY-4.0"
        raise ValueError(message)


def _validate_languages(languages: dict[str, Any]) -> None:
    """Check the language inventory, then each language in turn."""
    if "multi" in languages:
        message = "the MULTI configuration is not a language and must be excluded"
        raise ValueError(message)
    outside = set(languages) - set(target_languages())
    if outside:
        message = f"languages outside the 85-code inventory: {sorted(outside)}"
        raise ValueError(message)
    for code, language in languages.items():
        _validate_language(code, language)


def _validate_language(code: str, language: dict[str, Any]) -> None:
    """Check one language's split counts, file paths and digest policy."""
    _validate_split_counts(code, language)
    folder = f"{code.upper()}-{language['name']}"
    for split in _SPLITS:
        _validate_file(code, split, folder, language["files"][split])


def _validate_split_counts(code: str, language: dict[str, Any]) -> None:
    """Check that the three split counts are integers that add up to the total."""
    splits = language["splits"]
    if set(splits) != set(_SPLITS) or any(type(v) is not int for v in splits.values()):
        message = f"{code} must record integer train, dev and test counts"
        raise ValueError(message)
    if sum(splits.values()) != language["viewer_num_rows"]:
        message = f"split counts for {code} do not match the dataset-viewer total"
        raise ValueError(message)
    counts = {**splits, "viewer_num_rows": language["viewer_num_rows"]}
    if VERIFIED_SPLIT_COUNTS.get(code) != counts:
        message = f"split counts for {code} differ from the verified release"
        raise ValueError(message)


def _validate_file(code: str, split: str, folder: str, entry: dict[str, Any]) -> None:
    """Check one split file's path, size and digest claim."""
    expected = f"{folder}/{code}_{split}.conll"
    if entry["path"] != expected:
        message = f"{code} {split} file path must be {expected}"
        raise ValueError(message)
    if type(entry["bytes"]) is not int or entry["bytes"] <= 0:
        message = f"{expected} must record a positive byte size"
        raise ValueError(message)
    _validate_digest(expected, entry)


def _validate_digest(expected: str, entry: dict[str, Any]) -> None:
    """Dispatch on the digest algorithm; an unknown algorithm is refused."""
    algorithm = entry["digest_algorithm"]
    if algorithm == "sha256":
        _validate_lfs_digest(expected, entry)
    elif algorithm == "git-blob-sha1":
        _validate_git_blob_digest(expected, entry)
    else:
        message = f"{expected} uses an unknown digest algorithm: {algorithm}"
        raise ValueError(message)


def _validate_lfs_digest(expected: str, entry: dict[str, Any]) -> None:
    """An LFS file must carry a sha256 digest that equals its LFS digest."""
    if not _SHA256.fullmatch(entry["digest"]) or entry["sha256"] != entry["digest"]:
        message = f"{expected} sha256 must equal its LFS digest"
        raise ValueError(message)


def _validate_git_blob_digest(expected: str, entry: dict[str, Any]) -> None:
    """A plain git blob has a sha1 digest and must not claim a sha256."""
    if not re.fullmatch(r"^[0-9a-f]{40}$", entry["digest"]):
        message = f"{expected} git blob digest must be 40 hex characters"
        raise ValueError(message)
    if entry["sha256"] is not None:
        message = f"{expected} is not LFS, so it must not claim a sha256"
        raise ValueError(message)


def inventory_report(manifest: dict[str, Any]) -> dict[str, Any]:
    """Summarise the pinned inventory for the dry-run command."""
    languages = intersection_languages(manifest)
    return {
        "dataset": {
            "id": manifest["dataset"]["id"],
            "revision": manifest["dataset"]["revision"],
            "license": manifest["license"]["spdx"],
        },
        "languages": list(languages),
        "splits": {code: manifest["languages"][code]["splits"] for code in languages},
        "excluded": sorted(manifest["excluded_configurations"]),
        "clean_noisy_supported": manifest["clean_noisy_breakdown"][
            "supported_by_release"
        ],
        "gap_count": len(manifest["gaps"]),
        "gaps": manifest["gaps"],
    }


def check_local_file(
    path: Path, language: str, split: str, manifest: dict[str, Any]
) -> None:
    """Refuse a local file whose size or digest differs from its pinned split."""
    if language not in manifest["languages"]:
        message = f"{language!r} has no pinned split in this manifest"
        raise ValueError(message)
    entry = manifest["languages"][language]["files"][split]
    data = path.read_bytes()
    if len(data) != entry["bytes"]:
        message = (
            f"{path.name} has {len(data)} bytes; the pinned {language} {split} "
            f"file has {entry['bytes']}"
        )
        raise ValueError(message)
    if _local_digest(data, entry["digest_algorithm"]) != entry["digest"]:
        message = f"{path.name} does not match the pinned digest for {language} {split}"
        raise ValueError(message)


def _local_digest(data: bytes, algorithm: str) -> str:
    """Hash file bytes the way the manifest pinned them."""
    if algorithm == "sha256":
        return hashlib.sha256(data).hexdigest()
    # A git blob id hashes a short header and then the bytes.
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()  # noqa: S324 - git blob ids are sha1
