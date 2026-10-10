"""Pinned UNER source metadata and download-free split validation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from scripts.panx_benchmark.data import target_languages
from scripts.uner_benchmark.data import Corpus, parse_iob2

Split = Literal["train", "dev", "test"]
GitHash = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
Name = Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]+$")]
Nonempty = Annotated[str, Field(min_length=1)]
MANIFEST_PATH = Path(__file__).with_name("sources.json")


class File(BaseModel):
    """A repository-relative file and its pinned Git blob object ID."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    path: str
    git_blob_sha1: GitHash

    @field_validator("path")
    @classmethod
    def safe_path(cls, value: str) -> str:
        """Reject absolute, platform-specific and noncanonical local paths."""
        parts = PurePosixPath(value).parts
        _require_relative_path(value)
        if not parts:
            message = "Source path must remain within its repository"
            raise ValueError(message)
        if any(character in value for character in (":", "\\")):
            message = "Source path must be canonical POSIX syntax"
            raise ValueError(message)
        return _canonical_path(value)


def _require_relative_path(value: str) -> None:
    """Reject absolute paths and explicit parent traversal."""
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        message = "Source path must remain within its repository"
        raise ValueError(message)


def _canonical_path(value: str) -> str:
    """Reject dot components and repeated separators rather than normalizing."""
    if PurePosixPath(value).as_posix() != value:
        message = "Source path must be canonical POSIX syntax"
        raise ValueError(message)
    return value


class Dataset(BaseModel):
    """One source configuration; varieties remain distinct after mapping."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    configuration: Name
    repository: Name
    revision: GitHash
    source_language: Nonempty
    target_language: str | None
    domains: Annotated[tuple[Nonempty, ...], Field(min_length=1)]
    annotation_provenance: Nonempty
    domain_source: Nonempty
    mapping_note: Nonempty
    license_note: Nonempty
    license: Nonempty
    license_sources: tuple[File, ...]
    splits: Annotated[dict[Split, File], Field(min_length=1)]

    @model_validator(mode="after")
    def declared_license_has_source(self) -> Dataset:
        """Require a pinned notice for every asserted dataset license."""
        if self.license != "unknown" and not self.license_sources:
            message = "A declared license requires a pinned license source"
            raise ValueError(message)
        return self

    @field_validator("target_language")
    @classmethod
    def canonical_target(cls, value: str | None) -> str | None:
        """Accept only a canonical85 target or explicit out-of-inventory status."""
        if value is not None and value not in target_languages():
            message = "Target language is outside the canonical85 inventory"
            raise ValueError(message)
        return value

    @field_validator("splits")
    @classmethod
    def separate_splits(cls, value: dict[Split, File]) -> dict[Split, File]:
        """Reject aliases that expose the same source as a different split."""
        for field_name in ("path", "git_blob_sha1"):
            identifiers = {getattr(source, field_name) for source in value.values()}
            if len(identifiers) != len(value):
                message = f"Repeated source {field_name} across splits"
                raise ValueError(message)
        return value

    def status(self, split: Split) -> str:
        """Explain whether this source can supply the requested target split."""
        if self.target_language is None:
            return "outside_target_inventory"
        if split not in self.splits:
            return "split_unavailable"
        if self.license == "unknown":
            return "license_unverified"
        return "available"

    def url(self, source: File) -> str:
        """Return the immutable official source URL."""
        return (
            "https://raw.githubusercontent.com/UniversalNER/"
            f"{self.repository}/{self.revision}/{source.path}"
        )


@dataclass(frozen=True)
class LoadedCorpus:
    """Verified source bytes, their SHA-256 digest and the complete split."""

    path: Path
    sha256: str
    corpus: Corpus


def _blob_hash(payload: bytes) -> str:
    """Reproduce Git's content-addressed blob ID, without using it for security."""
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload, usedforsecurity=False).hexdigest()


def load_local(spec: Dataset, split: Split, cache_dir: Path) -> LoadedCorpus:
    """Validate only the requested local split, with no fallback or download."""
    if spec.status(split) != "available":
        message = f"Unavailable source split: {spec.configuration}/{split}: {spec.status(split)}"
        raise ValueError(message)
    source = spec.splits[split]
    path = cache_dir / spec.repository / source.path
    payload = path.read_bytes()
    if _blob_hash(payload) != source.git_blob_sha1:
        message = f"Source checksum mismatch: {spec.configuration}/{split}"
        raise ValueError(message)
    corpus = parse_iob2(
        payload.decode("utf-8"),
        language=str(spec.target_language),
        configuration=spec.configuration,
        split=split,
    )
    return LoadedCorpus(path, hashlib.sha256(payload).hexdigest(), corpus)


def _read_manifest_source(path: Path) -> dict[str, Any]:
    """Read and validate the local manifest header."""
    source = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(source, dict):
        message = "UNER source inventory must be a JSON object"
        raise TypeError(message)
    if source.get("audit_status") != "complete":
        message = (
            "UNER source inventory is incomplete; finish the release audit before use"
        )
        raise ValueError(message)
    return source


def _validate_declared_count(source: dict[str, Any], field: str, actual: int) -> None:
    """Check a declared inventory count when the manifest provides one."""
    if field in source and actual != source[field]:
        message = (
            f"UNER source inventory {field} does not match its datasets: "
            f"declared {source[field]}, found {actual}"
        )
        raise ValueError(message)


def _validate_manifest_counts(
    source: dict[str, Any], specs: tuple[Dataset, ...]
) -> None:
    """Check the declared configuration and source-file inventory totals."""
    _validate_declared_count(
        source, "repository_count", len({spec.repository for spec in specs})
    )
    _validate_declared_count(
        source, "split_file_count", sum(len(spec.splits) for spec in specs)
    )
    if len(specs) != source["expected_configuration_count"]:
        message = "UNER source inventory count does not match its declared scope"
        raise ValueError(message)


def read_manifest(path: Path = MANIFEST_PATH) -> tuple[Dataset, ...]:
    """Read the checked-in source inventory without consulting mutable remotes."""
    source = _read_manifest_source(path)
    specs = tuple(Dataset.model_validate(dataset) for dataset in source["datasets"])
    validate_configurations(specs)
    _validate_manifest_counts(source, specs)
    return specs


def _configuration(spec: Dataset, split: Split) -> dict[str, Any]:
    """Describe each slice, including ones that cannot contribute examples."""
    return {
        "configuration": spec.configuration,
        "source_language": spec.source_language,
        "target_language": spec.target_language,
        "domains": list(spec.domains),
        "domain_source": spec.domain_source,
        "mapping_note": spec.mapping_note,
        "license_note": spec.license_note,
        "annotation_provenance": spec.annotation_provenance,
        "license": spec.license,
        "status": spec.status(split),
        "available_splits": sorted(spec.splits),
        "revision": spec.revision,
        "sources": {
            name: {"url": spec.url(source), "git_blob_sha1": source.git_blob_sha1}
            for name, source in spec.splits.items()
        },
    }


def inventory_summary(
    specs: list[Dataset] | tuple[Dataset, ...], split: Split
) -> dict[str, Any]:
    """Keep every configuration and all canonical85 missing-language outcomes."""
    validate_configurations(specs)
    selected = [spec for spec in specs if spec.status(split) == "available"]
    covered = _covered_languages(selected)
    present = _present_languages(specs, split)
    return {
        "task": "recognition",
        "split": split,
        "training_overlap": "unknown",
        "target_language_count": len(target_languages()),
        "covered_target_languages": sorted(covered),
        "present_target_languages": sorted(present),
        "missing_target_languages": sorted(set(target_languages()) - present),
        "unavailable_target_languages": sorted(present - covered),
        "domain_configuration_counts": _domain_counts(selected),
        "selected_configurations": [spec.configuration for spec in selected],
        "configurations": [_configuration(spec, split) for spec in specs],
    }


def validate_configurations(specs: list[Dataset] | tuple[Dataset, ...]) -> None:
    """Reject duplicate source configurations before summary or selection."""
    names = [spec.configuration for spec in specs]
    if len(set(names)) != len(names):
        message = "Inventory has duplicate configuration names"
        raise ValueError(message)


def _covered_languages(specs: list[Dataset]) -> set[str]:
    """List the distinct canonical languages of available source varieties."""
    return {spec.target_language for spec in specs if spec.target_language is not None}


def _present_languages(
    specs: list[Dataset] | tuple[Dataset, ...], split: Split
) -> set[str]:
    """Count source presence separately from its license eligibility."""
    return {
        spec.target_language
        for spec in specs
        if spec.target_language is not None and split in spec.splits
    }


def _domain_counts(specs: list[Dataset]) -> dict[str, int]:
    """Count selected configurations tagged with each source domain."""
    counts = Counter(domain for spec in specs for domain in set(spec.domains))
    return dict(sorted(counts.items()))
