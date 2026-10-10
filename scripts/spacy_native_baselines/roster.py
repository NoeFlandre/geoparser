"""Versioned roster of native-language spaCy pipelines for the 85 target codes.

Every target code is either routed to one pinned native pipeline or recorded
as unsupported. Unsupported codes never fall back to another pipeline.
"""

from __future__ import annotations

import hashlib
import json
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from scripts.panx_benchmark.data import target_languages

ROSTER_PATH = Path(__file__).with_name("roster.json")
ROSTER_FORMAT = "spacy-native-baseline-roster-v1"
MODEL_VERSION = "3.8.0"
SPACY_RUNTIME = ">=3.8.0,<3.9.0"
ENGLISH_CONTROL_LANGUAGE = "en"
LOCATION_CLASS = "LOC"
Status = Literal["native", "unsupported"]


class UnknownLanguageError(ValueError):
    """A code outside the 85-language target list was requested."""


class RosterError(ValueError):
    """The checked-in roster violates one of its invariants."""


@dataclass(frozen=True, slots=True)
class NativePipeline:
    """One pinned spaCy pipeline and the explicit label harmonization for it."""

    language: str
    spacy_language: str
    package: str
    version: str
    wheel: str
    wheel_url: str
    wheel_bytes: int
    license: str
    ner_labels: tuple[str, ...]
    place_label_map: dict[str, str]
    label_meaning_basis: str
    extra_requirements: tuple[str, ...] = ()
    other_3_8_packages_not_selected: tuple[str, ...] = ()
    sha256: str | None = None
    role: str | None = None

    @property
    def release_tag(self) -> str:
        """The GitHub release tag that publishes this pinned wheel."""
        return f"{self.package}-{self.version}"

    @property
    def install_command(self) -> str:
        """Install the exact pinned wheel plus its recorded tokenizer requirements.

        The model is named by its release URL, so no index search selects it.
        Requirements are shell-quoted because specifiers such as ``>=`` are
        shell operators when left bare.
        """
        return shlex.join(["pip", "install", self.wheel_url, *self.extra_requirements])

    def harmonized_label(self, label: str) -> str | None:
        """Return the WikiANN class for a native label, or None when dropped."""
        return self.place_label_map.get(label)


@dataclass(frozen=True, slots=True)
class Route:
    """The outcome of routing one target code."""

    language: str
    pipeline: NativePipeline | None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class Roster:
    """The checked-in roster, validated on load."""

    languages: dict[str, Status]
    pipelines: dict[str, NativePipeline]
    unsupported: dict[str, str]
    english_control: NativePipeline
    verification: dict[str, Any] = field(default_factory=dict)

    def route(self, language: str) -> Route:
        """Route a target code to its pipeline, or to an explicit unsupported reason."""
        if language not in self.languages:
            message = f"{language!r} is not one of the 85 target language codes."
            raise UnknownLanguageError(message)
        if language in self.pipelines:
            return Route(language, self.pipelines[language])
        return Route(language, None, self.unsupported[language])


def _pipeline(language: str, entry: dict[str, Any]) -> NativePipeline:
    return NativePipeline(
        language=language,
        spacy_language=entry["spacy_language"],
        package=entry["package"],
        version=entry["version"],
        wheel=entry["wheel"],
        wheel_url=entry["wheel_url"],
        wheel_bytes=entry["wheel_bytes"],
        license=entry["license"],
        ner_labels=tuple(entry["ner_labels"]),
        place_label_map=dict(entry["place_label_map"]),
        label_meaning_basis=entry["label_meaning_basis"],
        extra_requirements=tuple(entry.get("extra_requirements", ())),
        other_3_8_packages_not_selected=tuple(
            entry.get("other_3_8_packages_not_selected", ())
        ),
        sha256=entry.get("sha256"),
        role=entry.get("role"),
    )


def _check_release(language: str, pipeline: NativePipeline) -> None:
    expected_wheel = f"{pipeline.package}-{MODEL_VERSION}-py3-none-any.whl"
    if pipeline.version != MODEL_VERSION or pipeline.wheel != expected_wheel:
        message = f"{language}: wheel is not the pinned {MODEL_VERSION} release"
        raise RosterError(message)
    if pipeline.release_tag not in pipeline.wheel_url:
        message = f"{language}: wheel URL does not name its release tag"
        raise RosterError(message)
    if not pipeline.package.startswith(pipeline.spacy_language + "_"):
        message = f"{language}: package does not belong to {pipeline.spacy_language}"
        raise RosterError(message)


def _check_labels(language: str, pipeline: NativePipeline) -> None:
    if not set(pipeline.place_label_map) <= set(pipeline.ner_labels):
        message = f"{language}: a mapped label is not in the pipeline's NER labels"
        raise RosterError(message)
    if set(pipeline.place_label_map.values()) - {LOCATION_CLASS}:
        message = f"{language}: place labels must map only to {LOCATION_CLASS}"
        raise RosterError(message)


def _check_pipeline(language: str, pipeline: NativePipeline) -> None:
    _check_release(language, pipeline)
    _check_labels(language, pipeline)


def _parse_native(language: str, entry: dict[str, Any]) -> NativePipeline:
    pipeline = _pipeline(language, entry)
    _check_pipeline(language, pipeline)
    return pipeline


def _parse_unsupported(language: str, entry: dict[str, Any]) -> str:
    reason = entry.get("reason")
    if not isinstance(reason, str) or not reason:
        message = f"{language}: unsupported entries need a reason"
        raise RosterError(message)
    return reason


def _check_header(document: dict[str, Any]) -> dict[str, Any]:
    if document.get("format") != ROSTER_FORMAT:
        message = f"Roster format must be {ROSTER_FORMAT!r}"
        raise RosterError(message)
    if document.get("model_version") != MODEL_VERSION:
        message = f"Roster must pin spaCy model version {MODEL_VERSION}"
        raise RosterError(message)
    entries = document.get("languages")
    if not isinstance(entries, dict):
        message = "Roster must contain a languages object"
        raise RosterError(message)
    if tuple(entries) != target_languages():
        message = "Roster languages must be the 85 target codes, in order, once each"
        raise RosterError(message)
    return entries


def _english_control(
    document: dict[str, Any],
    pipelines: dict[str, NativePipeline],
) -> NativePipeline:
    control_package = document.get("english_control", {}).get("package")
    english = pipelines.get(ENGLISH_CONTROL_LANGUAGE)
    if english is None or english.package != control_package:
        message = "The English control must be the native English pipeline"
        raise RosterError(message)
    return english


def parse_roster(document: dict[str, Any]) -> Roster:
    """Validate a decoded roster document against the 85 target codes."""
    entries = _check_header(document)
    statuses: dict[str, Status] = {}
    pipelines: dict[str, NativePipeline] = {}
    unsupported: dict[str, str] = {}
    for language, entry in entries.items():
        status = entry.get("status")
        if status == "native":
            pipelines[language] = _parse_native(language, entry)
        elif status == "unsupported":
            unsupported[language] = _parse_unsupported(language, entry)
        else:
            message = f"{language}: unknown status {status!r}"
            raise RosterError(message)
        statuses[language] = status
    return Roster(
        languages=statuses,
        pipelines=pipelines,
        unsupported=unsupported,
        english_control=_english_control(document, pipelines),
        verification=dict(document.get("verification", {})),
    )


def load_roster(path: Path = ROSTER_PATH) -> Roster:
    """Load and validate the checked-in roster JSON."""
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        message = "Roster JSON must be an object"
        raise RosterError(message)
    return parse_roster(document)


def configuration_id(
    pipeline: NativePipeline,
    *,
    extra: dict[str, Any] | None = None,
) -> str:
    """Return a deterministic 16-hex identity for one pinned configuration.

    The identity covers the language, the exact pinned package and wheel, the
    wheel URL and SHA-256 digest, the tokenizer requirements, the NER label
    scheme and the explicit label harmonization. Key order and dictionary
    insertion order do not change it.
    """
    payload = {
        "language": pipeline.language,
        "package": pipeline.package,
        "version": pipeline.version,
        "wheel": pipeline.wheel,
        "wheel_url": pipeline.wheel_url,
        "sha256": pipeline.sha256,
        "extra_requirements": list(pipeline.extra_requirements),
        "release_tag": pipeline.release_tag,
        "spacy_runtime": SPACY_RUNTIME,
        "ner_labels": sorted(pipeline.ner_labels),
        "place_label_map": sorted(pipeline.place_label_map.items()),
        "extra": extra or {},
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
