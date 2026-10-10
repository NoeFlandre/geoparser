"""Language coverage of the pinned DaMuEL release against the canonical targets.

The release record is a checked-in snapshot of public LINDAT metadata. Reading
it never opens a dataset archive, so the coverage report is reproducible
offline and cannot be mistaken for evidence about annotation quality.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

from scripts.panx_benchmark.data import target_languages

RELEASE_PATH = Path(__file__).with_name("damuel_1_0_release.json")
EXPECTED_TEXT_LANGUAGES = 53


def _calendar_day(value: str) -> str:
    """Reject strings shaped like dates that do not exist in the calendar."""
    try:
        date.fromisoformat(value)
    except ValueError:
        message = "the date must be a real calendar day in YYYY-MM-DD form"
        raise ValueError(message) from None
    return value


Text = Annotated[str, Field(min_length=1, pattern=r"\S")]
Md5 = Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]
LanguageCode = Annotated[str, Field(pattern=r"^[a-z]{2,3}$")]
Day = Annotated[
    str, Field(pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$"), AfterValidator(_calendar_day)
]


class Contract(BaseModel):
    """Reject unknown fields and coercions in the checked-in release record."""

    model_config = ConfigDict(extra="forbid", strict=True)


class Record(Contract):
    """Where the release is published and which paper describes it."""

    publisher: Text
    handle: Text
    url: Text
    paper: Text
    paper_authors: Annotated[list[Text], Field(min_length=1)]


class Dumps(Contract):
    """Upstream snapshot dates the release was built from."""

    wikipedia: Day
    wikidata: Day


class Licence(Contract):
    """The licence name and the evidence this repository actually checked."""

    name: Text
    verified_sources: Annotated[list[Text], Field(min_length=1)]
    not_verified: list[Text]


class UnofficialMirror(Contract):
    """A third-party copy, recorded so it is never mistaken for the source."""

    url: Text
    note: Text


class ReleaseFile(Contract):
    """One archive from the release, as listed on the public record."""

    file: Annotated[str, Field(pattern=r"^damuel_1\.0_(?:[a-z]{2,3}|wikidata)\.tar$")]
    kind: Literal["text", "knowledge_base"]
    language: LanguageCode | None
    size_display: Text
    size_bytes: Annotated[int, Field(gt=0)] | None = None
    md5: Md5
    listing_description: Text

    @model_validator(mode="after")
    def file_identity(self) -> ReleaseFile:
        """Tie each archive name to exactly one kind and language."""
        if self.kind == "knowledge_base":
            _require_knowledge_base_identity(self)
        else:
            _require_text_archive_identity(self)
        return self


def _require_knowledge_base_identity(entry: ReleaseFile) -> None:
    """Reject a knowledge base that has a language or another archive name."""
    if entry.language is not None or entry.file != "damuel_1.0_wikidata.tar":
        message = "the knowledge base must be damuel_1.0_wikidata.tar"
        raise ValueError(message)


def _require_text_archive_identity(entry: ReleaseFile) -> None:
    """Reject a text archive whose name does not match its language code."""
    if entry.language is None or entry.file != f"damuel_1.0_{entry.language}.tar":
        message = "a text archive name must match its language code"
        raise ValueError(message)


def _require_unique_archives(files: list[ReleaseFile]) -> None:
    """Reject a release that lists the same archive name more than once."""
    names = [entry.file for entry in files]
    if len(set(names)) != len(names):
        message = "the release lists an archive more than once"
        raise ValueError(message)


def _require_one_knowledge_base(files: list[ReleaseFile]) -> None:
    """Reject a release that lists zero or several knowledge bases."""
    knowledge_bases = [entry for entry in files if entry.kind == "knowledge_base"]
    if len(knowledge_bases) != 1:
        message = "the release must list exactly one knowledge base"
        raise ValueError(message)


def _require_text_archive_count(files: list[ReleaseFile]) -> None:
    """Reject a release that does not list one text archive per expected language."""
    languages = [entry.language for entry in files if entry.kind == "text"]
    if len(languages) != EXPECTED_TEXT_LANGUAGES:
        message = "the release must list 53 text archives"
        raise ValueError(message)


class Release(Contract):
    """The complete checked-in DaMuEL release record."""

    schema_version: Literal["1.0"]
    dataset: Literal["DaMuEL"]
    release: Literal["1.0"]
    record: Record
    retrieved_on: Day
    retrieval_method: Text
    dumps: Dumps
    licence: Licence
    bundled_metadata: dict[Text, Text]
    unofficial_mirror: UnofficialMirror
    files: Annotated[list[ReleaseFile], Field(min_length=2)]

    @model_validator(mode="after")
    def inventory_identity(self) -> Release:
        """Require one knowledge base and one text archive per language."""
        _require_unique_archives(self.files)
        _require_one_knowledge_base(self.files)
        _require_text_archive_count(self.files)
        return self

    def text_languages(self) -> tuple[str, ...]:
        """Return the sorted language codes that have a text archive."""
        codes = [
            entry.language
            for entry in self.files
            if entry.kind == "text" and entry.language is not None
        ]
        return tuple(sorted(codes))


@dataclass(frozen=True)
class Coverage:
    """How the canonical target list meets the DaMuEL text inventory."""

    canonical: tuple[str, ...]
    covered: tuple[str, ...]
    missing: tuple[str, ...]
    release_only: tuple[str, ...]


def load_release(path: Path = RELEASE_PATH) -> Release:
    """Validate the checked-in release record without touching the network."""
    return Release.model_validate_json(path.read_text(encoding="utf-8"))


def coverage(release: Release, targets: tuple[str, ...] | None = None) -> Coverage:
    """Compare canonical targets with release languages without normalizing codes.

    Codes are matched exactly. In particular ``no`` (Norwegian) is not treated as
    ``nn`` (Nynorsk), and script or regional variants are not inferred.
    """
    canonical = target_languages() if targets is None else targets
    available = set(release.text_languages())
    covered, missing = _split_by_membership(canonical, available)
    return Coverage(
        canonical=tuple(canonical),
        covered=covered,
        missing=missing,
        release_only=tuple(sorted(available.difference(canonical))),
    )


def _split_by_membership(
    codes: tuple[str, ...], available: set[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split canonical codes into those the release has and those it lacks."""
    covered = tuple(code for code in codes if code in available)
    missing = tuple(code for code in codes if code not in available)
    return covered, missing


class CoverageSummary(TypedDict):
    """The JSON-ready figures printed by the command line interface."""

    dataset: str
    release: str
    retrieved_on: str
    licence: str
    text_archive_count: int
    canonical_target_count: int
    covered_count: int
    missing_count: int
    release_only_count: int
    covered: list[str]
    missing: list[str]
    release_only: list[str]


def coverage_summary(release: Release) -> CoverageSummary:
    """Return a JSON-ready report with counts, so no figure is typed by hand."""
    report = coverage(release)
    return {
        "dataset": release.dataset,
        "release": release.release,
        "retrieved_on": release.retrieved_on,
        "licence": release.licence.name,
        "text_archive_count": len(release.text_languages()),
        "canonical_target_count": len(report.canonical),
        "covered_count": len(report.covered),
        "missing_count": len(report.missing),
        "release_only_count": len(report.release_only),
        "covered": list(report.covered),
        "missing": list(report.missing),
        "release_only": list(report.release_only),
    }
