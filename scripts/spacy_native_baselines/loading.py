"""Load an installed, pinned native pipeline without downloading anything.

A missing or differently versioned package is an error that names the pinned
release. This module never calls ``spacy.cli.download``, so a benchmark run
cannot silently fetch a model.
"""

from __future__ import annotations

import importlib.metadata
import json
import operator
import re
from collections.abc import Callable
from typing import Any

import spacy

from scripts.spacy_native_baselines.roster import (
    SPACY_RUNTIME,
    SPACY_RUNTIME_BELOW,
    SPACY_RUNTIME_MIN,
    NativePipeline,
    configuration_id,
)

KEPT_COMPONENTS = frozenset({"ner", "tok2vec"})
SPACY_PACKAGE = "spacy"
_REQUIREMENT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_VERSION = re.compile(
    r"(?P<release>\d+(?:\.\d+)*)"
    r"(?P<pre>(?:a|b|rc)\d+)?"
    r"(?P<post>\.post\d+)?"
    r"(?P<dev>\.dev\d+)?"
    # A local label such as +vendor.1 is parsed and ignored, as PEP 440 range
    # comparisons ignore it.
    r"(?:\+(?P<local>[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*))?"
)
_SPECIFIER_TERM = re.compile(r"(>=|<=|==|!=|>|<)(\d+(?:\.\d+)*)")
_OPERATORS: dict[str, Callable[[Any, Any], bool]] = {
    ">=": operator.ge,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    "<": operator.lt,
}


class MissingPipelineError(RuntimeError):
    """The pinned pipeline is absent or at a different version."""


class SpacyRuntimeError(RuntimeError):
    """The installed spaCy runtime is outside the roster's pinned range."""


class TokenizerRequirementError(RuntimeError):
    """A recorded tokenizer dependency is missing or outside its specifier."""


class ProvenanceError(RuntimeError):
    """An installed native pipeline did not come from its roster artifact."""


class LabelSchemeError(RuntimeError):
    """The loaded NER labels differ from the roster's recorded label scheme."""


def installed_version(package: str) -> str | None:
    """Return the installed distribution version, or None when absent."""
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def installed_direct_url(package: str) -> dict[str, Any] | None:
    """Return the PEP 610 record of how a distribution was installed, or None."""
    try:
        text = importlib.metadata.distribution(package).read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        return None
    return json.loads(text) if text else None


def check_installed(
    pipeline: NativePipeline,
    *,
    version_lookup: Callable[[str], str | None] = installed_version,
) -> None:
    """Require the exact pinned package version to be installed."""
    found = version_lookup(pipeline.package)
    if found is None:
        message = (
            f"{pipeline.package} is not installed. Install the pinned release "
            f"with `{pipeline.install_command}`. The baseline does not download "
            "pipelines."
        )
        raise MissingPipelineError(message)
    if found != pipeline.version:
        message = (
            f"{pipeline.package} {found} is installed, but the roster pins "
            f"{pipeline.version}. Install `{pipeline.install_command}`."
        )
        raise MissingPipelineError(message)


def _release_and_stage(version: str) -> tuple[tuple[int, ...], bool] | None:
    """Return the release numbers and whether the version sorts before its final.

    A pre-release, or a dev release that is not also a post-release, sorts
    before the final release it names. A local label is ignored. Returns None
    for any string the PEP 440 pattern does not recognize.
    """
    match = _VERSION.fullmatch(version)
    if match is None:
        return None
    numbers = _release_numbers(match.group("release"))
    before_final = match.group("pre") is not None or (
        match.group("dev") is not None and match.group("post") is None
    )
    return numbers, before_final


def _pad(parts: tuple[int, ...], width: int) -> tuple[int, ...]:
    return parts + (0,) * (width - len(parts))


def _within_runtime_range(version: str) -> bool:
    """Return True when a PEP 440 version lies in the pinned runtime range.

    The range is inclusive below and exclusive above, as the roster's
    ``SPACY_RUNTIME`` specifier reads. A prerelease or dev release of the lower
    bound's own release (3.8.0rc1, 3.8.0.dev1) sorts before that release, so it
    is refused. A prerelease of the exclusive upper bound (3.9.0rc1) is refused
    too, as PEP 440 requires. Local labels (3.8.16+vendor.1) do not change the
    release. Any version the pattern does not recognize is refused.
    """
    parsed = _release_and_stage(version)
    if parsed is None:
        return False
    numbers, before_final = parsed
    width = max(len(numbers), len(SPACY_RUNTIME_MIN), len(SPACY_RUNTIME_BELOW))
    current = _pad(numbers, width)
    low = _pad(SPACY_RUNTIME_MIN, width)
    high = _pad(SPACY_RUNTIME_BELOW, width)
    if current == low and before_final:
        return False
    return low <= current < high


def check_spacy_runtime(
    *,
    version_lookup: Callable[[str], str | None] = installed_version,
) -> None:
    """Require the installed spaCy runtime to lie in the roster's pinned range.

    A model package names only its own release, so a spaCy runtime outside
    the range the roster was verified against must be refused separately.
    """
    found = version_lookup(SPACY_PACKAGE)
    if found is None:
        message = (
            f"spaCy is not installed. The roster pins the runtime range "
            f"{SPACY_RUNTIME}."
        )
        raise SpacyRuntimeError(message)
    if not _within_runtime_range(found):
        message = (
            f"spaCy {found} is installed, but the roster pins the runtime range "
            f"{SPACY_RUNTIME}."
        )
        raise SpacyRuntimeError(message)


def _keep_only_ner(nlp: Any) -> None:
    for name in [name for name in nlp.pipe_names if name not in KEPT_COMPONENTS]:
        nlp.remove_pipe(name)


def check_label_scheme(nlp: Any, pipeline: NativePipeline) -> None:
    """Fail when the loaded NER labels are not the roster's label scheme."""
    if "ner" not in nlp.pipe_names:
        message = f"{pipeline.package} has no ner component to harmonize."
        raise LabelSchemeError(message)
    loaded = set(nlp.get_pipe("ner").labels)
    expected = set(pipeline.ner_labels)
    if loaded != expected:
        missing = sorted(expected - loaded)
        extra = sorted(loaded - expected)
        message = (
            f"{pipeline.package} NER labels differ from the roster: "
            f"missing {missing}, unexpected {extra}."
        )
        raise LabelSchemeError(message)


def load_pipeline(
    pipeline: NativePipeline,
    *,
    loader: Callable[[str], Any] = spacy.load,
    version_lookup: Callable[[str], str | None] = installed_version,
    direct_url_lookup: Callable[[str], dict[str, Any] | None] = installed_direct_url,
) -> Any:
    """Return the pinned pipeline with only its NER path kept."""
    check_installed(pipeline, version_lookup=version_lookup)
    check_spacy_runtime(version_lookup=version_lookup)
    check_tokenizers(pipeline, version_lookup=version_lookup)
    check_provenance(pipeline, direct_url_lookup=direct_url_lookup)
    nlp = loader(pipeline.package)
    check_label_scheme(nlp, pipeline)
    _keep_only_ner(nlp)
    return nlp


def requirement_name(requirement: str) -> str:
    """Return the distribution name at the start of a requirement specifier."""
    match = _REQUIREMENT_NAME.match(requirement)
    if match is None:
        message = f"Cannot read a distribution name from {requirement!r}."
        raise ValueError(message)
    return match.group(0)


def _release_numbers(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split("."))


def _specifier_terms(requirement: str) -> list[tuple[str, tuple[int, ...]]]:
    """Return the (operator, release) terms of a requirement, refusing others."""
    constraints = requirement[len(requirement_name(requirement)) :]
    terms: list[tuple[str, tuple[int, ...]]] = []
    for part in constraints.split(","):
        term = part.strip()
        if not term:
            continue
        match = _SPECIFIER_TERM.fullmatch(term)
        if match is None:
            message = f"Unsupported specifier {term!r} in {requirement!r}."
            raise ValueError(message)
        terms.append((match.group(1), _release_numbers(match.group(2))))
    return terms


def requirement_satisfied(requirement: str, installed: str) -> bool:
    """Return True when the installed release satisfies every term of a requirement.

    Terms compare plain release numbers with zero padding. An installed version
    that is not a plain release, such as 1.0rc1, never satisfies a requirement.
    """
    terms = _specifier_terms(requirement)
    parsed = _release_and_stage(installed)
    if parsed is None or parsed[1]:
        return False
    version = parsed[0]
    for operation, bound in terms:
        width = max(len(version), len(bound))
        if not _OPERATORS[operation](_pad(version, width), _pad(bound, width)):
            return False
    return True


def check_tokenizers(
    pipeline: NativePipeline,
    *,
    version_lookup: Callable[[str], str | None] = installed_version,
) -> None:
    """Require each recorded tokenizer dependency to be installed within its specifier.

    The roster's tokenizer specifiers are open-ended, so a release outside them
    can tokenize differently or fail to load. This check runs before the
    pipeline is loaded.
    """
    for requirement in pipeline.extra_requirements:
        name = requirement_name(requirement)
        found = version_lookup(name)
        if found is None:
            message = f"{name} is not installed. Install `{pipeline.install_command}`."
            raise TokenizerRequirementError(message)
        if not requirement_satisfied(requirement, found):
            message = (
                f"{name} {found} is installed, but the roster requires "
                f"{requirement!r}. Install `{pipeline.install_command}`."
            )
            raise TokenizerRequirementError(message)


def check_provenance(
    pipeline: NativePipeline,
    *,
    direct_url_lookup: Callable[[str], dict[str, Any] | None] = installed_direct_url,
) -> None:
    """Require the installed pipeline to be the roster's hashed wheel.

    A version match cannot tell a wheel from an sdist of the same release, or a
    replaced release asset. The installed distribution's recorded URL and archive
    hash must match the roster's wheel URL and SHA-256 digest. pip writes the URL
    without its fragment and the hash as ``sha256=<digest>``.
    """
    record = direct_url_lookup(pipeline.package) or {}
    installed_url = str(record.get("url", "")).split("#", 1)[0]
    archive = record.get("archive_info") or {}
    if installed_url != pipeline.wheel_url or archive.get("hash") != (
        f"sha256={pipeline.sha256}"
    ):
        message = (
            f"{pipeline.package} was not installed from its roster wheel "
            f"{pipeline.wheel_url} with digest {pipeline.sha256}. Install "
            f"`{pipeline.install_command}`."
        )
        raise ProvenanceError(message)


def resolved_versions(
    pipeline: NativePipeline,
    *,
    version_lookup: Callable[[str], str | None] = installed_version,
) -> dict[str, str | None]:
    """Return the installed version of spaCy, the pipeline and each tokenizer."""
    names = [SPACY_PACKAGE, pipeline.package]
    names.extend(
        requirement_name(requirement) for requirement in pipeline.extra_requirements
    )
    return {name: version_lookup(name) for name in names}


def run_identity(
    pipeline: NativePipeline,
    *,
    version_lookup: Callable[[str], str | None] = installed_version,
) -> str:
    """Return the configuration identity extended with the installed releases.

    The roster's tokenizer specifiers are open-ended, so clean installs at
    different times can resolve to different releases, and spaCy patch releases
    can change predictions. An identity built from the specifiers alone cannot
    tell those environments apart, so the resolved versions are part of it.
    """
    resolved = resolved_versions(pipeline, version_lookup=version_lookup)
    return configuration_id(pipeline, extra={"resolved": resolved})
