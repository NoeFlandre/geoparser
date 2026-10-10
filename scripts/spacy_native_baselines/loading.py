"""Load an installed, pinned native pipeline without downloading anything.

A missing or differently versioned package is an error that names the pinned
release. This module never calls ``spacy.cli.download``, so a benchmark run
cannot silently fetch a model.
"""

from __future__ import annotations

import importlib.metadata
import re
from collections.abc import Callable
from typing import Any

import spacy

from scripts.spacy_native_baselines.roster import (
    SPACY_RUNTIME,
    SPACY_RUNTIME_BELOW,
    SPACY_RUNTIME_MIN,
    NativePipeline,
)

KEPT_COMPONENTS = frozenset({"ner", "tok2vec"})
SPACY_PACKAGE = "spacy"
_RELEASE = re.compile(r"(\d+)\.(\d+)(?:\.(\d+))?")


class MissingPipelineError(RuntimeError):
    """The pinned pipeline is absent or at a different version."""


class SpacyRuntimeError(RuntimeError):
    """The installed spaCy runtime is outside the roster's pinned range."""


class LabelSchemeError(RuntimeError):
    """The loaded NER labels differ from the roster's recorded label scheme."""


def installed_version(package: str) -> str | None:
    """Return the installed distribution version, or None when absent."""
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


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


def _release(version: str) -> tuple[int, ...] | None:
    """Return the leading major.minor.patch of a version string, or None."""
    match = _RELEASE.match(version)
    if match is None:
        return None
    return tuple(int(part) for part in match.groups(default="0"))


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
    release = _release(found)
    if release is None or not SPACY_RUNTIME_MIN <= release < SPACY_RUNTIME_BELOW:
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
) -> Any:
    """Return the pinned pipeline with only its NER path kept."""
    check_installed(pipeline, version_lookup=version_lookup)
    check_spacy_runtime(version_lookup=version_lookup)
    nlp = loader(pipeline.package)
    check_label_scheme(nlp, pipeline)
    _keep_only_ner(nlp)
    return nlp
