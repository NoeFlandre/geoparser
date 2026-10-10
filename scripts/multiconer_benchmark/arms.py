"""Language support of the three #98 model arms on the MultiCoNER II languages.

The arms and their pins are reused from ``scripts.panx_benchmark``. Nothing here
loads a model. The support labels are the protocol vocabulary. An English-only
arm on another language is an explicit ``out_of_language_control`` outcome, never a zero.
"""

from __future__ import annotations

from typing import Literal

from scripts.panx_benchmark.constants import MODELS

Support = Literal["documented", "transfer", "unspecified", "out_of_language_control"]
_SPECS = {spec.key: spec for spec in MODELS}


def arm_keys() -> tuple[str, ...]:
    """Return the three pinned arm keys in their #98 order."""
    return tuple(spec.key for spec in MODELS)


def language_support(arm: str, language: str) -> Support:
    """Classify one arm on one language without guessing beyond the pins."""
    spec = _SPECS[arm]
    if spec.documented_languages is None:
        return "unspecified"
    if language in spec.documented_languages:
        return "documented"
    if spec.documented_languages == ("en",):
        return "out_of_language_control"
    return "transfer"
