"""Place-versus-non-place policy for MultiCoNER II fine-grained gold labels.

The policy is recognition only. A ``LOC`` label keeps a gold span in the place
denominator. ``ignore`` removes a non-place gold span from that denominator and
leaves the original fine-grained label in the parsed gold. A predicted span that
exactly matches a non-place gold span is still a false positive, because a place
recognizer that calls an organisation a place is wrong.
"""

from __future__ import annotations

from typing import Literal

# Coarse LOC group of the published tagset: Facility, OtherLOC, HumanSettlement, Station.
PLACE_LABELS: frozenset[str] = frozenset(
    {"Facility", "OtherLOC", "HumanSettlement", "Station"}
)

# The 33 fine-grained types in the README tagset at the pinned revision.
DOCUMENTED_LABELS: tuple[str, ...] = (
    "Facility",
    "OtherLOC",
    "HumanSettlement",
    "Station",
    "VisualWork",
    "MusicalWork",
    "WrittenWork",
    "ArtWork",
    "Software",
    "MusicalGRP",
    "PublicCORP",
    "PrivateCORP",
    "AerospaceManufacturer",
    "SportsGRP",
    "CarManufacturer",
    "ORG",
    "Scientist",
    "Artist",
    "Athlete",
    "Politician",
    "Cleric",
    "SportsManager",
    "OtherPER",
    "Clothing",
    "Vehicle",
    "Food",
    "Drink",
    "OtherPROD",
    "Medication/Vaccine",
    "MedicalProcedure",
    "AnatomicalStructure",
    "Symptom",
    "Disease",
)

# Names that appear only in the loader script at the pinned revision, or that
# differ in case from the README. None is a place. They must still be verified
# against the data bytes before a run, because the two sources disagree.
LOADER_ONLY_LABELS: frozenset[str] = frozenset(
    {"OtherCW", "OtherCorp", "TechCORP", "PublicCorp", "PrivateCorp"}
)

KNOWN_LABELS: frozenset[str] = frozenset(DOCUMENTED_LABELS) | LOADER_ONLY_LABELS

Mapped = Literal["LOC", "ignore"]


class UnknownLabelError(ValueError):
    """A gold entity type outside the known inventory, which is never guessed."""


def label_for(entity_type: str) -> Mapped:
    """Map one fine-grained entity type to the place policy, or refuse it."""
    if entity_type not in KNOWN_LABELS:
        message = f"unknown MultiCoNER II entity type: {entity_type!r}"
        raise UnknownLabelError(message)
    return "LOC" if entity_type in PLACE_LABELS else "ignore"


def label_mapping() -> dict[str, Mapped]:
    """Return the complete mapping recorded in a protocol configuration."""
    return {label: label_for(label) for label in sorted(KNOWN_LABELS)}
