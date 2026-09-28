"""Extract the curated changelog section for a release tag."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def extract_release_notes(changelog: str, tag: str) -> str:
    """Return the stable version's notes for a release tag.

    Release candidates use the notes for their eventual stable version, so
    ``0.6.0rc2`` resolves to the ``0.6.0`` section.
    """
    version = re.sub(r"rc\d+$", "", tag)
    headings = list(re.finditer(r"(?m)^## \[([^\]]+)\].*$", changelog))
    selected = next(
        (index for index, heading in enumerate(headings) if heading[1] == version),
        None,
    )
    if selected is None:
        msg = f"No changelog section for {version}"
        raise ValueError(msg)

    section_start = headings[selected].end()
    section_end = (
        headings[selected + 1].start()
        if selected + 1 < len(headings)
        else len(changelog)
    )
    notes = changelog[section_start:section_end].strip()
    if not notes:
        msg = f"Changelog section for {version} is empty"
        raise ValueError(msg)
    return notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag", help="release tag, for example 0.6.0 or 0.6.0rc1")
    parser.add_argument("--changelog", type=Path, default=Path("CHANGELOG.md"))
    args = parser.parse_args(argv)

    try:
        changelog = args.changelog.read_text(encoding="utf-8")
        notes = extract_release_notes(changelog, args.tag)
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1

    print(notes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
