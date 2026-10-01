"""Helpers for following Markdown page links in nested MkDocs navigation."""

from __future__ import annotations


def markdown_paths(items: list[object]) -> list[str]:
    """Return Markdown page references nested in MkDocs navigation data."""
    paths = []
    for item in items:
        if isinstance(item, dict):
            paths.extend(_mapping_paths(item))
    return paths


def _mapping_paths(item: dict[object, object]) -> list[str]:
    paths = []
    for value in item.values():
        paths.extend(_value_paths(value))
    return paths


def _value_paths(value: object) -> list[str]:
    if isinstance(value, str):
        if value.endswith(".md"):
            return [value]
        return []
    if isinstance(value, list):
        return markdown_paths(value)
    return []
