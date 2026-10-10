"""Shared loaders for the project contract tests.

Each loader keeps the parser that the original assertions used, so a test that
moved to another module reads the same data as before.
"""

from typing import Any

import yaml
import yaml.constructor
import yaml.resolver

from tests.conftest import PROJECT_ROOT

try:
    import tomllib  # ty: ignore[unresolved-import]
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10 CI.
    import tomli as tomllib


class UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate keys within one mapping."""


def _construct_unique_mapping(
    loader: Any, node: Any, deep: bool = False
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                None, None, f"duplicate key {key!r}", key_node.start_mark
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def load_pyproject() -> dict[str, Any]:
    """Parse pyproject.toml with the standard-library parser."""
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as pyproject_file:
        return tomllib.load(pyproject_file)


def job_steps(workflow_name: str, job_name: str) -> list[dict[str, Any]]:
    """Load one workflow job's steps for focused contract assertions."""
    workflow = yaml.load(
        (PROJECT_ROOT / ".github/workflows" / workflow_name).read_text(
            encoding="utf-8"
        ),
        Loader=yaml.BaseLoader,
    )
    return workflow["jobs"][job_name]["steps"]


def named_step(steps: list[dict[str, Any]], name: str) -> dict[str, Any]:
    """Select a named workflow step without duplicating lookups in tests."""
    return next(step for step in steps if step.get("name") == name)
