"""Contract tests for package metadata and the tool settings in pyproject.toml."""

import re

import pytest
import yaml

from tests.conftest import PROJECT_ROOT
from tests.unit.test_quality.project_contract_support import load_pyproject

try:
    import tomllib  # ty: ignore[unresolved-import]
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10 CI.
    import tomli as tomllib


def test_citation_version_matches_project_version() -> None:
    project = load_pyproject()
    citation = yaml.safe_load(
        (PROJECT_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    )

    assert citation["version"] == project["project"]["version"]


def test_citation_records_project_identity() -> None:
    citation = PROJECT_ROOT / "CITATION.cff"
    citation_data = yaml.safe_load(citation.read_text(encoding="utf-8"))
    assert citation_data["cff-version"] == "1.2.0"
    assert citation_data["title"] == "Irchel Geoparser"
    assert citation_data["license"] == "MIT"


def test_citation_links_the_repository_and_authors() -> None:
    citation_data = yaml.safe_load(
        (PROJECT_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    )
    assert citation_data["repository-code"].startswith("https://github.com/")
    assert len(citation_data["authors"]) >= 1


def test_manual_resolver_constructor_stays_within_crap_complexity_limit() -> None:
    radon = pytest.importorskip("radon.complexity")
    source = (PROJECT_ROOT / "geoparser/modules/resolvers/manual.py").read_text(
        encoding="utf-8"
    )
    constructor = next(
        block
        for block in radon.cc_visit(source)
        if block.fullname == "ManualResolver.__init__"
    )

    assert constructor.complexity <= 5


def _package_name(requirement: str) -> str:
    return re.split(r"[\[<>=!~;]", requirement, maxsplit=1)[0].strip().lower()


def test_quality_test_dependencies_are_declared() -> None:
    project = load_pyproject()

    test_dependencies = project["dependency-groups"]["test"]
    dependency_names = {
        _package_name(dependency)
        for dependency in test_dependencies
        if isinstance(dependency, str)
    }
    assert {"hypothesis", "pytest-bdd"} <= dependency_names
    assert "toml" not in dependency_names


def test_python310_declares_the_tomli_fallback() -> None:
    project = load_pyproject()
    test_dependencies = project["dependency-groups"]["test"]
    assert any(
        dependency.startswith("tomli") and 'python_version < "3.11"' in dependency
        for dependency in test_dependencies
        if isinstance(dependency, str)
    )


def test_pytest_quality_markers_and_temp_policy_are_declared() -> None:
    project = load_pyproject()
    marker_names = {
        marker.split(":", maxsplit=1)[0].strip()
        for marker in project["tool"]["pytest"]["ini_options"]["markers"]
    }
    assert {"property", "acceptance", "architecture"} <= marker_names
    assert (
        project["tool"]["pytest"]["ini_options"]["tmp_path_retention_policy"]
        == "failed"
    )


def test_ty_allows_the_optional_tomli_fallback() -> None:
    project = load_pyproject()

    assert project["tool"]["ty"]["analysis"]["allowed-unresolved-imports"] == ["tomli"]


def test_pyproject_parses_with_the_stdlib_parser_and_keeps_mutmut_config() -> None:
    project = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert "mutmut" in project["tool"]


def test_package_ships_the_pep561_typed_marker() -> None:
    assert (PROJECT_ROOT / "geoparser" / "py.typed").is_file()
    project = load_pyproject()

    assert "Typing :: Typed" in project["project"]["classifiers"]
    wheel = project["tool"]["hatch"]["build"]["targets"]["wheel"]
    assert "geoparser/py.typed" in wheel["include"]


def test_project_metadata_distributes_the_changelog() -> None:
    project = load_pyproject()
    assert project["project"]["urls"]["Changelog"].endswith("/CHANGELOG.md")
    assert (
        "CHANGELOG.md"
        in project["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]
    )


def test_changelog_has_unreleased_and_versioned_entries() -> None:
    changelog = (PROJECT_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "## [Unreleased]" in changelog
    assert re.search(r"^## \[\d+\.\d+\.\d+\]$", changelog, re.MULTILINE)


def test_deptry_is_declared_in_lint_dependencies() -> None:
    project = load_pyproject()
    lint_dependencies = {
        _package_name(dependency)
        for dependency in project["dependency-groups"]["lint"]
        if isinstance(dependency, str)
    }
    assert "deptry" in lint_dependencies


def _deptry_ignored_rules() -> dict[str, list[str]]:
    project = load_pyproject()
    return project["tool"]["deptry"]["per_rule_ignores"]


def test_deptry_ignores_only_documented_runtime_dependencies() -> None:
    ignored = _deptry_ignored_rules()
    assert set(ignored["DEP002"]) == {
        "accelerate",
        "numpy",
        "peft",
        "protobuf",
        "python-multipart",
        "safetensors",
        "sentencepiece",
        "spacy-curated-transformers",
    }
    assert "DEP003" not in ignored


def test_deptry_ignores_only_documented_tool_dependencies() -> None:
    ignored = _deptry_ignored_rules()
    assert set(ignored["DEP004"]) == {
        "coverage",
        "huggingface_hub",
        "mutmut",
        "plotly",
        "radon",
        "tomli",
    }


def test_deptry_ignored_rules_are_documented_in_pyproject() -> None:
    pyproject_text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "# DEP002:" in pyproject_text
    assert "# DEP004:" in pyproject_text
