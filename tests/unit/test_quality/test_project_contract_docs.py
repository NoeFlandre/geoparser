"""Contract tests for the MkDocs site, the contributor guides, and the issue and pull-request templates."""

import yaml

from tests.conftest import PROJECT_ROOT
from tests.unit.test_quality.mkdocs_navigation import markdown_paths


def test_mkdocs_configuration_has_strict_site_identity() -> None:
    configuration = PROJECT_ROOT / "mkdocs.yml"
    assert configuration.is_file()
    parsed = yaml.safe_load(configuration.read_text(encoding="utf-8"))

    assert parsed["site_name"]
    assert parsed["site_url"].startswith("https://")
    assert parsed["strict"] is True


def test_mkdocs_loads_python_api_documentation() -> None:
    parsed = yaml.safe_load((PROJECT_ROOT / "mkdocs.yml").read_text(encoding="utf-8"))
    assert any(
        isinstance(plugin, dict) and "mkdocstrings" in plugin
        for plugin in parsed["plugins"]
    )


def test_public_documentation_has_no_reStructuredText_pages() -> None:
    assert not list((PROJECT_ROOT / "docs").rglob("*.rst"))


def test_mkdocs_navigation_links_to_existing_pages() -> None:
    parsed = yaml.safe_load((PROJECT_ROOT / "mkdocs.yml").read_text(encoding="utf-8"))
    for relative_path in markdown_paths(parsed["nav"]):
        assert (PROJECT_ROOT / "docs" / relative_path).is_file(), relative_path


def test_retargeting_requires_fresh_validation_is_documented() -> None:
    guide = (PROJECT_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    assert "retargeting" in guide
    assert "PR rebuilds the new base comparison" in guide


def test_contributing_documents_precommit_installation() -> None:
    contributing = (PROJECT_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    assert "uv run pre-commit install" in contributing


def test_contributing_documents_package_coverage_scope() -> None:
    contributing = (PROJECT_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")

    assert "coverage report --include='geoparser/*' --fail-under=100" in contributing
    assert (
        "GEOPARSER_TEST_REMOTE_MODELS=1 uv run python scripts/quality_gauntlet.py"
        in contributing
    )


def test_development_docs_describe_distinct_coverage_and_crap_scopes() -> None:
    development = (PROJECT_ROOT / "docs/development.md").read_text(encoding="utf-8")

    assert (
        "GEOPARSER_TEST_REMOTE_MODELS=1 uv run python scripts/quality_gauntlet.py"
        in development
    )
    assert "100% line-coverage threshold applies to `geoparser/`" in development
    assert "`geoparser/`, `scripts/`, and `tests/`" in development


def test_bug_issue_template_explains_reproduction() -> None:
    issue_dir = PROJECT_ROOT / ".github" / "ISSUE_TEMPLATE"
    issue_config_path = issue_dir / "config.yml"
    bug_report = (issue_dir / "bug_report.yml").read_text(encoding="utf-8")
    bug_form = yaml.safe_load(bug_report)
    assert issue_config_path.is_file()
    issue_config = yaml.safe_load(issue_config_path.read_text(encoding="utf-8"))
    assert bug_form["name"] == "Bug report"
    assert issue_config["blank_issues_enabled"] is False
    assert "steps to reproduce" in bug_report.lower()


def test_feature_issue_template_asks_for_a_proposed_solution() -> None:
    feature_request = (
        PROJECT_ROOT / ".github" / "ISSUE_TEMPLATE" / "feature_request.yml"
    ).read_text(encoding="utf-8")
    feature_form = yaml.safe_load(feature_request)
    assert feature_form["name"] == "Feature request"
    assert "proposed solution" in feature_request.lower()


def test_pull_request_template_mentions_release_notes() -> None:
    pull_request = (PROJECT_ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md").read_text(
        encoding="utf-8"
    )
    assert "CHANGELOG.md" in pull_request
