import re
from pathlib import Path

import pytest
import yaml

from tests.unit import test_docs as docs_guard

try:
    import tomllib  # ty: ignore[unresolved-import]
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10 CI.
    import tomli as tomllib

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _package_name(requirement: str) -> str:
    return re.split(r"[\[<>=!~;]", requirement, maxsplit=1)[0].strip().lower()


def test_project_quality_dependencies_and_pytest_markers_are_declared() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as pyproject_file:
        project = tomllib.load(pyproject_file)

    test_dependencies = project["dependency-groups"]["test"]
    dependency_names = {
        _package_name(dependency)
        for dependency in test_dependencies
        if isinstance(dependency, str)
    }
    assert {"hypothesis", "pytest-bdd"} <= dependency_names
    assert any(
        dependency.startswith("tomli") and 'python_version < "3.11"' in dependency
        for dependency in test_dependencies
        if isinstance(dependency, str)
    )

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
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as pyproject_file:
        project = tomllib.load(pyproject_file)

    assert project["tool"]["ty"]["analysis"]["allowed-unresolved-imports"] == ["tomli"]


def test_mutation_runner_copies_quality_support_modules() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as pyproject_file:
        project = tomllib.load(pyproject_file)

    copied_paths = set(project["tool"]["mutmut"]["also_copy"])
    assert {
        "docs",
        "scripts",
        ".github",
        ".dockerignore",
        "CITATION.cff",
        "Dockerfile",
        "mkdocs.yml",
    } <= copied_paths
    # The documentation guard reads these public surfaces directly, so a
    # mutant run that left them behind would fail for want of a file rather
    # than because a mutant survived.
    assert set(docs_guard.PUBLIC_ROOTS) <= copied_paths


def test_public_documentation_uses_strict_mkdocs_material() -> None:
    configuration = PROJECT_ROOT / "mkdocs.yml"
    assert configuration.is_file()
    content = configuration.read_text(encoding="utf-8")
    parsed = yaml.safe_load(content)

    assert "site_name: Irchel Geoparser" in content
    assert "site_url: https://docs.geoparser.app/" in content
    assert "strict: true" in content
    assert "mkdocstrings" in content
    assert "- Home: index.md" in content
    assert parsed["strict"] is True
    assert parsed["plugins"][1] == {
        "mkdocstrings": {
            "handlers": {
                "python": {
                    "options": {
                        "allow_inspection": True,
                        "annotations_path": "brief",
                        "docstring_section_style": "table",
                        "docstring_style": "google",
                        "docstring_options": {
                            "warn_missing_types": False,
                            "warn_unknown_params": False,
                        },
                        "force_inspection": True,
                        "members_order": "source",
                        "separate_signature": True,
                        "show_object_full_path": False,
                        "show_root_heading": True,
                        "show_source": False,
                    }
                }
            }
        }
    }
    assert not list((PROJECT_ROOT / "docs").rglob("*.rst"))

    def nav_paths(items: list[object]) -> list[str]:
        paths: list[str] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            for value in item.values():
                if isinstance(value, str) and value.endswith(".md"):
                    paths.append(value)
                elif isinstance(value, list):
                    paths.extend(nav_paths(value))
        return paths

    for relative_path in nav_paths(parsed["nav"]):
        assert (PROJECT_ROOT / "docs" / relative_path).is_file(), relative_path


def test_pyproject_is_compatible_with_mutmut_legacy_toml_parser() -> None:
    legacy_toml = pytest.importorskip("toml")

    legacy_toml.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_runtime_packaging_is_locked_and_does_not_copy_local_state() -> None:
    dockerfile = PROJECT_ROOT / "Dockerfile"
    dockerignore = PROJECT_ROOT / ".dockerignore"
    citation = PROJECT_ROOT / "CITATION.cff"
    pyproject = PROJECT_ROOT / "pyproject.toml"

    assert dockerfile.is_file()
    assert dockerignore.is_file()
    assert citation.is_file()

    dockerfile_content = dockerfile.read_text(encoding="utf-8")
    dockerignore_content = dockerignore.read_text(encoding="utf-8")
    assert "python:3.12-slim" in dockerfile_content
    assert "uv sync --locked --no-dev" in dockerfile_content
    assert 'ENTRYPOINT ["python", "-m", "geoparser"]' in dockerfile_content
    assert 'CMD ["--help"]' in dockerfile_content
    assert ".git" in dockerignore_content
    assert ".venv" in dockerignore_content
    assert "secrets" in dockerignore_content

    pyproject_content = pyproject.read_text(encoding="utf-8")
    assert 'name = "pytorch-cpu"' in pyproject_content
    assert 'url = "https://download.pytorch.org/whl/cpu"' in pyproject_content

    with (PROJECT_ROOT / "uv.lock").open("rb") as lockfile:
        lock = tomllib.load(lockfile)
    cpu_torch = [
        package
        for package in lock["package"]
        if package["name"] == "torch"
        and package.get("source", {}).get("registry")
        == "https://download.pytorch.org/whl/cpu"
    ]
    assert cpu_torch
    assert not any(
        package["name"].startswith(("cuda-", "nvidia-")) for package in lock["package"]
    )

    citation_data = yaml.safe_load(citation.read_text(encoding="utf-8"))
    assert citation_data["cff-version"] == "1.2.0"
    assert citation_data["title"] == "Irchel Geoparser"
    assert citation_data["version"] == "0.6.0"
    assert citation_data["license"] == "MIT"
    assert citation_data["repository-code"].startswith("https://github.com/")
    assert len(citation_data["authors"]) >= 1


def test_ci_runs_the_full_gate_and_publishes_strict_mkdocs() -> None:
    quality = (PROJECT_ROOT / ".github/workflows/quality.yml").read_text(
        encoding="utf-8"
    )
    docs = (PROJECT_ROOT / ".github/workflows/docs.yml").read_text(encoding="utf-8")

    assert "pull_request:" in quality
    assert "uv sync --locked" in quality
    assert "scripts/quality_gauntlet.py" in quality
    assert "--skip-mutation" not in quality
    assert "--skip-baseline" in quality
    assert "--skip-docker" in quality
    assert "mkdocs build --strict" in docs
    assert "deploy-pages" in docs


def test_pull_request_base_edits_trigger_guarded_ci() -> None:
    """Changing a PR base starts CI, while title and description edits do not."""
    workflows = ["test.yml", "lint.yml", "docs.yml", "quality.yml"]

    for filename in workflows:
        content = (PROJECT_ROOT / ".github/workflows" / filename).read_text(
            encoding="utf-8"
        )
        assert "types: [opened, synchronize, reopened, edited]" in content
        assert (
            "if: github.event_name != 'pull_request' || github.event.action != "
            "'edited' || github.event.changes.base != null" in content
        )


def test_test_matrix_runs_every_python_on_ubuntu_and_endpoints_elsewhere() -> None:
    """All Python versions stay covered with a smaller OS cross-product."""
    workflow = yaml.load(
        (PROJECT_ROOT / ".github/workflows/test.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    matrix = workflow["jobs"]["pytest"]["strategy"]["matrix"]

    assert len(matrix["include"]) == 9
    assert [
        entry["python-version"]
        for entry in matrix["include"]
        if entry["os"] == "ubuntu-latest"
    ] == ["3.10", "3.11", "3.12", "3.13", "3.14"]
    assert {
        (entry["os"], entry["python-version"])
        for entry in matrix["include"]
        if entry["os"] != "ubuntu-latest"
    } == {
        ("windows-latest", "3.10"),
        ("windows-latest", "3.14"),
        ("macos-latest", "3.10"),
        ("macos-latest", "3.14"),
    }


def test_nightly_quality_runs_remote_models_and_docker() -> None:
    """The scheduled gauntlet enables real integrations and both images."""
    quality = (PROJECT_ROOT / ".github/workflows/quality.yml").read_text(
        encoding="utf-8"
    )

    assert "GEOPARSER_TEST_REMOTE_MODELS" in quality
    assert "github.event_name == 'schedule'" in quality
    assert (
        "uv run --no-sync python scripts/quality_gauntlet.py --skip-baseline\n"
        in quality
    )
    assert (
        "uv run --no-sync python scripts/quality_gauntlet.py "
        "--skip-baseline --skip-docker" in quality
    )
    gauntlet = (PROJECT_ROOT / "scripts/quality_gauntlet.py").read_text(
        encoding="utf-8"
    )
    assert "demo/Dockerfile" in gauntlet


def test_property_workflows_select_ci_and_nightly_profiles() -> None:
    test_workflow = yaml.load(
        (PROJECT_ROOT / ".github/workflows/test.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    quality_workflow = yaml.load(
        (PROJECT_ROOT / ".github/workflows/quality.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    property_env = quality_workflow["jobs"]["quality"]["steps"][-1]["env"]

    assert test_workflow["env"]["HYPOTHESIS_PROFILE"] == "ci"
    assert property_env["HYPOTHESIS_PROFILE"] == (
        "${{ github.event_name == 'schedule' && 'nightly' || 'ci' }}"
    )


def test_fast_lint_workflow_runs_ty_with_the_lightweight_environment() -> None:
    workflow = yaml.load(
        (PROJECT_ROOT / ".github/workflows/lint.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    ty_job = workflow["jobs"]["ty"]
    commands = [step.get("run", "") for step in ty_job["steps"]]

    assert any(
        "uv sync --locked --no-default-groups --only-group lint --no-install-project"
        in command
        for command in commands
    )
    assert any(
        "ty check --config-file ty-lint.toml geoparser scripts tests" in command
        for command in commands
    )


def test_ci_pins_setup_uv_to_a_resolvable_release() -> None:
    workflow_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((PROJECT_ROOT / ".github/workflows").glob("*.yml"))
    )

    # Pinned by commit SHA, with the release it resolves to named alongside.
    assert re.search(r"astral-sh/setup-uv@[0-9a-f]{40} # v10\.1\.0\n", workflow_text)
    assert "astral-sh/setup-uv@v10\n" not in workflow_text


def test_every_action_is_pinned_to_a_commit_sha() -> None:
    uses = re.compile(r"^\s*(?:-\s*)?uses:\s*(\S+)", re.MULTILINE)
    unpinned = [
        f"{path.name}: {ref}"
        for path in sorted((PROJECT_ROOT / ".github/workflows").glob("*.yml"))
        for ref in uses.findall(path.read_text(encoding="utf-8"))
        if not ref.startswith("./") and not re.search(r"@[0-9a-f]{40}$", ref)
    ]

    assert unpinned == []


def test_precommit_config_runs_ruff_format_and_basic_file_checks() -> None:
    config = yaml.load(
        (PROJECT_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    hooks = {
        hook["id"] for repository in config["repos"] for hook in repository["hooks"]
    }

    assert {"ruff", "ruff-format", "check-yaml", "check-toml"} <= hooks
    assert "trailing-whitespace" in hooks
    assert "end-of-file-fixer" in hooks


def test_github_templates_cover_bug_feature_and_release_notes() -> None:
    issue_dir = PROJECT_ROOT / ".github" / "ISSUE_TEMPLATE"
    bug_report = (issue_dir / "bug_report.yml").read_text(encoding="utf-8")
    feature_request = (issue_dir / "feature_request.yml").read_text(encoding="utf-8")
    pull_request = (PROJECT_ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md").read_text(
        encoding="utf-8"
    )
    bug_form = yaml.safe_load(bug_report)
    feature_form = yaml.safe_load(feature_request)

    assert bug_form["name"] == "Bug report"
    assert feature_form["name"] == "Feature request"
    assert "steps to reproduce" in bug_report.lower()
    assert "proposed solution" in feature_request.lower()
    assert "CHANGELOG.md" in pull_request


def test_release_workflow_requires_changelog_notes_for_the_tag() -> None:
    workflow = yaml.load(
        (PROJECT_ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    build_steps = workflow["jobs"]["build"]["steps"]
    release_step = next(
        step
        for step in workflow["jobs"]["github-release"]["steps"]
        if "gh release create" in step.get("run", "")
    )

    assert any("scripts/changelog.py" in step.get("run", "") for step in build_steps)
    assert "--notes-file" in release_step["run"]
    assert "--generate-notes" not in release_step["run"]


def test_changelog_is_distributed_with_the_source_archive() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as pyproject_file:
        project = tomllib.load(pyproject_file)
    changelog = (PROJECT_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

    assert project["project"]["urls"]["Changelog"].endswith("/CHANGELOG.md")
    assert (
        "CHANGELOG.md"
        in project["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]
    )
    assert "## [Unreleased]" in changelog
    assert "## [0.6.0]" in changelog


def test_quality_workflow_mutates_changed_python_modules_on_pull_requests() -> None:
    workflow = yaml.load(
        (PROJECT_ROOT / ".github/workflows/quality.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    job = workflow["jobs"]["changed-mutation"]
    commands = [step.get("run", "") for step in job["steps"]]

    assert "github.event_name == 'pull_request'" in job["if"]
    assert any("changed_mutation_patterns.py" in command for command in commands)
    assert any("mutmut run" in command for command in commands)
    assert any("--max-no-tests 0" in command for command in commands)
