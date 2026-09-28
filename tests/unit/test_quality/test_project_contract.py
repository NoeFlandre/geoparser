import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
import yaml.constructor
import yaml.resolver

from tests.unit import test_docs as docs_guard

try:
    import tomllib  # ty: ignore[unresolved-import]
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10 CI.
    import tomli as tomllib

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _bash_executable() -> str:
    if sys.platform == "win32":
        git = shutil.which("git")
        if git is not None:
            git_bash = Path(git).parent.parent / "bin" / "bash.exe"
            if git_bash.is_file():
                return str(git_bash)
        raise FileNotFoundError("Git Bash is required to parse workflow shell scripts")
    return shutil.which("bash") or "bash"


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


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


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def test_citation_version_matches_project_version() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as pyproject_file:
        project = tomllib.load(pyproject_file)
    citation = yaml.safe_load(
        (PROJECT_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    )

    assert citation["version"] == project["project"]["version"]


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
    assert "toml" in dependency_names
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
        ".pre-commit-config.yaml",
        "CHANGELOG.md",
        "mkdocs.yml",
    } <= copied_paths
    # The documentation guard reads these public surfaces directly, so a
    # mutant run that left them behind would fail for want of a file rather
    # than because a mutant survived.
    assert set(docs_guard.PUBLIC_ROOTS) <= copied_paths


def test_public_documentation_uses_strict_mkdocs_material() -> None:
    configuration = PROJECT_ROOT / "mkdocs.yml"
    assert configuration.is_file()
    parsed = yaml.safe_load(configuration.read_text(encoding="utf-8"))

    assert isinstance(parsed["site_name"], str) and parsed["site_name"]
    assert parsed["site_url"].startswith("https://")
    assert parsed["strict"] is True
    assert any(
        isinstance(plugin, dict) and "mkdocstrings" in plugin
        for plugin in parsed["plugins"]
    )
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
    import toml

    toml.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_runtime_packaging_is_locked_and_does_not_copy_local_state() -> None:
    dockerfile = PROJECT_ROOT / "Dockerfile"
    dockerignore = PROJECT_ROOT / ".dockerignore"
    citation = PROJECT_ROOT / "CITATION.cff"
    pyproject = PROJECT_ROOT / "pyproject.toml"

    assert dockerfile.is_file()
    assert dockerignore.is_file()
    assert citation.is_file()

    dockerignore_content = dockerignore.read_text(encoding="utf-8")
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
    assert citation_data["license"] == "MIT"
    assert citation_data["repository-code"].startswith("https://github.com/")
    assert len(citation_data["authors"]) >= 1


def test_ci_runs_the_full_gate_and_publishes_strict_mkdocs() -> None:
    quality = yaml.safe_load(
        (PROJECT_ROOT / ".github/workflows/quality.yml").read_text(encoding="utf-8")
    )
    docs = yaml.safe_load(
        (PROJECT_ROOT / ".github/workflows/docs.yml").read_text(encoding="utf-8")
    )
    quality_triggers = quality.get("on", quality.get(True, {}))
    quality_steps = quality["jobs"]["quality"]["steps"]
    quality_commands = [step.get("run", "") for step in quality_steps]
    docs_build_steps = docs["jobs"]["build"]["steps"]
    deploy_steps = docs["jobs"]["deploy"]["steps"]

    assert "pull_request" in quality_triggers
    assert any("scripts/quality_gauntlet.py" in command for command in quality_commands)
    quality_run = next(
        step["run"]
        for step in quality_steps
        if step.get("name") == "Run the complete deterministic quality gauntlet"
    )
    scheduled_run = quality_run.split("else", maxsplit=1)[0]
    assert "--skip-mutation" not in scheduled_run
    assert "--skip-mutation" in quality_run
    assert any(
        "mkdocs build --strict" in step.get("run", "") for step in docs_build_steps
    )
    assert any("deploy-pages" in step.get("uses", "") for step in deploy_steps)


def test_pull_request_base_edits_trigger_guarded_ci() -> None:
    """Changing a PR base starts CI, while title and description edits do not."""
    workflows_dir = PROJECT_ROOT / ".github/workflows"
    for path in sorted(workflows_dir.glob("*.yml")):
        workflow = yaml.load(
            path.read_text(encoding="utf-8"),
            Loader=yaml.BaseLoader,
        )
        if "pull_request" not in workflow.get("on", {}):
            continue
        pull_request = workflow["on"]["pull_request"]
        assert {"opened", "synchronize", "reopened", "edited"} <= set(
            pull_request["types"]
        )
        guards = [
            str(job.get("if", "")).replace(" ", "") for job in workflow["jobs"].values()
        ]
        assert any(
            "github.event.changes.base" in guard and "edited" in guard
            for guard in guards
        )
        for job in workflow["jobs"].values():
            if "github.event.changes.base" in str(job.get("if", "")):
                assert "metadata-edit-ignored" in str(job.get("name", ""))
        cancellation_policies = []
        if "cancel-in-progress" in workflow.get("concurrency", {}):
            cancellation_policies.append(workflow["concurrency"])
        cancellation_policies.extend(
            job["concurrency"]
            for job in workflow["jobs"].values()
            if "cancel-in-progress" in job.get("concurrency", {})
            and "github.event.changes.base" in str(job.get("if", ""))
        )
        for policy in cancellation_policies:
            cancel_condition = str(policy["cancel-in-progress"])
            group = str(policy["group"])
            assert "github.event.action != 'edited'" in cancel_condition
            assert "github.event.changes.base != null" in cancel_condition
            assert "metadata-" in group
            assert "github.run_id" in group


def test_metadata_edits_do_not_cancel_or_satisfy_the_test_gate() -> None:
    workflow = yaml.load(
        (PROJECT_ROOT / ".github/workflows/test.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    cancel_condition = str(workflow["concurrency"]["cancel-in-progress"])
    test_gate = workflow["jobs"]["tests-passed"]

    assert "github.event.action != 'edited'" in cancel_condition
    assert "github.event.changes.base != null" in cancel_condition
    assert "metadata-edit-ignored" in test_gate["name"]
    assert "tests-passed" in test_gate["name"]


def test_github_workflows_have_no_duplicate_yaml_keys() -> None:
    workflows_dir = PROJECT_ROOT / ".github/workflows"
    for workflow in workflows_dir.glob("*.yml"):
        yaml.load(workflow.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)


def test_github_workflow_bash_scripts_parse() -> None:
    workflows_dir = PROJECT_ROOT / ".github/workflows"
    for workflow_path in workflows_dir.glob("*.yml"):
        workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
        for job_name, job in workflow.get("jobs", {}).items():
            for step in job["steps"]:
                if step.get("shell", "bash") != "bash" or "run" not in step:
                    continue
                result = subprocess.run(
                    [_bash_executable(), "-n"],
                    input=step["run"],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                assert result.returncode == 0, (
                    f"{workflow_path.name}:{job_name}: {result.stderr}"
                )


def test_windows_uses_git_bash_instead_of_the_wsl_shim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    git = tmp_path / "Git" / "cmd" / "git.exe"
    git.parent.mkdir(parents=True)
    git.touch()
    git_bash = tmp_path / "Git" / "bin" / "bash.exe"
    git_bash.parent.mkdir(parents=True)
    git_bash.touch()

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(
        shutil,
        "which",
        lambda command: (
            str(git) if command == "git" else r"C:\Windows\System32\bash.exe"
        ),
    )

    assert _bash_executable() == str(git_bash)


def test_docker_workflow_runs_the_runtime_cli() -> None:
    workflow = yaml.safe_load(
        (PROJECT_ROOT / ".github/workflows/docker.yml").read_text(encoding="utf-8")
    )
    commands = [
        step["run"]
        for job in workflow["jobs"].values()
        for step in job["steps"]
        if "run" in step
    ]

    assert any(
        "docker run --rm geoparser:smoke --help" in command for command in commands
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
    assert {"check-added-large-files", "check-merge-conflict"} <= hooks

    ruff_repository = next(
        repository
        for repository in config["repos"]
        if repository["repo"] == "https://github.com/astral-sh/ruff-pre-commit"
    )
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as pyproject_file:
        project = tomllib.load(pyproject_file)
    ruff_requirement = next(
        requirement
        for requirement in project["dependency-groups"]["lint"]
        if requirement.startswith("ruff==")
    )
    assert ruff_requirement == f"ruff=={ruff_repository['rev'].removeprefix('v')}"
    contributing = (PROJECT_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    assert "uv run pre-commit install" in contributing


def test_precommit_is_available_after_the_documented_sync() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as pyproject_file:
        project = tomllib.load(pyproject_file)

    lint_dependencies = project["dependency-groups"]["lint"]
    assert any(
        dependency.startswith("pre-commit")
        for dependency in lint_dependencies
        if isinstance(dependency, str)
    )


def test_github_templates_cover_bug_feature_and_release_notes() -> None:
    issue_dir = PROJECT_ROOT / ".github" / "ISSUE_TEMPLATE"
    issue_config_path = issue_dir / "config.yml"
    bug_report = (issue_dir / "bug_report.yml").read_text(encoding="utf-8")
    feature_request = (issue_dir / "feature_request.yml").read_text(encoding="utf-8")
    pull_request = (PROJECT_ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md").read_text(
        encoding="utf-8"
    )
    bug_form = yaml.safe_load(bug_report)
    feature_form = yaml.safe_load(feature_request)
    assert issue_config_path.is_file()
    issue_config = yaml.safe_load(issue_config_path.read_text(encoding="utf-8"))

    assert bug_form["name"] == "Bug report"
    assert feature_form["name"] == "Feature request"
    assert issue_config["blank_issues_enabled"] is False
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


def test_github_release_downloads_notes_from_the_build_job() -> None:
    workflow = yaml.load(
        (PROJECT_ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    steps = workflow["jobs"]["github-release"]["steps"]
    download_index = next(
        index
        for index, step in enumerate(steps)
        if step.get("name") == "Download curated release notes"
    )
    release_index = next(
        index
        for index, step in enumerate(steps)
        if step.get("name") == "Publish the GitHub Release"
    )

    assert steps[download_index]["with"] == {
        "name": "release-notes",
        "path": "release-notes",
    }
    assert download_index < release_index


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
    assert re.search(r"^## \[\d+\.\d+\.\d+\]$", changelog, re.MULTILINE)


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
    assert any(
        "mutation_gate.py" in command and "--patterns" in command
        for command in commands
    )
    assert any("mutmut results --all true" in command for command in commands)
    assert any("mutmut show" in command for command in commands)


def test_quality_gate_fails_when_changed_mutation_fails() -> None:
    workflow = yaml.load(
        (PROJECT_ROOT / ".github/workflows/quality.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    job = workflow["jobs"]["quality"]
    guard = job["steps"][0]

    assert job["needs"] == ["changed-mutation"]
    assert "always()" in job["if"]
    assert "always()" in guard["if"]
    assert "needs.changed-mutation.result != 'success'" in guard["if"]
    assert "exit 1" in guard["run"]


def test_test_only_changes_run_the_full_mutation_sweep() -> None:
    workflow = yaml.load(
        (PROJECT_ROOT / ".github/workflows/quality.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    job = workflow["jobs"]["changed-mutation"]
    script = next(
        step["run"]
        for step in job["steps"]
        if step.get("name") == "Mutate changed package modules"
    )

    assert '"FULL_MUTATION"' in script
    assert "--max-no-tests 69" in script
    assert job["timeout-minutes"] == "240"


def test_changed_mutation_runs_serially_to_avoid_pytest_temp_races() -> None:
    workflow = yaml.load(
        (PROJECT_ROOT / ".github/workflows/quality.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    script = next(
        step["run"]
        for step in workflow["jobs"]["changed-mutation"]["steps"]
        if step.get("name") == "Mutate changed package modules"
    )

    assert script.count("mutmut run --max-children 1") == 2


def test_changed_mutation_job_installs_project_and_test_dependencies() -> None:
    workflow = yaml.load(
        (PROJECT_ROOT / ".github/workflows/quality.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    install_step = next(
        step
        for step in workflow["jobs"]["changed-mutation"]["steps"]
        if step.get("name") == "Install locked test dependencies"
    )

    assert install_step["run"] == "uv sync --locked --no-default-groups --group test"


def test_changed_mutation_job_installs_duckdb_spatial_extension() -> None:
    workflow = yaml.load(
        (PROJECT_ROOT / ".github/workflows/quality.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    steps = workflow["jobs"]["changed-mutation"]["steps"]
    install_step = next(
        step for step in steps if step.get("name") == "Install DuckDB spatial extension"
    )
    mutation_step = next(
        step for step in steps if step.get("name") == "Mutate changed package modules"
    )

    assert "install_extension('spatial')" in install_step["run"]
    assert steps.index(install_step) < steps.index(mutation_step)


def test_quality_workflow_skips_the_full_mutation_sweep_on_pull_requests() -> None:
    workflow = yaml.load(
        (PROJECT_ROOT / ".github/workflows/quality.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    quality_step = next(
        step
        for step in workflow["jobs"]["quality"]["steps"]
        if step.get("name") == "Run the complete deterministic quality gauntlet"
    )

    assert (
        "quality_gauntlet.py --skip-baseline --skip-docker --skip-mutation"
        in quality_step["run"]
    )


def test_deptry_is_installed_and_run_by_the_lightweight_lint_job() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as pyproject_file:
        project = tomllib.load(pyproject_file)
    lint_dependencies = {
        _package_name(dependency)
        for dependency in project["dependency-groups"]["lint"]
        if isinstance(dependency, str)
    }
    workflow = yaml.safe_load(
        (PROJECT_ROOT / ".github/workflows/lint.yml").read_text(encoding="utf-8")
    )
    ruff_steps = workflow["jobs"]["ruff"]["steps"]
    commands = [step.get("run", "") for step in ruff_steps]

    assert "deptry" in lint_dependencies
    assert any("deptry ." in command for command in commands)


def test_deptry_ignores_only_documented_runtime_and_tool_dependencies() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as pyproject_file:
        project = tomllib.load(pyproject_file)
    ignored = project["tool"]["deptry"]["per_rule_ignores"]

    assert set(ignored["DEP002"]) == {
        "accelerate",
        "peft",
        "protobuf",
        "python-multipart",
        "sentencepiece",
        "spacy-curated-transformers",
    }
    assert "DEP003" not in ignored
    assert set(ignored["DEP004"]) == {
        "coverage",
        "huggingface_hub",
        "mutmut",
        "plotly",
        "radon",
        "toml",
    }
    pyproject_text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "# DEP002:" in pyproject_text
    assert "# DEP004:" in pyproject_text


def test_benchmark_checkout_disables_persisted_credentials() -> None:
    workflow = yaml.safe_load(
        (PROJECT_ROOT / ".github/workflows/benchmark.yml").read_text(encoding="utf-8")
    )
    checkout = next(
        step
        for step in workflow["jobs"]["compare"]["steps"]
        if step.get("uses", "").startswith("actions/checkout@")
    )

    assert checkout["with"]["persist-credentials"] is False
    assert checkout["with"]["fetch-depth"] == 0


def test_benchmark_workflow_runs_algorithmic_guards_without_timings() -> None:
    workflow = yaml.safe_load(
        (PROJECT_ROOT / ".github/workflows/benchmark.yml").read_text(encoding="utf-8")
    )
    guard_step = next(
        (
            step
            for step in workflow["jobs"]["compare"]["steps"]
            if step.get("name") == "Run algorithmic guards"
        ),
        None,
    )

    assert guard_step is not None
    assert "--benchmark-disable" in guard_step["run"]
    assert "test_guards.py" in guard_step["run"]


def test_benchmark_dispatch_uses_base_ref_and_its_locked_environment() -> None:
    workflow = yaml.load(
        (PROJECT_ROOT / ".github/workflows/benchmark.yml").read_text(encoding="utf-8"),
        Loader=_UniqueKeyLoader,
    )
    triggers = workflow.get("on", workflow.get(True))
    dispatch = triggers["workflow_dispatch"]
    base_step = next(
        step
        for step in workflow["jobs"]["compare"]["steps"]
        if step.get("name") == "Benchmark pull request base"
    )

    assert dispatch["inputs"]["base_ref"]["default"] == "main"
    assert '"$BASE_REF"' in base_step["run"]
    assert "uv sync --locked --project .tmp/main" in base_step["run"]
    assert (
        'uv pip install --python .tmp/main/.venv/bin/python "pytest-benchmark==5.3.0"'
        in base_step["run"]
    )
    assert "$GITHUB_WORKSPACE/.tmp/main/.venv/bin/python" in base_step["run"]
