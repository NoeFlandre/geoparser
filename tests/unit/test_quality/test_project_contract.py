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
from tests.unit.test_quality.mkdocs_navigation import markdown_paths

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


def test_module_test_fragments_are_merged_into_their_owning_files() -> None:
    fragments = (
        "tests/unit/test_db/test_functions/test_levenshtein_nulls.py",
        "tests/unit/test_db/test_functions/test_soundex_reference.py",
        "tests/unit/test_db/test_crud/test_document_batching.py",
        "tests/unit/test_db/test_crud/test_reference_update.py",
        "tests/unit/test_db/test_crud/test_composite_filters.py",
        "tests/unit/test_db/test_models/test_module_repr.py",
        "tests/unit/test_modules/test_resolvers/test_manual_lookup.py",
        "tests/unit/test_modules/test_resolvers/test_context_window.py",
        "tests/unit/test_evaluation_distance.py",
        "tests/unit/test_project/test_project_dataflow.py",
        "tests/unit/test_project/test_project_persistence.py",
        "tests/unit/test_services/test_resolution_dataflow.py",
        "tests/unit/test_services/test_prediction_recording.py",
        "tests/unit/test_pilot_report.py",
    )

    remaining = [path for path in fragments if (PROJECT_ROOT / path).exists()]

    assert not remaining


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


def test_quality_test_dependencies_are_declared() -> None:
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


def test_python310_declares_the_tomli_fallback() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as pyproject_file:
        project = tomllib.load(pyproject_file)
    test_dependencies = project["dependency-groups"]["test"]
    assert any(
        dependency.startswith("tomli") and 'python_version < "3.11"' in dependency
        for dependency in test_dependencies
        if isinstance(dependency, str)
    )


def test_pytest_quality_markers_and_temp_policy_are_declared() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as pyproject_file:
        project = tomllib.load(pyproject_file)
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
        "MUTATION_TESTING.md",
    } <= copied_paths
    # The documentation guard reads these public surfaces directly, so a
    # mutant run that left them behind would fail for want of a file rather
    # than because a mutant survived.
    assert set(docs_guard.PUBLIC_ROOTS) <= copied_paths


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


def test_pyproject_is_compatible_with_mutmut_legacy_toml_parser() -> None:
    import toml

    toml.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_dockerfile_is_present() -> None:
    assert (PROJECT_ROOT / "Dockerfile").is_file()


def test_docker_ignore_excludes_local_state() -> None:
    dockerignore = PROJECT_ROOT / ".dockerignore"
    assert dockerignore.is_file()
    dockerignore_content = dockerignore.read_text(encoding="utf-8")
    assert ".git" in dockerignore_content
    assert ".venv" in dockerignore_content
    assert "secrets" in dockerignore_content


def test_pyproject_uses_the_cpu_torch_index() -> None:
    pyproject = PROJECT_ROOT / "pyproject.toml"
    pyproject_content = pyproject.read_text(encoding="utf-8")
    assert 'name = "pytorch-cpu"' in pyproject_content
    assert 'url = "https://download.pytorch.org/whl/cpu"' in pyproject_content


def _project_lock() -> dict[str, Any]:
    with (PROJECT_ROOT / "uv.lock").open("rb") as lockfile:
        return tomllib.load(lockfile)


def test_lock_resolves_torch_from_the_cpu_index() -> None:
    lock = _project_lock()
    cpu_torch = [
        package
        for package in lock["package"]
        if package["name"] == "torch"
        and package.get("source", {}).get("registry")
        == "https://download.pytorch.org/whl/cpu"
    ]
    assert cpu_torch


def test_lock_omits_accelerator_packages() -> None:
    lock = _project_lock()
    assert not any(
        package["name"].startswith(("cuda-", "nvidia-")) for package in lock["package"]
    )


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


def test_quality_workflow_runs_on_pull_requests() -> None:
    quality = yaml.safe_load(
        (PROJECT_ROOT / ".github/workflows/quality.yml").read_text(encoding="utf-8")
    )
    quality_triggers = quality.get("on", quality.get(True, {}))
    quality_steps = quality["jobs"]["quality"]["steps"]
    quality_commands = [step.get("run", "") for step in quality_steps]

    assert "pull_request" in quality_triggers
    assert any("scripts/quality_gauntlet.py" in command for command in quality_commands)


def test_quality_workflow_skips_mutation_only_for_scheduled_runs() -> None:
    quality = yaml.safe_load(
        (PROJECT_ROOT / ".github/workflows/quality.yml").read_text(encoding="utf-8")
    )
    quality_steps = quality["jobs"]["quality"]["steps"]
    quality_run = next(
        step["run"]
        for step in quality_steps
        if step.get("name") == "Run the complete deterministic quality gauntlet"
    )
    scheduled_run = quality_run.split("else", maxsplit=1)[0]
    assert "--skip-mutation" not in scheduled_run
    assert "--skip-mutation" in quality_run


def test_docs_workflow_builds_strictly_and_deploys_pages() -> None:
    docs = yaml.safe_load(
        (PROJECT_ROOT / ".github/workflows/docs.yml").read_text(encoding="utf-8")
    )
    docs_build_steps = docs["jobs"]["build"]["steps"]
    deploy_steps = docs["jobs"]["deploy"]["steps"]
    assert any(
        "mkdocs build --strict" in step.get("run", "") for step in docs_build_steps
    )
    assert any("deploy-pages" in step.get("uses", "") for step in deploy_steps)


def test_pull_request_base_edits_trigger_guarded_ci() -> None:
    """Changing a PR base starts CI, while title and description edits do not."""
    for workflow in _pull_request_workflows():
        _assert_base_edit_policy(workflow)


def _pull_request_workflows() -> list[dict[str, Any]]:
    """Load workflows that receive pull-request events."""
    workflows_dir = PROJECT_ROOT / ".github/workflows"
    found = []
    for path in sorted(workflows_dir.glob("*.yml")):
        workflow = yaml.load(
            path.read_text(encoding="utf-8"),
            Loader=yaml.BaseLoader,
        )
        if "pull_request" in workflow.get("on", {}):
            found.append(workflow)
    return found


def _assert_base_edit_policy(workflow: dict[str, Any]) -> None:
    """Check that edits to a PR base trigger guarded workflow jobs."""
    pull_request = workflow["on"]["pull_request"]
    assert {"opened", "synchronize", "reopened", "edited"} <= set(pull_request["types"])
    jobs = workflow["jobs"].values()
    assert _has_base_edit_guard(jobs)
    _assert_metadata_edit_jobs(jobs)
    _assert_metadata_cancellation_policies(_cancellation_policies(workflow))


def _has_base_edit_guard(jobs: Any) -> bool:
    """Whether a workflow job checks the changed PR base on edit events."""
    return any(
        "github.event.changes.base" in str(job.get("if", ""))
        and "edited" in str(job.get("if", ""))
        for job in jobs
    )


def _assert_metadata_edit_jobs(jobs: Any) -> None:
    """Require base-edit jobs to remain visible as metadata-only checks."""
    for job in jobs:
        if "github.event.changes.base" in str(job.get("if", "")):
            assert "metadata-edit-ignored" in str(job.get("name", ""))


def _cancellation_policies(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect cancellation guards that special-case pull-request edits."""
    jobs = workflow["jobs"].values()
    policies = []
    if "cancel-in-progress" in workflow.get("concurrency", {}):
        policies.append(workflow["concurrency"])
    policies.extend(_job_cancellation_policy(job) for job in jobs)
    return [policy for policy in policies if policy is not None]


def _job_cancellation_policy(job: dict[str, Any]) -> dict[str, Any] | None:
    """Return a job policy only when it guards cancellation on base edits."""
    concurrency = job.get("concurrency", {})
    if "cancel-in-progress" not in concurrency:
        return None
    if "github.event.changes.base" not in str(job.get("if", "")):
        return None
    return concurrency


def _assert_metadata_cancellation_policies(
    policies: list[dict[str, Any]],
) -> None:
    """Ensure metadata edits cannot cancel runs or join ordinary groups."""
    for policy in policies:
        _assert_metadata_cancellation_policy(policy)


def _assert_metadata_cancellation_policy(policy: dict[str, Any]) -> None:
    """Check one cancellation guard's event filter and concurrency group."""
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


def _is_bash_step(step: dict[str, Any]) -> bool:
    """Whether a workflow step declares an inline Bash command."""
    return step.get("shell", "bash") == "bash" and "run" in step


def _bash_script_cases() -> list[tuple[str, str, str]]:
    """Return each inline Bash command with its workflow and job labels."""
    cases = []
    workflows_dir = PROJECT_ROOT / ".github/workflows"
    for workflow_path in workflows_dir.glob("*.yml"):
        workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
        for job_name, job in workflow.get("jobs", {}).items():
            cases.extend(
                (workflow_path.name, job_name, step["run"])
                for step in job["steps"]
                if _is_bash_step(step)
            )
    return cases


def _assert_bash_command_parses(
    workflow_name: str, job_name: str, command: str
) -> None:
    """Ask Bash to parse one inline command without executing it."""
    result = subprocess.run(
        [_bash_executable(), "-n"],
        input=command,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, f"{workflow_name}:{job_name}: {result.stderr}"


def test_github_workflow_bash_scripts_parse() -> None:
    for workflow_name, job_name, command in _bash_script_cases():
        _assert_bash_command_parses(
            workflow_name,
            job_name,
            command,
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


def test_coverage_matrix_runs_opt_in_model_tests_in_one_linux_cell() -> None:
    pytest_step = _named_step(_job_steps("test.yml", "pytest"), "Run pytest")

    assert pytest_step["env"]["GEOPARSER_TEST_REMOTE_MODELS"] == (
        "${{ matrix.os == 'ubuntu-latest' && matrix.python-version == '3.12' && '1' || '' }}"
    )


def test_quality_gate_runs_remote_model_tests_for_complete_crap_coverage() -> None:
    quality_step = _named_step(
        _job_steps("quality.yml", "quality"),
        "Run the complete deterministic quality gauntlet",
    )

    assert quality_step["env"]["GEOPARSER_TEST_REMOTE_MODELS"] == "1"


def test_nightly_quality_builds_both_docker_images() -> None:
    """The scheduled gauntlet runs without the Docker skip switch."""
    quality = (PROJECT_ROOT / ".github/workflows/quality.yml").read_text(
        encoding="utf-8"
    )
    assert "uv run --no-sync python scripts/quality_gauntlet.py\n" in quality


def test_pull_request_quality_skips_docker_builds() -> None:
    quality = (PROJECT_ROOT / ".github/workflows/quality.yml").read_text(
        encoding="utf-8"
    )
    assert (
        "uv run --no-sync python scripts/quality_gauntlet.py "
        "--skip-docker --skip-mutation" in quality
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


def _job_steps(workflow_name: str, job_name: str) -> list[dict[str, Any]]:
    """Load one workflow job's steps for focused contract assertions."""
    workflow = yaml.load(
        (PROJECT_ROOT / ".github/workflows" / workflow_name).read_text(
            encoding="utf-8"
        ),
        Loader=yaml.BaseLoader,
    )
    return workflow["jobs"][job_name]["steps"]


def _named_step(steps: list[dict[str, Any]], name: str) -> dict[str, Any]:
    """Select a named workflow step without duplicating lookups in tests."""
    return next(step for step in steps if step.get("name") == name)


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


def _action_ref_is_unpinned(ref: str) -> bool:
    """Whether an external action is not pinned to a full commit SHA."""
    return not ref.startswith("./") and not re.search(r"@[0-9a-f]{40}$", ref)


def _unpinned_workflow_actions() -> list[str]:
    """Collect non-local workflow actions without immutable revisions."""
    uses = re.compile(r"^\s*(?:-\s*)?uses:\s*(\S+)", re.MULTILINE)
    unpinned = []
    for path in sorted((PROJECT_ROOT / ".github/workflows").glob("*.yml")):
        unpinned.extend(
            f"{path.name}: {ref}"
            for ref in uses.findall(path.read_text(encoding="utf-8"))
            if _action_ref_is_unpinned(ref)
        )
    return unpinned


def test_every_action_is_pinned_to_a_commit_sha() -> None:
    assert _unpinned_workflow_actions() == []


def _precommit_hook_ids() -> set[str]:
    config = yaml.load(
        (PROJECT_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    return {
        hook["id"] for repository in config["repos"] for hook in repository["hooks"]
    }


def test_precommit_config_has_ruff_and_format_checks() -> None:
    hooks = _precommit_hook_ids()
    assert {"ruff", "ruff-format", "check-yaml", "check-toml"} <= hooks


def test_precommit_config_has_trailing_space_and_eof_checks() -> None:
    hooks = _precommit_hook_ids()
    assert "trailing-whitespace" in hooks
    assert "end-of-file-fixer" in hooks


def test_precommit_config_checks_large_and_conflicted_files() -> None:
    hooks = _precommit_hook_ids()
    assert {"check-added-large-files", "check-merge-conflict"} <= hooks


def _ruff_precommit_revision() -> str:
    config = yaml.load(
        (PROJECT_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    ruff_repository = next(
        repository
        for repository in config["repos"]
        if repository["repo"] == "https://github.com/astral-sh/ruff-pre-commit"
    )
    return ruff_repository["rev"].removeprefix("v")


def _ruff_lint_dependency() -> str:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as pyproject_file:
        project = tomllib.load(pyproject_file)
    return next(
        requirement
        for requirement in project["dependency-groups"]["lint"]
        if requirement.startswith("ruff==")
    )


def test_precommit_ruff_revision_matches_the_locked_lint_dependency() -> None:
    assert _ruff_lint_dependency() == f"ruff=={_ruff_precommit_revision()}"


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


def test_precommit_is_available_after_the_documented_sync() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as pyproject_file:
        project = tomllib.load(pyproject_file)

    lint_dependencies = project["dependency-groups"]["lint"]
    assert any(
        dependency.startswith("pre-commit")
        for dependency in lint_dependencies
        if isinstance(dependency, str)
    )


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


def _release_workflow_steps() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    workflow = yaml.load(
        (PROJECT_ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    return (
        workflow["jobs"]["build"]["steps"],
        workflow["jobs"]["github-release"]["steps"],
    )


def test_release_build_creates_curated_changelog_notes() -> None:
    build_steps, _ = _release_workflow_steps()
    assert any("scripts/changelog.py" in step.get("run", "") for step in build_steps)


def test_release_uses_the_curated_notes_file() -> None:
    _, release_steps = _release_workflow_steps()
    release_step = next(
        step for step in release_steps if "gh release create" in step.get("run", "")
    )
    assert "--notes-file" in release_step["run"]
    assert "--generate-notes" not in release_step["run"]


def _release_step_index(steps: list[dict[str, Any]], name: str) -> int:
    """Find a named release workflow step in declaration order."""
    return next(index for index, step in enumerate(steps) if step.get("name") == name)


def test_github_release_downloads_curated_notes() -> None:
    _, release_steps = _release_workflow_steps()
    download = next(
        step
        for step in release_steps
        if step.get("name") == "Download curated release notes"
    )
    assert download["with"] == {
        "name": "release-notes",
        "path": "release-notes",
    }


def test_github_release_downloads_notes_before_publishing() -> None:
    _, release_steps = _release_workflow_steps()
    download_index = _release_step_index(
        release_steps, "Download curated release notes"
    )
    release_index = _release_step_index(release_steps, "Publish the GitHub Release")
    assert download_index < release_index


def test_project_metadata_distributes_the_changelog() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as pyproject_file:
        project = tomllib.load(pyproject_file)
    assert project["project"]["urls"]["Changelog"].endswith("/CHANGELOG.md")
    assert (
        "CHANGELOG.md"
        in project["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]
    )


def test_changelog_has_unreleased_and_versioned_entries() -> None:
    changelog = (PROJECT_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "## [Unreleased]" in changelog
    assert re.search(r"^## \[\d+\.\d+\.\d+\]$", changelog, re.MULTILINE)


def _mutation_report_text() -> str:
    return " ".join(
        (PROJECT_ROOT / "MUTATION_TESTING.md").read_text(encoding="utf-8").split()
    )


def test_mutation_report_records_the_latest_full_sweep() -> None:
    assert "`7cdd960` (2026-10-01)" in _mutation_report_text()


def test_mutation_report_keeps_timeout_results_inconclusive() -> None:
    report = _mutation_report_text()
    assert (
        "3,704 killed, zero survived, zero had no covering tests, and 32 timed out"
        in report
    )
    assert "the 32 timeouts remain inconclusive" in report
    assert "not established" in report


def test_mutation_report_keeps_historical_no_tests_allowance() -> None:
    report = _mutation_report_text()
    assert "`--max-no-tests 69` allowance" in report


def _changed_mutation_job() -> tuple[dict[str, Any], list[str]]:
    workflow = yaml.load(
        (PROJECT_ROOT / ".github/workflows/quality.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    job = workflow["jobs"]["changed-mutation"]
    commands = [step.get("run", "") for step in job["steps"]]
    return job, commands


def _has_command(commands: list[str], fragment: str) -> bool:
    """Whether any workflow step contains the requested shell fragment."""
    return any(fragment in command for command in commands)


def test_changed_mutation_job_is_pull_request_scoped() -> None:
    job, _ = _changed_mutation_job()
    assert "github.event_name == 'pull_request'" in job["if"]


def test_changed_mutation_job_selects_modules_and_runs_mutmut() -> None:
    _, commands = _changed_mutation_job()
    assert _has_command(commands, "changed_mutation_patterns.py")
    assert _has_command(commands, "mutmut run")


def test_changed_mutation_job_gates_and_reports_mutant_results() -> None:
    _, commands = _changed_mutation_job()
    assert _has_command(commands, "--max-no-tests 0")
    assert _has_command(commands, "mutation_gate.py")
    assert _has_command(commands, "mutation_evidence.py")


def _mutation_evidence_artifact_step() -> dict[str, Any]:
    job, _ = _changed_mutation_job()
    return next(
        step
        for step in job["steps"]
        if step.get("name") == "Upload per-mutant evidence"
    )


def test_changed_mutation_job_uploads_evidence_even_on_failure() -> None:
    artifact_step = _mutation_evidence_artifact_step()

    assert artifact_step["if"] == "always()"


def test_changed_mutation_artifact_name_identifies_run_and_head() -> None:
    artifact_step = _mutation_evidence_artifact_step()

    assert (
        "mutation-evidence-${{ github.run_id }}-${{ github.event.pull_request.head.sha }}"
        in (artifact_step["with"]["name"])
    )


def test_changed_mutation_artifact_keeps_exact_run_files() -> None:
    artifact_step = _mutation_evidence_artifact_step()

    assert artifact_step["with"]["path"] == "mutation-evidence/"
    assert artifact_step["with"]["retention-days"] == "90"


def test_quality_workflow_waits_for_changed_mutation() -> None:
    workflow = yaml.load(
        (PROJECT_ROOT / ".github/workflows/quality.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    job = workflow["jobs"]["quality"]

    assert job["needs"] == ["changed-mutation"]
    assert "always()" in job["if"]


def test_quality_workflow_fails_closed_when_mutation_does_not_pass() -> None:
    workflow = yaml.load(
        (PROJECT_ROOT / ".github/workflows/quality.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    job = workflow["jobs"]["quality"]
    guard = job["steps"][0]
    assert "always()" in guard["if"]
    assert "needs.changed-mutation.result != 'success'" in guard["if"]
    assert "exit 1" in guard["run"]


def test_test_only_changes_run_the_full_mutation_sweep() -> None:
    script = _changed_mutation_script()
    assert '"FULL_MUTATION"' in script
    assert "--max-no-tests 69" in script


def _changed_mutation_script() -> str:
    workflow = yaml.load(
        (PROJECT_ROOT / ".github/workflows/quality.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    return next(
        step["run"]
        for step in workflow["jobs"]["changed-mutation"]["steps"]
        if step.get("name") == "Mutate changed package modules"
    )


def test_changed_mutation_job_has_a_bounded_timeout() -> None:
    workflow = yaml.load(
        (PROJECT_ROOT / ".github/workflows/quality.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    assert workflow["jobs"]["changed-mutation"]["timeout-minutes"] == "240"


def test_changed_mutation_runs_serially_to_avoid_pytest_temp_races() -> None:
    assert _changed_mutation_script().count("mutmut run --max-children 1") == 2


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


def _duckdb_spatial_steps() -> tuple[
    list[dict[str, Any]], dict[str, Any], dict[str, Any]
]:
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
    return steps, install_step, mutation_step


def test_changed_mutation_job_installs_duckdb_spatial_extension() -> None:
    _, install_step, _ = _duckdb_spatial_steps()
    assert "install_extension('spatial')" in install_step["run"]


def test_spatial_extension_installs_before_mutation() -> None:
    steps, install_step, mutation_step = _duckdb_spatial_steps()
    assert steps.index(install_step) < steps.index(mutation_step)


def test_quality_gate_installs_spatial_before_its_test_suite() -> None:
    steps = _job_steps("quality.yml", "quality")
    install_step = _named_step(steps, "Install DuckDB spatial extension")
    quality_step = _named_step(steps, "Run the complete deterministic quality gauntlet")

    assert (
        "install_extension('spatial')" in install_step["run"],
        steps.index(install_step) < steps.index(quality_step),
    ) == (True, True)


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

    assert "quality_gauntlet.py --skip-docker --skip-mutation" in quality_step["run"]


def test_deptry_is_declared_in_lint_dependencies() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as pyproject_file:
        project = tomllib.load(pyproject_file)
    lint_dependencies = {
        _package_name(dependency)
        for dependency in project["dependency-groups"]["lint"]
        if isinstance(dependency, str)
    }
    assert "deptry" in lint_dependencies


def test_deptry_runs_in_the_lightweight_lint_job() -> None:
    workflow = yaml.safe_load(
        (PROJECT_ROOT / ".github/workflows/lint.yml").read_text(encoding="utf-8")
    )
    ruff_steps = workflow["jobs"]["ruff"]["steps"]
    commands = [step.get("run", "") for step in ruff_steps]
    assert any("deptry ." in command for command in commands)


def _deptry_ignored_rules() -> dict[str, list[str]]:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as pyproject_file:
        project = tomllib.load(pyproject_file)
    return project["tool"]["deptry"]["per_rule_ignores"]


def test_deptry_ignores_only_documented_runtime_dependencies() -> None:
    ignored = _deptry_ignored_rules()
    assert set(ignored["DEP002"]) == {
        "accelerate",
        "peft",
        "protobuf",
        "python-multipart",
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
        "numpy",
        "plotly",
        "radon",
        "toml",
    }


def test_deptry_ignored_rules_are_documented_in_pyproject() -> None:
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


def _benchmark_guard_step() -> dict[str, Any]:
    workflow = yaml.safe_load(
        (PROJECT_ROOT / ".github/workflows/benchmark.yml").read_text(encoding="utf-8")
    )
    return next(
        step
        for step in workflow["jobs"]["compare"]["steps"]
        if step.get("name") == "Run algorithmic guards"
    )


def test_benchmark_workflow_disables_timing_for_algorithmic_guards() -> None:
    assert "--benchmark-disable" in _benchmark_guard_step()["run"]


def test_benchmark_workflow_runs_algorithmic_guard_cases() -> None:
    assert "test_guards.py" in _benchmark_guard_step()["run"]


def _benchmark_base_step() -> tuple[dict[str, Any], dict[str, Any]]:
    workflow = yaml.load(
        (PROJECT_ROOT / ".github/workflows/benchmark.yml").read_text(encoding="utf-8"),
        Loader=_UniqueKeyLoader,
    )
    triggers = workflow.get("on", workflow.get(True))
    base_step = next(
        step
        for step in workflow["jobs"]["compare"]["steps"]
        if step.get("name") == "Benchmark pull request base"
    )
    return triggers["workflow_dispatch"], base_step


def test_benchmark_base_dispatch_defaults_to_main() -> None:
    dispatch, _ = _benchmark_base_step()
    assert dispatch["inputs"]["base_ref"]["default"] == "main"


def test_benchmark_base_checkout_uses_the_requested_ref() -> None:
    _, base_step = _benchmark_base_step()
    assert '"$BASE_REF"' in base_step["run"]


def test_benchmark_base_syncs_its_locked_environment() -> None:
    _, base_step = _benchmark_base_step()
    assert "uv sync --locked --project .tmp/main" in base_step["run"]
    assert (
        'uv pip install --python .tmp/main/.venv/bin/python "pytest-benchmark==5.3.0"'
        in base_step["run"]
    )


def test_benchmark_base_uses_its_own_python() -> None:
    _, base_step = _benchmark_base_step()
    assert "$GITHUB_WORKSPACE/.tmp/main/.venv/bin/python" in base_step["run"]


def test_coverage_preview_runs_after_successful_pull_request_tests() -> None:
    workflow = yaml.load(
        (PROJECT_ROOT / ".github/workflows/coverage-preview.yml").read_text(
            encoding="utf-8"
        ),
        Loader=_UniqueKeyLoader,
    )
    triggers = workflow.get("on", workflow.get(True))
    upload = workflow["jobs"]["upload"]
    assert triggers["workflow_run"]["workflows"] == ["Tests"]
    assert "github.event.workflow_run.event == 'pull_request'" in upload["if"]
    assert "github.event.workflow_run.conclusion == 'success'" in upload["if"]


def _coverage_download_step() -> dict[str, Any]:
    workflow = yaml.load(
        (PROJECT_ROOT / ".github/workflows/coverage-preview.yml").read_text(
            encoding="utf-8"
        ),
        Loader=_UniqueKeyLoader,
    )
    upload = workflow["jobs"]["upload"]
    return next(
        step
        for step in upload["steps"]
        if step.get("uses", "").startswith("actions/download-artifact@")
    )


def test_coverage_preview_selects_the_triggering_run_artifact() -> None:
    download = _coverage_download_step()
    assert download["with"]["name"] == "coverage-html"
    assert download["with"]["run-id"] == "${{ github.event.workflow_run.id }}"


def test_coverage_preview_download_uses_scoped_token_and_path() -> None:
    download = _coverage_download_step()
    assert download["with"]["github-token"] == "${{ github.token }}"
    assert download["with"]["path"] == "coverage-html"


def test_coverage_preview_validates_report_and_skips_checkout() -> None:
    workflow = yaml.load(
        (PROJECT_ROOT / ".github/workflows/coverage-preview.yml").read_text(
            encoding="utf-8"
        ),
        Loader=_UniqueKeyLoader,
    )
    steps = workflow["jobs"]["upload"]["steps"]
    validate = next(
        step
        for step in steps
        if step.get("name") == "Validate coverage report artifact"
    )
    assert validate["run"].strip() == "test -s coverage-html/index.html"
    assert not any(
        step.get("uses", "").startswith("actions/checkout@") for step in steps
    )


def test_smokeshow_requires_explicit_credential_rotation() -> None:
    workflow = yaml.load(
        (PROJECT_ROOT / ".github/workflows/coverage-preview.yml").read_text(
            encoding="utf-8"
        ),
        Loader=_UniqueKeyLoader,
    )
    steps = workflow["jobs"]["upload"]["steps"]
    upload = next(
        step for step in steps if step.get("name") == "Publish coverage preview"
    )

    assert upload["if"] == "vars.GEOPARSER_SMOKESHOW_AUTH_ROTATION_CONFIRMED == 'true'"


def test_smokeshow_secret_is_supplied_only_as_an_environment_variable() -> None:
    workflow = yaml.load(
        (PROJECT_ROOT / ".github/workflows/coverage-preview.yml").read_text(
            encoding="utf-8"
        ),
        Loader=_UniqueKeyLoader,
    )
    steps = workflow["jobs"]["upload"]["steps"]
    upload = next(
        step for step in steps if step.get("name") == "Publish coverage preview"
    )
    assert upload["env"]["SMOKESHOW_AUTH_KEY"] == "${{ secrets.SMOKESHOW_AUTH_KEY }}"


def _coverage_publish_step() -> dict[str, Any]:
    workflow = yaml.load(
        (PROJECT_ROOT / ".github/workflows/coverage-preview.yml").read_text(
            encoding="utf-8"
        ),
        Loader=_UniqueKeyLoader,
    )
    steps = workflow["jobs"]["upload"]["steps"]
    return next(
        step for step in steps if step.get("name") == "Publish coverage preview"
    )


def _coverage_preview_commands() -> str:
    workflow = yaml.load(
        (PROJECT_ROOT / ".github/workflows/coverage-preview.yml").read_text(
            encoding="utf-8"
        ),
        Loader=_UniqueKeyLoader,
    )
    steps = workflow["jobs"]["upload"]["steps"]
    return "\n".join(step.get("run", "") for step in steps)


def test_smokeshow_upload_runs_with_shell_tracing_disabled() -> None:
    assert _coverage_publish_step()["run"].splitlines() == [
        "set +x",
        "smokeshow upload coverage-html",
    ]


def test_smokeshow_upload_commands_do_not_reveal_the_credential() -> None:
    commands = _coverage_preview_commands()
    assert "SMOKESHOW_AUTH_KEY" not in commands
    assert "printenv" not in commands
    assert "set -x" not in commands
