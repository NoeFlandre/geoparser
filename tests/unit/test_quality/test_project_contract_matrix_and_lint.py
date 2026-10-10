"""Contract tests for the test matrix, the quality workflow's model and Docker switches, and lint."""

import yaml

from tests.conftest import PROJECT_ROOT
from tests.unit.test_quality.project_contract_support import (
    job_steps,
    load_pyproject,
    named_step,
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
    pytest_step = named_step(job_steps("test.yml", "pytest"), "Run pytest")

    assert pytest_step["env"]["GEOPARSER_TEST_REMOTE_MODELS"] == (
        "${{ matrix.os == 'ubuntu-latest' && matrix.python-version == '3.12' && '1' || '' }}"
    )


def test_quality_gate_runs_remote_model_tests_for_complete_crap_coverage() -> None:
    quality_step = named_step(
        job_steps("quality.yml", "quality"),
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


def test_deptry_runs_in_the_lightweight_lint_job() -> None:
    workflow = yaml.safe_load(
        (PROJECT_ROOT / ".github/workflows/lint.yml").read_text(encoding="utf-8")
    )
    ruff_steps = workflow["jobs"]["ruff"]["steps"]
    commands = [step.get("run", "") for step in ruff_steps]
    assert any("deptry ." in command for command in commands)


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
    project = load_pyproject()
    return next(
        requirement
        for requirement in project["dependency-groups"]["lint"]
        if requirement.startswith("ruff==")
    )


def test_precommit_ruff_revision_matches_the_locked_lint_dependency() -> None:
    assert _ruff_lint_dependency() == f"ruff=={_ruff_precommit_revision()}"


def test_precommit_is_available_after_the_documented_sync() -> None:
    project = load_pyproject()

    lint_dependencies = project["dependency-groups"]["lint"]
    assert any(
        dependency.startswith("pre-commit")
        for dependency in lint_dependencies
        if isinstance(dependency, str)
    )
