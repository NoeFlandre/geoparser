"""Contract tests for the docs deployment, the release workflow, benchmarks, and coverage preview."""

from typing import Any

import yaml

from tests.conftest import PROJECT_ROOT
from tests.unit.test_quality.project_contract_support import UniqueKeyLoader


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
        Loader=UniqueKeyLoader,
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
        Loader=UniqueKeyLoader,
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
        Loader=UniqueKeyLoader,
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
        Loader=UniqueKeyLoader,
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
        Loader=UniqueKeyLoader,
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
        Loader=UniqueKeyLoader,
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
        Loader=UniqueKeyLoader,
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
        Loader=UniqueKeyLoader,
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
