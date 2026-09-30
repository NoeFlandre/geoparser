import json
from pathlib import Path

import yaml

from tests.unit.test_quality.mkdocs_navigation import markdown_paths

ROOT = Path(__file__).resolve().parents[3]
EXPECTED_CHECKS = (
    ("tests-passed", ".github/workflows/test.yml", "tests-passed"),
    ("ruff", ".github/workflows/lint.yml", "ruff"),
    ("build", ".github/workflows/docs.yml", "build"),
    ("quality-gate", ".github/workflows/quality.yml", "quality"),
)


def test_ci_safety_guide_is_in_the_mkdocs_navigation() -> None:
    configuration = yaml.safe_load((ROOT / "mkdocs.yml").read_text())
    assert "guides/ci-safety.md" in markdown_paths(configuration["nav"])


def test_contributing_documents_all_stable_required_checks() -> None:
    contributing = (ROOT / "CONTRIBUTING.md").read_text()
    for context, _, _ in EXPECTED_CHECKS:
        assert f"`{context}`" in contributing


def _branch_policy() -> dict:
    return json.loads((ROOT / ".github/branch-protection/main.json").read_text())


def test_branch_policy_protects_current_main() -> None:
    policy = _branch_policy()
    assert policy["branch"] == "main"
    assert policy["require_pull_request"] is True
    assert policy["require_up_to_date_branch"] is True


def test_branch_policy_requires_the_expected_status_checks() -> None:
    policy = _branch_policy()
    assert (
        tuple(
            (item["context"], item["workflow"], item["job"])
            for item in policy["required_status_checks"]
        )
        == EXPECTED_CHECKS
    )


def test_required_workflow_jobs_match_branch_policy() -> None:
    for context, workflow_path, job_id in EXPECTED_CHECKS:
        workflow = yaml.safe_load((ROOT / workflow_path).read_text())
        job = workflow["jobs"][job_id]
        assert context in str(job.get("name", job_id))


def test_ci_safety_guide_lists_the_required_checks() -> None:
    guide = (ROOT / "docs/guides/ci-safety.md").read_text()
    for context, _, _ in EXPECTED_CHECKS:
        assert f"`{context}`" in guide


def test_ci_safety_guide_documents_the_active_main_ruleset() -> None:
    guide = (ROOT / "docs/guides/ci-safety.md").read_text()
    assert "`Protect main with required checks` is active" in guide
    assert "targets only `refs/heads/main`" in guide
    assert "enforces the rule for administrators" in guide
    assert "empty bypass list" in guide
