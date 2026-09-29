import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
EXPECTED_CHECKS = (
    ("tests-passed", ".github/workflows/test.yml", "tests-passed"),
    ("ruff", ".github/workflows/lint.yml", "ruff"),
    ("build", ".github/workflows/docs.yml", "build"),
    ("quality-gate", ".github/workflows/quality.yml", "quality"),
)


def _markdown_paths(items: list[object]) -> list[str]:
    paths: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        for value in item.values():
            if isinstance(value, str) and value.endswith(".md"):
                paths.append(value)
            elif isinstance(value, list):
                paths.extend(_markdown_paths(value))
    return paths


def test_ci_safety_guide_is_in_the_mkdocs_navigation() -> None:
    configuration = yaml.safe_load((ROOT / "mkdocs.yml").read_text())
    assert "guides/ci-safety.md" in _markdown_paths(configuration["nav"])


def test_contributing_documents_all_stable_required_checks() -> None:
    contributing = (ROOT / "CONTRIBUTING.md").read_text()
    for context, _, _ in EXPECTED_CHECKS:
        assert f"`{context}`" in contributing


def test_documented_policy_matches_stable_workflow_jobs() -> None:
    policy = json.loads((ROOT / ".github/branch-protection/main.json").read_text())
    assert policy["branch"] == "main"
    assert policy["require_pull_request"] is True
    assert policy["require_up_to_date_branch"] is True
    assert (
        tuple(
            (item["context"], item["workflow"], item["job"])
            for item in policy["required_status_checks"]
        )
        == EXPECTED_CHECKS
    )

    for context, workflow_path, job_id in EXPECTED_CHECKS:
        workflow = yaml.safe_load((ROOT / workflow_path).read_text())
        job = workflow["jobs"][job_id]
        assert context in str(job.get("name", job_id))

    guide = (ROOT / "docs/guides/ci-safety.md").read_text()
    for context, _, _ in EXPECTED_CHECKS:
        assert f"`{context}`" in guide


def test_ci_safety_guide_documents_the_active_main_ruleset() -> None:
    guide = (ROOT / "docs/guides/ci-safety.md").read_text()
    assert "`Protect main with required checks` is active" in guide
    assert "targets only `refs/heads/main`" in guide
    assert "enforces the rule for administrators" in guide
    assert "empty bypass list" in guide
