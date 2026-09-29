# CI artifact safety and merge protection

This page records the CI contracts for the multilingual pilot and the
intended merge policy for `main`.

## Stable checks for `main`

The reference policy is stored in
`.github/branch-protection/main.json`. It names one stable check for each
workflow:

| Required check | Workflow | What it covers |
| --- | --- | --- |
| `tests-passed` | Tests | The OS/Python test matrix, combined 100% coverage, and the CRAP limit. |
| `ruff` | Lint | Ruff lint and format checks. |
| `build` | Documentation | The strict MkDocs build. |
| `quality-gate` | Quality | The deterministic quality workflow, including the changed-module mutation gate on pull requests. |

The matrix job names include operating system and Python version. Requiring
those individual names would couple branch protection to matrix details, so
the aggregate `tests-passed` job is the required test check. The scheduled
Quality run performs the full mutation and Docker gates.

The repository ruleset `Protect main with required checks` is active and
targets only `refs/heads/main`. It requires pull requests, an up-to-date branch,
and the four contexts above; it enforces the rule for administrators and has
an empty bypass list. The JSON file is the version-controlled reference for
recreating the ruleset. After workflow changes, compare the exact status
contexts against a recent pull request before updating the ruleset.

## Coverage artifact flow

The `Coverage preview` workflow listens for a completed `Tests` workflow. It
proceeds only when that run came from a pull request and succeeded. It uses the
`actions/download-artifact` v4, pinned to
`d3f86a106a0bac45b974a628896c90dbdf5c8093`, fetches the `coverage-html`
artifact by the triggering `workflow_run.id`, extracts it under
`coverage-html/`, and checks that `coverage-html/index.html` is non-empty
before any report step can run.

Smokeshow installation and publication require the repository variable
`GEOPARSER_SMOKESHOW_AUTH_ROTATION_CONFIRMED` to be `true`. The owner should set
that variable only after rotating the Smokeshow key and reviewing the affected
historical logs. The authentication key is scoped to the publish step, and
shell tracing is disabled before the upload.

To verify a later pull request, inspect the `Coverage preview` run's job and
step summaries: the exact-run artifact download and report validation should
succeed. While the rotation variable is unset, the Smokeshow steps should be
skipped. Use step metadata for this check; do not fetch unfiltered logs or
publish credential material.
