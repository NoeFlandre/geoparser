# CI artifact safety and merge protection

This page records the CI contracts for the multilingual pilot. It also records the merge policy that we intend for `main`.

## Stable checks for `main`

The reference policy is in `.github/branch-protection/main.json`. It names one stable check for each workflow:

| Required check | Workflow | What it covers |
| --- | --- | --- |
| `tests-passed` | Tests | The OS and Python test matrix, the combined 100% coverage, and the CRAP limit. |
| `ruff` | Lint | The Ruff lint and format checks. |
| `build` | Documentation | The strict MkDocs build. |
| `quality-gate` | Quality | The deterministic quality workflow. On pull requests, it includes the changed-module mutation gate. |

The names of the matrix jobs include the operating system and the Python version. If you require those individual names, branch protection depends on the details of the matrix. Therefore the aggregate job `tests-passed` is the required test check. The scheduled Quality run does the full mutation gates and the Docker gates.

The repository ruleset `Protect main with required checks` is active. It targets only `refs/heads/main`. It requires pull requests, an up-to-date branch, and the four contexts above. It enforces the rule for administrators. It has an empty bypass list. The JSON file is the reference under version control. Use it to create the ruleset again. After you change a workflow, compare the exact status contexts with a recent pull request. Then update the ruleset.

## Coverage artifact flow

The `Coverage preview` workflow listens for a completed `Tests` workflow. It continues only when that run came from a pull request and succeeded. It uses `actions/download-artifact` v4, pinned to `d3f86a106a0bac45b974a628896c90dbdf5c8093`. It gets the `coverage-html` artifact with the `workflow_run.id` of the triggering run. It extracts the artifact under `coverage-html/`. It checks that `coverage-html/index.html` is not empty before any report step can run.

The installation and publication of Smokeshow need the repository variable `GEOPARSER_SMOKESHOW_AUTH_ROTATION_CONFIRMED` set to `true`. The owner must set that variable only after the owner rotates the Smokeshow key and reviews the affected historical logs. The scope of the authentication key is the publish step only. The workflow disables shell tracing before the upload.

To verify a later pull request, inspect the job summaries and the step summaries of the `Coverage preview` run. The download of the artifact for the exact run and the report validation must succeed. While the rotation variable is not set, the Smokeshow steps must be skipped. Use the step metadata for this check. Do not get unfiltered logs. Do not publish credential material.
