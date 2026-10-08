# GitHub Actions status for 72fb776 (as of the push, 2026-10-08 about 16:10 UTC)

Triggered by the push that created PR #182 (head `issue-141-cli-parsers` at 72fb776).
These runs are GitHub-side. They are not cancelled by this handoff. Their results after this
time are not covered by this document. Check the run links.

| Workflow | Run | Status at last check |
|---|---|---|
| Tests | 37806683133 | queued |
| Lint | 37806682957 | queued |
| (security/CodeQL/dependency jobs) | 37806683426 | queued |
| (changed-mutation / exact-mutant replay) | 37806683189 | queued (exact-mutant replay skipped) |
| (compare) | 37806683324 | queued |
| (build) | 37806683311 | queued |
| (docker smoke) | 37806683021 | queued |

Required job list at the last check (20 check runs): pytest on ubuntu 3.10, 3.11, 3.12, 3.13, 3.14; windows 3.10, 3.14; macos 3.10, 3.14; ruff; ty; workflow-lint; codeql (python, actions); dependency-audit; build; docker-smoke; compare; changed-mutation; exact-mutant-replay (skipped).

Links: https://github.com/NoeFlandre/geoparser/actions/runs/37806683133 and the other run IDs above, at https://github.com/NoeFlandre/geoparser/actions/runs/<id>.
