# Reviews (issue #141 handoff)

All reviews were independent and read-only: the reviewer had no write access to the checkout and did not author the change. Reviewers were subagents. Verdicts are tied to the SHA each review covered.

| # | Reviewed SHA (range) | Scope | Verdict | Findings and disposition |
|---|---|---|---|---|
| R1 | e19b95b..3f9f6a0 (earlier state; scripts identical to 72fb776) | behaviour: scripts diff line by line, harness, file scope, excluded paths, adversarial probes (prog, description, default Path, repeated `main()`, import-time state) | approve | no findings. Harness: 42 cases identical to baseline. Scope: only the three files. |
| R2 | 3f9f6a0 (earlier state) | tests and gates: vacuity, golden strings, gate weakening, CRAP decision points | approve | minor only: `refuse()` guard does not cover `is_file`, `is_dir`, `stat`, `glob`, `write_text`, `os.listdir`; `allow_abbrev` untested; `from argparse import ArgumentParser` evades the construction check (since fixed); fail-open missing path pinned on purpose. |
| R3 | 7aa4a9d | final-commit review | approve | minor: module-mode check was a substring match; bare-import guard; colour under FORCE_COLOR (all since fixed) |
| R4 | d109524 | review of the fix for R3 | approve | one minor: module-mode check still substring-based, so `python -m scripts.changelog_old` and `my_changelog.py` would pass. Fixed in 36ac09b. |
| R5 | 36ac09b (delta over d109524) | does the final assertion close R4 without weakening tests | changes_requested | Gap closed. One line was loosened: the literal `startswith("usage: ")` check was dropped. Fixed in 72fb776. |
| R6 | 72fb776 (delta over 36ac09b) | re-review of the fix for R5 | approve | none. Startswith check restored. Probes matched; targeted pytest 33 passed; ruff format clean. |

Remote review sessions (earlier, created in this session, now archived and completed; they
reviewed earlier states): one on e498ed6 found no breakage (minor items: audit hook, annotations,
an unverified release-candidate mapping); one on 0d24c5b found no behaviour change (minor
test and typing nits); one on ca29d05 found no blockers (two nits: verify program name on 3.14, which was then done and fixed; one nit about a function that no longer exists).

Reviews do not cover CI on GitHub, the Windows, macOS, Python 3.10 cells, or the combined-coverage CRAP run.
