# Contributing

Thank you for your contribution to Irchel Geoparser. For product use, read the [documentation](https://docs.geoparser.app). This file describes local development and the checks that run in CI.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. Install uv. Then run these commands from the repository root:

```bash
uv sync --locked
uv run pre-commit install
```

The first command makes `.venv/`. It installs the runtime dependencies and the development dependencies at the versions in `uv.lock`. This includes the spaCy models that the tests use. It installs geoparser in editable mode. The second command installs the pre-commit hooks of the repository. uv downloads a suitable interpreter automatically. You do not need a separate Python installation.

Run the tools through uv:

```bash
uv run <command>
```

To activate the environment in your current shell, use this command:

```bash
source .venv/bin/activate
```

The supported Python versions are `>=3.10,<3.15`. Keep your changes compatible with all versions in that range.

## Code style

[Ruff](https://docs.astral.sh/ruff/) does the formatting, the import sorting, the removal of unused code, and the linting. It replaces the former tools black, isort, and autoflake. `pyproject.toml` pins its version and `uv.lock` locks it. Thus CI and your machine format the code in the same way.

```bash
uv run ruff check --fix .
uv run ruff format .
```

CI runs the checking form of both commands. CI fails if the code is not formatted or not clean:

```bash
uv run ruff check .
uv run ruff format --check .
```

`[tool.ruff.lint]` in `pyproject.toml` declares the enabled rule sets. Fix a finding. Do not silence it. A `# noqa` needs a specific code and a reason.

## Tests

The tests are in `tests/`. They have this structure:

- `tests/unit/` has fast and isolated tests (usually mocked).
- `tests/integration/` tests real components together (models, DB, gazetteers).
- `tests/e2e/` has full pipeline tests.

The markers are declared under `[tool.pytest.ini_options]` in `pyproject.toml`, the single source of the pytest configuration: `unit`, `integration`, `e2e`, `property`, `acceptance`, `architecture`, and `benchmark`. A test in `tests/unit/`, `tests/integration/`, `tests/e2e/`, `tests/property/`, or `tests/acceptance/` receives the marker named after its directory automatically, so `pytest -m <marker>` selects every test in that directory. `architecture` and `benchmark` are declared explicitly on the tests that use them.

Hypothesis uses the registered profiles `dev`, `ci`, and `nightly`. Local runs use `dev` by default. Set `HYPOTHESIS_PROFILE=ci` for deterministic examples of CI size. Set `HYPOTHESIS_PROFILE=nightly` for the larger randomized run. The workflows select `ci` for pull requests. They select `nightly` for the scheduled quality run.

Two integration files are opt-in. They skip with an explicit reason unless you set `GEOPARSER_TEST_REMOTE_MODELS=1`:

```bash
GEOPARSER_TEST_REMOTE_MODELS=1 uv run pytest tests/integration/test_recognizers/test_gliner_recognizer_integration.py tests/integration/test_resolvers/test_jina_resolver_integration.py
```

These tests use `GLiNER2Recognizer` and `JinaResolver` with the real checkpoints. Together the checkpoints are several gigabytes. Run the tests when you change either module. They are the only tests that can find these problems: a zero-shot label that the model does not respond to, a prompt name that the checkpoint does not define, or a change in the return shape of the reranker. A mock hides all of these problems. A local `uv run pytest` with the default settings still skips these files. CI enables them in the Ubuntu and Python 3.12 coverage cell. CI also enables them in the single-run quality gauntlet. Thus both strict CRAP reports include their assertions. The other cells of the operating-system and Python matrix keep the downloads disabled. The first run on a new runner can download several gigabytes of model files.

To run the full suite, use this command:

```bash
uv run pytest
```

The tests collect coverage for `geoparser/`, `scripts/`, and `tests/`. Thus the CRAP gate can score each function. The quality gauntlet also appends the coverage from the deterministic benchmark contracts with timing disabled. CI combines the coverage across its matrix. It enforces a hard floor of 100% line coverage on `geoparser/`, including `geoparser/annotator/`. The HTML report is in `htmlcov/`. Open `htmlcov/index.html`. To check the same package floor locally with the real model tests enabled, run these commands:

```bash
GEOPARSER_TEST_REMOTE_MODELS=1 uv run pytest
uv run coverage report --include='geoparser/*' --fail-under=100
```

In a new environment, the real-model tests can download several gigabytes of checkpoints. The quality gauntlet collects the main suite and the deterministic benchmark contracts. It verifies this package floor. Then it runs the strict whole-tree CRAP check.

We keep the suite fast on purpose. Two points are important when you add tests:

- **Do not import the heavy stack at module scope in `geoparser/cli/` or in a package `__init__`.** The CLI resolves spaCy, torch, and the build pipeline for each command. For this reason `python -m geoparser --help` takes a fraction of a second. Before, it took about 9 s. A test pins this.
- The tests measure coverage with the `sys.monitoring` core (`core = "sysmon"`). It costs less than 1%, unlike the overhead of the tracer. It falls back automatically on Python 3.10 and 3.11.

We measured `pytest-xdist` and did not adopt it. The start-up of the workers is much longer than the run of 638 fast unit tests. The tests were three times slower. On the full suite, the wall clock changed by less than 2% and the CPU use increased by about 80%.

These subsets are useful:

```bash
uv run pytest tests/unit
uv run pytest tests/integration/test_geoparser_integration.py
```

## Quality checks

One command reproduces the deterministic quality gate that CI enforces. Run it from the repository root before you open a pull request:

```bash
uv sync --locked
GEOPARSER_TEST_REMOTE_MODELS=1 uv run python scripts/quality_gauntlet.py
```

The gate removes its temporary reports, its mutation tree, and its Docker image for each run. If the system volume is small, use the [resource-safe local gate](docs/development.md#resource-safe-local-gate) command before you run it.

For fast feedback during TDD, run one stage directly:

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check geoparser scripts tests
uv run pytest
uv run mkdocs build --strict --site-dir /tmp/geoparser-site
```

This list shows what each step guards:

- **ruff check / ruff format**: lint, import order, unused code, and formatting.
- **[ty](https://github.com/astral-sh/ty)**: static type checking of the configured source tree. Fix the type error. Do not add a blanket `# type: ignore`. If a suppression is correct, make it specific and add a comment with the reason.
- **pytest / coverage report**: the unit tests, integration tests, end-to-end tests, and deterministic benchmark contract tests record coverage for all CRAP roots. `coverage report --include='geoparser/*' --fail-under=100` enforces the package floor.
- **scripts/crap.py**: the [CRAP score](https://testing.googleblog.com/2011/02/this-code-is-crap.html) gate. The formula is `complexity² × (1 − coverage)³ + complexity` for each function. For fully covered code, this is a ceiling on the cyclomatic complexity. Thus it fails on code with no tests. It also fails on code with too many branches. It reads the coverage data that pytest just wrote. Run it after both test passes.
- **[mutmut](https://mutmut.readthedocs.io/)**: mutation testing. It changes the source in small ways and runs the tests again. A mutant that survives is a line that the suite does not check. The configuration is under `[tool.mutmut]` in `pyproject.toml`. `scripts/mutation_gate.py` reads the exported statistics. It fails when more mutants survive than the agreed baseline.

Mutation testing targets selected package modules with the **unit** suite. [MUTATION_TESTING.md](./MUTATION_TESTING.md) lists the scope and the exclusions. The latest recorded fit-coverage campaign generated 3,568 mutants. The tests killed 3,424. 62 survived. 69 had no covering unit test. 13 timed out. This is a historical snapshot. It is not a verified count for the current tree. The allowance for mutants with no test stays at 69. The limit for survivors stays at zero. A mutant with no covering test does not count as killed. The quality gauntlet and CI use both limits.

We measured the judgment of mutants with the integration suite also, and we rejected it. It is more thorough. Every `no tests` mutant disappears and the survival falls from 29% to about 11%. But each mutant that it reaches builds a real gazetteer again. This takes about 23 seconds for each mutant and about thirteen hours for the package. The integration suites, the e2e suites, and the 100% coverage gate cover the build pipeline instead. To do the thorough run, add `"tests/integration"` to `pytest_add_cli_args_test_selection`. Reserve an evening.

The older clean sweep recorded one timeout and 212 mutants with no covering unit test. Those figures are older than the later fit-coverage campaign. They are not the current allowance. They do not show that the current tree passes the mutation gates.

To inspect the survivors, use these commands:

```bash
uv run mutmut results
uv run mutmut show <id>
```

To fix a surviving mutant, normally make a test stronger. Do not delete the mutant.

[MUTATION_TESTING.md](./MUTATION_TESTING.md) tracks the scope decisions, the current numbers, and the work that is not complete. Keep it up to date when you work on it.

## Documentation

The user documentation is in Markdown sources in `docs/`. MkDocs Material publishes it through GitHub Pages. After `uv sync --locked`, build the documentation locally with this command:

```bash
uv run mkdocs build --strict --site-dir /tmp/geoparser-site
```

Open `/tmp/geoparser-site/index.html` in a browser. When you change the public APIs or the behavior, update the applicable guides or API pages in `docs/`.

Write the documentation in ASD-STE100 Simplified Technical English. Use short sentences, active voice, and imperative steps. Use the terms in the [glossary](docs/glossary.md).

## CLI

The package CLI is available with this command:

```bash
uv run python -m geoparser --help
```

Common commands are gazetteer `install`, `list`, and `uninstall`. Another command starts the annotator.

## Branches and pull requests

`main` is the only long-lived branch. Work on feature branches that you cut from `main`. Send the work back through a pull request. The repository rejects direct pushes to `main`.

To contribute code, open a pull request. We also welcome issues. Use them for questions, support, bug reports, or to discuss an idea before you start.

These tips make reviews easier:

- Run the local check sequence below before you submit.
- Add or update tests when the behavior changes.
- Update the documentation when the user-facing behavior changes.

CI runs on pull requests into `main` and on `main` itself. It never runs on pushes to feature branches. The matrix has three operating systems and Python 3.10 to 3.14. uv supplies the interpreter on all of them. A new push to an open pull request cancels the previous run.

Four workflows run: **Lint** (Ruff), **Tests** (the platform matrix, combined coverage, and CRAP), **Quality** (the ordered gauntlet), and **Documentation** (strict MkDocs and GitHub Pages). On pull requests, the quality gauntlet skips the Docker builds and the full mutation sweep. The separate changed-mutation job checks the changed package code. The scheduled quality run keeps the full mutation stages and the Docker stages. The stable required contexts for the `main` ruleset are `tests-passed`, `ruff`, `build`, and `quality-gate`. The [CI safety and merge protection guide](docs/guides/ci-safety.md) gives their workflow mapping and the reasons.

If you add a dependency, commit the updated `uv.lock` with `pyproject.toml`. (`uv add <package>` updates both.) Use packages with permissive licenses. Geoparser has the MIT license.

## Releasing

This section is for maintainers. Tags control the releases. The tag name is the version. Tags have no `v` prefix. There is no release branch.

Set the version with `uv version <version>` in the last pull request of the cycle. Set the final version also when release candidates come first. After the merge, tag from `main`:

```bash
git switch main && git pull
VERSION=$(uv version --short)
git tag "${VERSION}rc1" && git push origin "${VERSION}rc1"
```

Examine the candidate in a clean environment. The option `--pre` is necessary because resolvers hide pre-releases:

```bash
python -m venv /tmp/rc
/tmp/rc/bin/pip install --pre "geoparser==${VERSION}rc1"
/tmp/rc/bin/python -c "from importlib.metadata import version; print(version('geoparser'))"
```

Then tag the release. You do not need a second version change:

```bash
git tag "$VERSION" && git push origin "$VERSION"
```

If the candidate needs fixes, merge them through a pull request. Then tag `${VERSION}rc2`.

### What each tag produces

| Tag | PyPI | GitHub Release | GitHub Pages |
| --- | --- | --- | --- |
| `1.4.0rc1` | pre-release, needs `--pre` | none | preview build |
| `1.4.0` | release | created, Sigstore-signed | published from `main` |

You can also publish a final release from the GitHub web UI. Do this when you want to write the notes by hand. This creates the tag and starts the same workflow. The workflow attaches the signed artifacts to the release. Only publishing creates the tag. A saved draft does not create it.

### If a tag publishes nothing

- The tag must point to a commit on `main`.
- The tag must agree with `pyproject.toml`. Ignore an `rc` suffix. If the file has `1.4.0`, the workflow accepts `1.4.0` and `1.4.0rc2`. It rejects `1.4.1`, `1.5.0`, and `1.5.0rc1`.
- The tag must have the form `MAJOR.MINOR.PATCH` or `MAJOR.MINOR.PATCHrcN`. Other forms start no workflow. Examples are `v1.4.0`, `1.4`, or `1.4.0-rc1`. Therefore there is no failed run to inspect.

## Licensing

This project has the MIT license. Read [LICENSE](./LICENSE). The project declares its dependencies. It does not bundle them. Each maintainer distributes their dependency under its own license.

## Pull-request validation lifecycle

These events run the full required checks with stable job names: code pushes, opened or reopened pull requests, promotion from draft, and edits to open pull requests. When retargeting a PR, you change its base branch. The PR rebuilds the new base comparison. Metadata edits also run the validation again. This costs another CI run on purpose. The alternative is to publish skipped suites that hide the actual results. Make all title and description edits before the final validation. We do not remove or relax any branch-protection requirement. Manual dispatch is still available for explicit diagnostics.

Push events and pull-request events have separate concurrency groups. An edit to a closed or merged PR skips the validation. It cannot cancel a main-branch run that is in progress.
