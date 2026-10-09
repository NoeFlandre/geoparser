# Changelog

This file records all notable changes to GeoParser. The project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Releases use
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Add an offline MultiCoNER II place-recognition adapter, pinned to revision
  `4be2d62c912977ee26ed14d2553a4fe17ca3d980` under CC BY 4.0. It keeps the twelve
  dataset languages that are in the 85-code inventory, scores exact spans with
  the shared protocol contract, and reports every invalid record. It runs no
  model and downloads no data. Clean and noisy breakdowns are not supported by
  the release.

### Fixed

- Show a clear deprecation notice in `geoparser download --help`. It names the
  replacement `install` command and states that `download` does not download
  anything. The command still exits with status 1.

- Store project database foreign keys to UUID primary keys as text, as the
  keys themselves are. A numeric-looking UUID such as
  `12345678-1234-4234-8234-123456789012` no longer breaks its foreign-key match
  on SQLite. This applies to new databases only. Existing database files keep
  their column types until a separate migration is approved.

- Give each benchmark database session an independent connection to a
  temporary SQLite file. This matches the transaction isolation of production.
  Sessions no longer share one in-memory connection.

- Isolate CI event concurrency. An edit to a merged pull request can no longer
  cancel the main-branch checks. Skip the redundant validation for closed pull
  requests.

- Keep the names of the required CI checks stable. Do not replace validation
  with skipped suites when the pull-request metadata changes.
- Exclude the metric bookkeeping from the PAN-X prediction timing. Keep the
  count of malformed gold tags in the micro aggregates. Mark the historical
  sample evidence that you cannot reproduce. Bind the diagnostic mutant IDs to
  their source checkout.

- Reject a missing sentence-transformer tokenizer with a clear error. Test the
  resolver input alignment without accepting unrelated exceptions.
- Remove the redundant resolver validation. Record the exact mutation-replay
  evidence.

### Security

- Validate gazetteer names before constructing artifact paths. Reject path
  separators and invalid names in the CLI and Python API. Uninstall cannot
  remove an artifact outside the gazetteers directory through its name.

- Update the locked JupyterLab development dependency to 4.6.4. This fixes
  CVE-2026-102830, CVE-2026-102831, and CVE-2026-102904.
- Update the locked Notebook demo dependency to 7.6.3. This fixes
  PYSEC-2026-4112.

## [0.6.0]

### Added

- Add the `geoparser` command, the version output, and a `parse` workflow. The
  workflow accepts text files, standard input, JSONL, JSON, and GeoJSON.
- Add command options to host the annotator and to install gazetteers. Add
  clear errors for unknown gazetteers and failed installations.
- Add the population-prior resolver and the sentence-transformer resolver.
  Document the benchmark pipelines, the corpora, and the reproducibility
  workflows.

### Changed

- Use a population-only `PriorResolver` by default. The weight is `0.3`. The
  inflection fallback is disabled.
- Support Python 3.10 through 3.14. Limit Transformers and spaCy to compatible
  releases.
- Find the application data directory with platformdirs. Keep the existing
  platform paths.

### Fixed

- Cache the gazetteer access and the fit-time location data. Remove the
  redundant resolver caches. Limit the size of the retained cache state.
- Share the setup of the recognition service and the resolution service.
  Require precomputed similarities. Remove the unreachable service code.
- Narrow the error handling of the annotator database. Delay the changes to
  model logging until a transformer model is loaded.
- Validate the annotation inputs. Report malformed UTF-8 uploads. Do not
  interrupt the legacy batch imports.
