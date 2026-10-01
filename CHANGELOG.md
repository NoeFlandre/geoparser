# Changelog

All notable changes to GeoParser are recorded here. This project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases use
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed

- Keep required CI check names stable and avoid replacing validation with
  skipped suites when pull-request metadata changes.
- Exclude metric bookkeeping from PAN-X prediction timing and preserve malformed
  gold-tag counts in micro aggregates. Mark unreproducible historical sample
  evidence explicitly, and bind diagnostic mutant IDs to their source checkout.

- Reject missing sentence-transformer tokenizers with a clear error, and test
  resolver input alignment without accepting unrelated exceptions.
- Remove redundant resolver validation and record exact mutation-replay evidence.

### Security

- Update the locked JupyterLab development dependency to 4.6.4 to address
  CVE-2026-102830, CVE-2026-102831, and CVE-2026-102904.
- Update the locked Notebook demo dependency to 7.6.3 to address
  PYSEC-2026-4112.

## [0.6.0]

### Added

- Add the `geoparser` command, version output, and a `parse` workflow for text
  files, standard input, JSONL, JSON, and GeoJSON.
- Add command options for annotator hosting and gazetteer installation, plus
  clear errors for unknown gazetteers and failed installs.
- Add population-prior and sentence-transformer resolver choices and document
  benchmark pipelines, corpora, and reproducibility workflows.

### Changed

- Use a population-only `PriorResolver` by default, with weight `0.3` and
  inflection fallback disabled.
- Support Python 3.10 through 3.14 and constrain Transformers and spaCy to
  compatible releases.
- Resolve the application data directory through platformdirs while preserving
  existing platform paths.

### Fixed

- Cache gazetteer access and fit-time location data, remove redundant resolver
  caches, and bound retained cache state.
- Share recognition and resolution service setup, require precomputed
  similarities, and remove unreachable service code.
- Narrow annotator database error handling and defer model logging changes
  until a transformer model is loaded.
- Validate annotation inputs and report malformed UTF-8 uploads without
  interrupting legacy batch imports.
