# Data and models

The source repository contains code, small deterministic fixtures, and public configuration examples. Large gazetteers and model checkpoints are runtime artifacts. You download them explicitly. The library caches them outside the repository. Nobody commits them silently.

## Gazetteers

To install a pre-configured gazetteer, use the CLI. For example, use `geoparser install geonames`. The [custom gazetteers guide](guides/custom-gazetteers.md) describes custom gazetteers. The result is a self-contained SQLite file. The library stores it in the platform data directory.

## Model checkpoints

The built-in modules get their declared checkpoints from the upstream model registries. The exact model and the configuration are part of the identity of a module. If you change one of them, you get a separately identifiable result set. The unit tests do not need model downloads. The integration tests must use a fixture or an explicitly provisioned cache.

## Hugging Face provenance

Hugging Face is an upstream registry for model checkpoints. The project can also use it for project-owned datasets or models when a release needs it. Before you upload, record these items in the release documentation: the repository identifier, the revision, the license, the schema, and the generation command.

This repository has no project-owned artifact that is authorized for automatic upload. Therefore the quality gate does not publish data files or model files.

## Reproducibility

Use locked dependencies, deterministic fixtures, explicit schemas, golden outputs, and seeded replay. Version large external resources by their upstream revision or checksum. Do not copy them into source control.
