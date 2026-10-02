# Technical debt

This page lists the known limits. We do not hide them behind quality metrics. Each item has a reason and a cleanup direction.

## Database migration support

The database schema is not stable between releases. There is no automatic migration. To upgrade, you can need to export the results and create the local database again.

The cleanup path is a versioned migration layer. It will have upgrade tests that use fixtures. We will add it before the project reaches 1.0.

## Model and gazetteer availability

Some integration paths need large third-party model checkpoints or gazetteers. The package does not include them. The reasons are size, licensing, and update frequency.

The cleanup path is a documented and versioned fixture and cache contract for CI. We will also add a small offline test artifact for each supported module family.

## Annotator boundary

The annotator is a standalone tool. It has its own database and an import and export boundary. It is older than the current project architecture. We can replace it.

The cleanup path is to make the JSON interchange schema stable first. Then we decide if we integrate the user interface or remove it.

## Public API evolution

The package version is below 1.0. A minor release can include breaking changes.

The cleanup path is to use the API reference and the acceptance scenarios as the compatibility contract. Then we will add explicit deprecation periods when the API is stable.

## Mutation coverage scope

Mutation testing judges the unit-test surface only. Integration gates, end-to-end gates, or coverage gates cover the gazetteer build mutants and the server-rendered annotator. It is too expensive now to mutate them with full fixtures.

The cleanup path is to add small deterministic mutation fixtures for those boundaries. Then we will widen the mutation selection. The quality gate must not become an unbounded run. [`MUTATION_TESTING.md`](https://github.com/NoeFlandre/geoparser/blob/main/MUTATION_TESTING.md) tracks the current scope and counts.
