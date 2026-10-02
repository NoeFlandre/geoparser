# ADR 0001: Quality gates and MkDocs

## Status

Accepted — 2026-09-11

## Context

The project needs one reproducible developer workflow. It also needs public documentation that CI can build. The previous documentation stack used Sphinx with configuration for a specific theme. The quality checks were in several commands. They did not always enforce the architecture results or the mutation results.

## Decision

Use `uv` and `uv.lock` for reproducible dependencies. Use Ruff and `ty` for static quality. Use pytest, Hypothesis, and pytest-bdd for behavior. Use an AST-based architecture check for the dependency boundaries. Use a single ordered quality gauntlet for local verification and CI verification. Move the public site to MkDocs Material. Generate the API reference pages with mkdocstrings. Keep large data and model artifacts outside the repository. Document the provenance before any publication on Hugging Face.

## Tradeoffs

MkDocs has a smaller configuration surface. Its Markdown authoring workflow is simpler. But it does not have the full cross-reference system of Sphinx. The project accepts explicit Markdown links and focused API directives. In exchange, local builds and public hosting are easier. Mutation checks and CRAP checks add runtime. They run as explicit gates. They keep generated artifacts outside the source tree.

## Consequences

A strict MkDocs build must check the documentation links and the API directives. A new architectural exception needs a deliberate update of the checker and the tests. The quality gate is the canonical command for acceptance. The individual commands are still available for fast feedback during TDD.
