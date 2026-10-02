# Architecture

The package has a small and explicit pipeline:

```text
text → recognizer → reference spans → resolver → gazetteer features
```

`Geoparser` is the stateless entry point. `Project` adds persistence. It also lets you compare result sets again and again. Recognizers and resolvers implement independent interfaces. Services connect these interfaces to the database.

## Dependency boundaries

The import direction has one way only:

```text
modules → (no database or project imports)
services → modules, database
project → services, context, database
gazetteer → artifact and build internals
```

The architecture checker is an executable script: [`scripts/check_architecture.py`](https://github.com/NoeFlandre/geoparser/blob/main/scripts/check_architecture.py). It rejects forbidden layer imports and import cycles. It ignores `TYPE_CHECKING` imports because they do not cause runtime coupling.

## Pure domain modules

Some modules hold decision logic. This logic must not know what produced its inputs. `PURE_MODULES` in the same checker lists these modules. The checker fails if one of them imports anything other than the standard library. `geoparser.modules.resolvers.context` is the current example. It decides how much text around a reference fits in the token budget of an encoder.

The purpose is testability. Tidiness is not the purpose. To size a context, the code does arithmetic on sentence costs. Before, this logic was in the resolver. You could reach it only through a tokenizer, a sentence splitter, and an embedding model. To test it, you had to build all three as mocks. Now a small value object holds the logic. You can test it directly, also with generated inputs. Refer to `tests/property/test_context_invariants.py`.

## Side effects

Model inference, network access, filesystem access, and database writes stay at the edges of the system. You can test the pure matching helpers and the module contracts without a database or a downloaded model. The integration tests and the acceptance tests use deterministic fixtures. They test the boundaries that must work together.

## Extension point

To add a recognizer or a resolver, implement the applicable base interface. Keep the configuration in the module identity. Do not reach through the database layer. The [modules guide](guides/modules.md) shows the complete contract and examples.
