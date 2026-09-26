# Database API

The database-backed repositories are supported for applications that need
lower-level access to persisted projects and results. Most applications should
start with [`Project`](project.md) or [`Geoparser`](geoparser.md); use these
interfaces when you need repository queries or direct SQL access.

Repository methods receive a SQLModel `Session`. Open sessions with
`get_session()` and let its context manager close them after use.

::: geoparser.db.db.get_session

## Connections

Use `get_connection()` for statements that need a SQLAlchemy connection
directly. The context manager closes the connection on exit.

::: geoparser.db.db.get_connection

## Common repository operations

Every concrete repository inherits these operations from `BaseRepository`:

::: geoparser.db.crud.base.BaseRepository
    options:
      members:
        - create
        - create_many
        - get
        - get_all
        - update
        - delete

## Recognition queries

::: geoparser.db.crud.recognition.RecognitionRepository
    options:
      members:
        - get_by_document
        - get_by_recognizer
        - get_by_document_and_recognizer
        - get_processed_document_ids
        - get_unprocessed_documents

## Resolution queries

::: geoparser.db.crud.resolution.ResolutionRepository
    options:
      members:
        - get_by_reference
        - get_by_resolver
        - get_by_reference_and_resolver
        - get_processed_reference_ids
        - get_unprocessed_references

## Reference queries

::: geoparser.db.crud.reference.ReferenceRepository
    options:
      members:
        - get_by_document
        - get_by_document_and_span
