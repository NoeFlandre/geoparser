"""Persist UUID relationships without SQLite numeric-affinity conversions."""

import uuid

import pytest
from sqlalchemy import delete, select
from sqlalchemy.dialects import mssql, mysql, postgresql, sqlite
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, SQLModel

from geoparser.db.models import (
    Context,
    Document,
    Recognition,
    Reference,
    Referent,
    Resolution,
)
from tests.unit.test_db.test_db import (
    _ForeignKeySeed,
    _seed_document_table,
    _seed_lookup_tables,
    _seed_reference_table,
)

pytestmark = pytest.mark.unit

_PROJECT = uuid.UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
_DOCUMENT = uuid.UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
_REFERENCE = uuid.UUID("cccccccc-cccc-4ccc-8ccc-cccccccccccc")
_CHILD = uuid.UUID("dddddddd-dddd-4ddd-8ddd-dddddddddddd")
_RELATIONSHIPS = (
    (Document, "project_id", "project"),
    (Context, "project_id", "project"),
    (Reference, "document_id", "document"),
    (Recognition, "document_id", "document"),
    (Referent, "reference_id", "reference"),
    (Resolution, "reference_id", "reference"),
)
_UUID_CASES = (
    "12345678123442348234123456789012",
    "12345678123442348234123456789013",
    "00000000000000000000000000000001",
    "000000000000000000000000000001e2",
    "abcdefabcdef4abc8defabcdefabcdef",
)


def _insert(connection, table, **values):
    connection.execute(SQLModel.metadata.tables[table].insert(), values)


def _parent_row(connection, parent, identifier):
    """Seed only the ancestors required by the relationship under test."""
    if parent == "project":
        _insert(connection, "project", id=identifier, name="Parent")
        return
    _insert(connection, "project", id=_PROJECT, name="Parent")
    document_id = identifier if parent == "document" else _DOCUMENT
    _insert(connection, "document", id=document_id, project_id=_PROJECT, text="Paris")
    if parent == "reference":
        _insert(
            connection,
            "reference",
            id=identifier,
            document_id=_DOCUMENT,
            recognizer_id="recognizer",
            start=0,
            end=5,
            text="Paris",
        )


def _child_row(column, identifier):
    # Extra keys are filtered before insertion, so each row uses one literal
    # oracle without coupling the test to the model's defaults.
    return {
        "id": _CHILD,
        column: identifier,
        "text": "Paris",
        "start": 0,
        "end": 5,
        "tag": "Test",
        "recognizer_id": "recognizer",
        "resolver_id": "resolver",
        "gazetteer_name": "test",
        "feature_identifier": "Paris",
    }


def _seed_relationship(connection, parent, identifier):
    _insert(connection, "recognizer", id="recognizer", name="Test", config={})
    _insert(connection, "resolver", id="resolver", name="Test", config={})
    _parent_row(connection, parent, identifier)


def _insert_child(connection, table, column, identifier):
    values = _child_row(column, identifier)
    connection.execute(
        table.insert(), {key: value for key, value in values.items() if key in table.c}
    )


def _assert_stored_relationship(connection, table, column, parent, hex_value):
    stored = connection.exec_driver_sql(
        f'SELECT "{column}", typeof("{column}") FROM "{table.name}"'
    ).one()
    assert tuple(stored) == (hex_value, "text")
    parent_table = SQLModel.metadata.tables[parent]
    joined = connection.execute(
        select(parent_table.c.id).join(table, parent_table.c.id == table.c[column])
    ).all()
    assert joined == [(uuid.UUID(hex=hex_value),)]
    assert connection.exec_driver_sql("PRAGMA foreign_key_check").all() == []


@pytest.mark.parametrize("model,column,parent", _RELATIONSHIPS)
@pytest.mark.parametrize("hex_value", _UUID_CASES)
def test_uuid_relationship_round_trips(test_engine, model, column, parent, hex_value):
    identifier = uuid.UUID(hex=hex_value)
    table = SQLModel.metadata.tables[model.__tablename__]
    with test_engine.begin() as connection:
        assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar_one() == 1
        _seed_relationship(connection, parent, identifier)
        _insert_child(connection, table, column, identifier)

    # A new session ensures the relationship is loaded from the stored values.
    with Session(test_engine) as session:
        child = session.get(model, _CHILD)
        assert getattr(child, column) == identifier
        assert getattr(child, parent).id == identifier

    with test_engine.connect() as connection:
        _assert_stored_relationship(connection, table, column, parent, hex_value)


@pytest.mark.parametrize("model,column,parent", _RELATIONSHIPS)
def test_uuid_foreign_keys_still_reject_missing_parents(
    test_engine, model, column, parent
):
    table = SQLModel.metadata.tables[model.__tablename__]
    with test_engine.begin() as connection:
        _seed_relationship(connection, parent, _REFERENCE)
        with pytest.raises(IntegrityError, match="FOREIGN KEY constraint failed"):
            _insert_child(
                connection, table, column, uuid.UUID("12345678123442348234123456789012")
            )
        assert connection.execute(select(table.c.id)).all() == []
        assert connection.execute(
            select(SQLModel.metadata.tables[parent].c.id)
        ).all() == [(_REFERENCE,)]


def test_adjacent_numeric_uuids_remain_distinct_and_cascade_selectively(test_engine):
    project = SQLModel.metadata.tables["project"]
    document = SQLModel.metadata.tables["document"]
    first = uuid.UUID("12345678123442348234123456789012")
    second = uuid.UUID("12345678123442348234123456789013")
    with test_engine.begin() as connection:
        _insert(connection, "project", id=first, name="First")
        _insert(connection, "project", id=second, name="Second")
        _insert(connection, "document", id=_DOCUMENT, project_id=first, text="First")
        _insert(connection, "document", id=_CHILD, project_id=second, text="Second")
        rows = connection.execute(
            select(document.c.text, project.c.name)
            .join(project)
            .order_by(document.c.text)
        ).all()
        assert rows == [("First", "First"), ("Second", "Second")]
        connection.execute(delete(project).where(project.c.id == first))
        assert connection.execute(
            select(document.c.id, document.c.project_id)
        ).all() == [(_CHILD, second)]
        assert connection.exec_driver_sql("PRAGMA foreign_key_check").all() == []


def test_original_reference_fixture_accepts_numeric_document_uuid(test_engine):
    """Pin the exact insertion path which failed on Otter PR #172."""
    identifier = uuid.UUID("12345678-1234-4234-8234-123456789012")
    ids = _ForeignKeySeed(
        project_ids=[_PROJECT],
        document_ids=[identifier],
        reference_ids=[_REFERENCE],
        referent_ids=[],
        resolution_ids=[],
        recognition_ids=[],
        recognizer_ids=["recognizer"],
        resolver_ids=["resolver"],
    )
    with test_engine.begin() as connection:
        _seed_lookup_tables(connection, ids)
        _seed_document_table(connection, ids)
        _seed_reference_table(connection, ids)
        assert connection.exec_driver_sql(
            "SELECT document_id, typeof(document_id) FROM reference"
        ).all() == [("12345678123442348234123456789012", "text")]


@pytest.mark.parametrize("model,column,parent", _RELATIONSHIPS)
@pytest.mark.parametrize(
    "dialect,expected",
    [
        (sqlite.dialect(), "CHAR(32)"),
        (postgresql.dialect(), "UUID"),
        (mysql.dialect(), "CHAR(32)"),
        (mssql.dialect(), "UNIQUEIDENTIFIER"),
    ],
)
def test_uuid_foreign_keys_compile_like_parent_keys(
    model, column, parent, dialect, expected
):
    """Check emitted backend DDL rather than a Python type identity."""
    foreign_key = SQLModel.metadata.tables[model.__tablename__].c[column]
    primary_key = SQLModel.metadata.tables[parent].c.id
    assert str(foreign_key.type.compile(dialect=dialect)) == expected
    assert str(primary_key.type.compile(dialect=dialect)) == expected
