"""Tests for annotator database setup and session lifecycle helpers."""

from collections.abc import Generator
from typing import Any, cast
from unittest.mock import Mock

from sqlalchemy import inspect
from sqlmodel import create_engine

from geoparser.annotator.db import db as annotator_db
from geoparser.annotator.db.models import AnnotatorDocument, AnnotatorSession


def test_enable_foreign_keys_executes_sqlite_pragma():
    """Each new annotator connection enables SQLite foreign key checks."""
    connection = Mock()

    annotator_db.enable_foreign_keys(connection, None)

    connection.cursor.return_value.execute.assert_called_once_with(
        "PRAGMA foreign_keys=ON"
    )
    connection.cursor.return_value.close.assert_called_once_with()


def test_create_db_and_tables_creates_annotator_schema():
    """The database initializer creates the registered SQLModel tables."""
    engine = create_engine("sqlite://")
    try:
        annotator_db.create_db_and_tables(engine)

        tables = set(inspect(engine).get_table_names())
        expected_tables = {
            AnnotatorSession.__tablename__,
            AnnotatorDocument.__tablename__,
        }
        assert expected_tables <= tables
    finally:
        engine.dispose()


def test_get_db_closes_session_when_generator_finishes(monkeypatch):
    """The request dependency always closes its yielded database session."""
    fake_session = Mock()
    monkeypatch.setattr(annotator_db, "Session", lambda engine: fake_session)
    dependency = cast(Generator[Any, None, None], annotator_db.get_db())

    assert next(dependency) is fake_session
    dependency.close()

    fake_session.close.assert_called_once_with()
