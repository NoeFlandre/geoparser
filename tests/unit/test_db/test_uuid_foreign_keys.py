"""
Behavioural tests for UUID foreign keys on SQLite.

Primary keys store UUIDs as CHAR(32) text. Foreign keys that reference them must
store the same text. A foreign-key column declared with the generic ``UUID`` type
gets NUMERIC affinity on SQLite, which turns an all-digit UUID such as
``12345678-1234-4234-8234-123456789012`` into a number and breaks the match.
"""

import uuid

import pytest
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, col, select, text

from geoparser.db.models import Document, Project, Recognizer, Reference

NUMERIC_LOOKING_UUID = uuid.UUID("12345678-1234-4234-8234-123456789012")
LEADING_ZERO_UUID = uuid.UUID("00000000-0000-4000-8000-000000000001")
ORDINARY_HEX_UUID = uuid.UUID("3f2a9c1e-7b4d-4e8a-9c2f-1a6b5d8e0f47")


def _seed_reference(session: Session, identifier: uuid.UUID) -> None:
    """Insert a project, document and reference that all use ``identifier``."""
    session.add(Recognizer(id="recognizer", name="recognizer", config={}))
    session.add(Project(id=identifier, name="project"))
    session.add(Document(id=identifier, project_id=identifier, text="Paris"))
    session.add(
        Reference(
            id=identifier,
            document_id=identifier,
            recognizer_id="recognizer",
            start=0,
            end=5,
            text="Paris",
        )
    )
    session.commit()


@pytest.mark.unit
class TestUuidForeignKeyRoundTrip:
    """UUID foreign-key values keep their text form and still match their keys."""

    @pytest.mark.parametrize(
        "identifier",
        [NUMERIC_LOOKING_UUID, LEADING_ZERO_UUID, ORDINARY_HEX_UUID],
        ids=["numeric-looking", "leading-zeros", "ordinary-hex"],
    )
    def test_foreign_key_round_trips_unchanged(self, test_session, identifier):
        """A reference reads back with the same document, project and id."""
        _seed_reference(test_session, identifier)
        test_session.expire_all()

        reference = test_session.get(Reference, identifier)
        assert reference is not None
        assert reference.id == identifier
        assert reference.document_id == identifier
        assert reference.document.id == identifier
        assert reference.document.project_id == identifier

    @pytest.mark.parametrize(
        "identifier",
        [NUMERIC_LOOKING_UUID, LEADING_ZERO_UUID, ORDINARY_HEX_UUID],
        ids=["numeric-looking", "leading-zeros", "ordinary-hex"],
    )
    def test_foreign_key_is_stored_as_the_key_text(self, test_session, identifier):
        """The stored foreign-key value is text identical to the primary key."""
        _seed_reference(test_session, identifier)

        row = test_session.exec(
            text(
                "SELECT typeof(document_id), document_id, typeof(id), id FROM reference"
            )
        ).one()

        assert row == ("text", identifier.hex, "text", identifier.hex)

    def test_join_matches_numeric_looking_uuid(self, test_session):
        """A join between a foreign key and its primary key finds the row."""
        _seed_reference(test_session, NUMERIC_LOOKING_UUID)

        joined = test_session.exec(
            select(col(Reference.id), col(Document.id))
            .join(Document, col(Reference.document_id) == col(Document.id))
            .where(col(Document.project_id) == NUMERIC_LOOKING_UUID)
        ).all()

        assert joined == [(NUMERIC_LOOKING_UUID, NUMERIC_LOOKING_UUID)]

    def test_foreign_key_is_still_enforced(self, test_session):
        """A reference to a missing document is still rejected."""
        _seed_reference(test_session, NUMERIC_LOOKING_UUID)

        test_session.add(
            Reference(
                id=uuid.uuid4(),
                document_id=uuid.uuid4(),
                recognizer_id="recognizer",
                start=0,
                end=1,
                text="x",
            )
        )
        with pytest.raises(IntegrityError):
            test_session.commit()
