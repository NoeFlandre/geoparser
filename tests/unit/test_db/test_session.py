"""Transaction and cleanup contracts for the shared session context."""

import pytest
from sqlalchemy import event
from sqlmodel import Session, select

import geoparser.db.db as db
from geoparser.db.models import Project


@pytest.fixture
def session_lifecycle(monkeypatch):
    events = []

    class ObservedSession(Session):
        def close(self):
            events.append(("close", self.in_transaction()))
            super().close()

    event.listen(ObservedSession, "after_rollback", lambda _: events.append("rollback"))
    monkeypatch.setattr(db, "Session", ObservedSession)
    return events


@pytest.mark.unit
@pytest.mark.parametrize(
    "error_type", [RuntimeError, KeyboardInterrupt, SystemExit, GeneratorExit]
)
def test_rolls_back_before_close_and_preserves_the_error(error_type, session_lifecycle):
    error = error_type("interrupted transaction")
    with db.get_session() as session:
        session.add(Project(name="committed"))
        session.commit()
    session_lifecycle.clear()

    with pytest.raises(error_type) as raised, db.get_session() as session:
        session.add(Project(name="uncommitted"))
        session.flush()
        assert session.exec(select(Project.name).order_by(Project.name)).all() == [
            "committed",
            "uncommitted",
        ]
        raise error

    assert raised.value is error
    assert session_lifecycle == ["rollback", ("close", False)]
    with db.get_session() as session:
        assert session.exec(select(Project.name)).all() == ["committed"]


@pytest.mark.unit
def test_success_keeps_explicit_commits_and_detached_values(session_lifecycle):
    with db.get_session() as session:
        project = Project(name="committed")
        session.add(project)
        session.commit()

    assert session_lifecycle == [("close", False)]
    assert project.name == "committed"
    with db.get_session() as session:
        assert session.exec(select(Project.name)).all() == ["committed"]


@pytest.mark.unit
def test_success_does_not_commit_pending_changes(session_lifecycle):
    with db.get_session() as session:
        session.add(Project(name="committed"))
        session.commit()
        session.add(Project(name="uncommitted"))
        session.flush()
        assert session.exec(select(Project.name).order_by(Project.name)).all() == [
            "committed",
            "uncommitted",
        ]

    assert session_lifecycle == [("close", True)]
    with db.get_session() as session:
        assert session.exec(select(Project.name)).all() == ["committed"]


@pytest.mark.unit
def test_closes_even_when_rollback_fails(monkeypatch, session_lifecycle):
    error = RuntimeError("transaction failed")
    rollback_error = RuntimeError("rollback failed")

    def fail_rollback():
        raise rollback_error

    with pytest.raises(RuntimeError) as raised, db.get_session() as session:
        session.add(Project(name="uncommitted"))
        session.flush()
        monkeypatch.setattr(session, "rollback", fail_rollback)
        raise error

    assert raised.value is rollback_error
    assert raised.value.__context__ is error
    assert session_lifecycle == [("close", True)]
    assert not session.in_transaction()
    assert len(session.identity_map) == 0
