"""Helpers shared by the recognition and resolution services."""

from collections.abc import Callable
from typing import Protocol

from sqlmodel import Session, SQLModel

from geoparser.db.db import get_session


class _Module(Protocol):
    @property
    def id(self) -> str: ...

    @property
    def name(self) -> str: ...

    @property
    def config(self) -> dict: ...


class _NamedModule(Protocol):
    @property
    def name(self) -> str: ...


class _Record(Protocol):
    id: str


class _Repository(Protocol):
    @classmethod
    def get(cls, db: Session, id: str) -> _Record | None: ...

    @classmethod
    def create(cls, db: Session, obj_in: SQLModel) -> _Record: ...


def ensure_module_record(
    repository: type[_Repository],
    create_model: type[SQLModel],
    module: _Module,
) -> str:
    """Ensure a recognizer or resolver row exists and return its ID."""
    with get_session() as session:
        record = repository.get(session, id=module.id)
        if record is None:
            create = create_model(id=module.id, name=module.name, config=module.config)
            record = repository.create(session, create)
        return record.id


def require_fit(module: _NamedModule, kind: str) -> Callable[..., None]:
    """Return a module's fit method or raise a consistent service error."""
    fit = getattr(module, "fit", None)
    if fit is None:
        raise ValueError(f"{kind} '{module.name}' does not implement a fit method")
    return fit
