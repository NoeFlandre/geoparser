import uuid
from collections.abc import Iterable
from typing import Any, Generic, TypeVar

from sqlalchemy import insert
from sqlmodel import Session, SQLModel, select

T = TypeVar("T", bound=SQLModel)


def _default_batch_id(model: type[SQLModel]) -> uuid.UUID | None:
    id_field = model.model_fields.get("id")
    if id_field is None or id_field.default_factory is None:
        return None
    return uuid.uuid4()


def _batch_row(
    model: type[SQLModel], obj: SQLModel, column_names: set[str]
) -> dict[str, Any]:
    row = {key: value for key, value in obj.model_dump().items() if key in column_names}
    if "id" not in row:
        generated_id = _default_batch_id(model)
        if generated_id is not None:
            row["id"] = generated_id
    return row


def _batch_rows(
    model: type[SQLModel], objects: Iterable[SQLModel], column_names: set[str]
) -> tuple[list[dict[str, Any]], list[uuid.UUID | str]]:
    rows: list[dict[str, Any]] = []
    identifiers: list[uuid.UUID | str] = []
    for obj in objects:
        row = _batch_row(model, obj, column_names)
        identifiers.append(row["id"])
        rows.append(row)
    return rows, identifiers


def _insert_batch(db: Session, table: Any, rows: list[dict[str, Any]]) -> None:
    try:
        db.execute(insert(table), rows)  # ty: ignore[deprecated]
        db.commit()
    except Exception:
        db.rollback()
        raise


class BaseRepository(Generic[T]):
    """
    Base repository with common CRUD operations for all models.
    """

    # Set by every concrete repository. There is no meaningful default, and
    # all nine subclasses assign it, so it is declared without one.
    model: type[T]

    @classmethod
    def create(cls, db: Session, obj_in: SQLModel) -> T:
        """
        Create a new record.

        Args:
            db: Database session
            obj_in: Object to create (typically a Create model)

        Returns:
            Created object
        """
        # Convert input to model instance using the model's data
        data = obj_in.model_dump()
        db_obj = cls.model(**data)

        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    @classmethod
    def create_many(
        cls, db: Session, objects: Iterable[SQLModel]
    ) -> list[uuid.UUID | str]:
        """Create and commit a batch without one transaction per row."""
        table = cls.model.__table__  # ty: ignore[unresolved-attribute]
        column_names = set(table.columns.keys())
        rows, identifiers = _batch_rows(cls.model, objects, column_names)

        if not rows:
            return []

        _insert_batch(db, table, rows)
        return identifiers

    @classmethod
    def get(cls, db: Session, id: uuid.UUID | str) -> T | None:
        """
        Get a record by ID.

        Args:
            db: Database session
            id: Record ID (UUID or string)

        Returns:
            Record if found, None otherwise
        """
        # SQLModel columns are typed as their instance value (uuid.UUID), but
        # at class level they are SQLAlchemy column expressions carrying
        # .id/.in_/.desc. No type checker models this duality without a
        # SQLAlchemy plugin, so the access below is suppressed narrowly.
        statement = select(cls.model).where(cls.model.id == id)  # ty: ignore[unresolved-attribute]
        return db.exec(statement).unique().first()

    @classmethod
    def get_all(cls, db: Session) -> list[T]:
        """
        Get all records.

        Args:
            db: Database session

        Returns:
            List of all records
        """
        statement = select(cls.model)
        return list(db.exec(statement).unique().all())

    @classmethod
    def update(cls, db: Session, *, db_obj: T, obj_in: SQLModel) -> T:
        """
        Update a record.

        Args:
            db: Database session
            db_obj: Existing database object
            obj_in: New data to update with

        Returns:
            Updated object
        """
        update_data = obj_in.model_dump(exclude_unset=True)
        for field in update_data:
            if hasattr(db_obj, field):
                setattr(db_obj, field, update_data[field])
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    @classmethod
    def delete(cls, db: Session, *, id: uuid.UUID | str) -> T | None:
        """
        Delete a record.

        Args:
            db: Database session
            id: Record ID (UUID or string)

        Returns:
            Deleted object if found, None otherwise
        """
        obj = cls.get(db, id)
        if obj:
            db.delete(obj)
            db.commit()
        return obj
