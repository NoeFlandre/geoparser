import json
import typing as t
import uuid

from fastapi.encoders import jsonable_encoder
from pydantic import ValidationError
from sqlmodel import Session as DBSession

from geoparser.annotator.db.crud.base import BaseRepository
from geoparser.annotator.db.crud.document import DocumentRepository
from geoparser.annotator.db.crud.settings import SessionSettingsRepository
from geoparser.annotator.db.models.document import AnnotatorDocumentCreate
from geoparser.annotator.db.models.session import (
    AnnotatorSession,
    AnnotatorSessionCreate,
    AnnotatorSessionDownload,
    AnnotatorSessionUpdate,
)
from geoparser.annotator.exceptions import (
    InvalidUploadException,
    SessionNotFoundException,
)


class SessionRepository(BaseRepository[AnnotatorSession]):
    model = AnnotatorSession
    exception_factory: t.Callable = lambda x, y: SessionNotFoundException(
        f"{x} with ID {y} not found."
    )

    @classmethod
    # BaseRepository declares the widest input type (SQLModel); each repository
    # deliberately accepts its own Create/Update model. Callers always go
    # through the concrete repository, so the precise signature is worth more
    # here than strict substitutability.
    def create(  # ty: ignore[invalid-method-override]
        cls,
        db: DBSession,
        item: AnnotatorSessionCreate,
        exclude: list[str] | None = None,
        additional: dict[str, t.Any] | None = None,
    ) -> AnnotatorSession:
        # Create the main session object
        session = super().create(
            db,
            item,
            exclude=["settings", "documents", *(exclude or [])],
            additional=additional,
        )
        # Create settings if provided
        if item.settings:
            SessionSettingsRepository.create(
                db, item.settings, additional={"session_id": session.id}
            )
        # Create documents if provided
        if item.documents:
            for document in item.documents:
                DocumentRepository.create(
                    db, document, additional={"session_id": session.id}
                )
        # session object expired when creating settings and documents so we refresh it
        db.refresh(session)
        return session

    @classmethod
    def create_from_json(
        cls,
        db: DBSession,
        json_str: str,
        keep_id: bool = False,  # noqa: FBT001, FBT002 - positional bool kept for API compatibility; make keyword-only in the next major release
    ) -> AnnotatorSession:
        try:
            content = cls._parse_json_content(json_str)
            session = cls._session_create_from_import(content)
            additional = cls._session_import_additional(content, keep_id=keep_id)
        except (
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
            ValidationError,
        ) as error:
            msg = "Invalid session JSON: required fields are missing or malformed."
            raise InvalidUploadException(msg) from error

        return cls.create(db, session, additional=additional)

    @staticmethod
    def _parse_json_content(json_str: str) -> dict[str, t.Any]:
        """Load and check the outer structure of a serialized session."""
        content = json.loads(json_str)
        if not isinstance(content, dict):
            msg = "session JSON must contain an object"
            raise TypeError(msg)
        if not isinstance(content.get("documents"), list):
            msg = "session JSON must contain a documents list"
            raise TypeError(msg)
        return content

    @classmethod
    def _document_create_from_import(
        cls, document_dict: dict[str, t.Any]
    ) -> AnnotatorDocumentCreate:
        """Validate one imported document and its toponym children."""
        if not isinstance(document_dict, dict):
            # pragma: no mutate start - TypeError text is not behavior; tests pin
            # the exception type and invalid-document handling.
            msg = "each imported document must be an object"
            raise TypeError(msg)
            # pragma: no mutate end
        # An export always lists its toponyms, even when there are none.
        if not isinstance(document_dict.get("toponyms"), list):
            # pragma: no mutate start - TypeError text is not behavior; tests pin
            # the exception type and malformed-toponyms handling.
            msg = "each imported document must contain a toponyms list"
            raise TypeError(msg)
            # pragma: no mutate end
        return AnnotatorDocumentCreate.model_validate(
            {**document_dict, "spacy_applied": True}
        )

    @classmethod
    def _session_create_from_import(
        cls, content: dict[str, t.Any]
    ) -> AnnotatorSessionCreate:
        """Validate a session and materialize its nested document inputs."""
        documents = [
            cls._document_create_from_import(document_dict)
            for document_dict in content["documents"]
        ]
        return AnnotatorSessionCreate.model_validate(
            {**content, "documents": documents}
        )

    @staticmethod
    def _session_import_additional(
        content: dict[str, t.Any], *, keep_id: bool
    ) -> dict[str, uuid.UUID]:
        """Return imported database fields not represented in the create model."""
        additional = {}
        if keep_id and (session_id := content.get("session_id")):
            additional["id"] = uuid.UUID(session_id)
        return additional

    @classmethod
    def read(cls, db: DBSession, id: uuid.UUID) -> AnnotatorSession:
        return super().read(db, id)

    @classmethod
    def read_to_json(cls, db: DBSession, id: uuid.UUID) -> dict:
        item = cls.read(db, id)
        result = AnnotatorSessionDownload(
            **item.model_dump(),
            documents=[
                AnnotatorDocumentCreate(
                    **document.model_dump(), toponyms=document.toponyms
                )
                for document in item.documents
            ],
        )
        return jsonable_encoder(result)

    @classmethod
    def read_all(cls, db: DBSession, **filters) -> list[AnnotatorSession]:
        return super().read_all(db, **filters)

    @classmethod
    def update(cls, db: DBSession, item: AnnotatorSessionUpdate) -> AnnotatorSession:  # ty: ignore[invalid-method-override]
        return super().update(db, item)

    @classmethod
    def delete(cls, db: DBSession, id: uuid.UUID) -> AnnotatorSession:
        return super().delete(db, id)
