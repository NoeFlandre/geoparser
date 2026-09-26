"""Session creation, import, continuation and deletion routes."""

import typing as t
import uuid

from fastapi import APIRouter, Depends, Form, Request, UploadFile, status
from fastapi.responses import RedirectResponse
from sqlmodel import Session as DBSession

from geoparser.annotator.db.crud import DocumentRepository, SessionRepository
from geoparser.annotator.db.db import db_location, get_db
from geoparser.annotator.db.models import AnnotatorSession, AnnotatorSessionCreate
from geoparser.annotator.dependencies import get_session
from geoparser.annotator.exceptions import (
    InvalidUploadException,
    SessionNotFoundException,
)
from geoparser.annotator.models.api import BaseResponse, LegacyFilesResponse

router = APIRouter()


@router.post("/session", tags=["session"])
def create_session(
    request: Request,
    files: list[UploadFile],
    gazetteer: t.Annotated[str, Form()],
    spacy_model: t.Annotated[str, Form()],
    db: t.Annotated[DBSession, Depends(get_db)],
) -> RedirectResponse:
    DocumentRepository.validate_text_files(files)
    session = SessionRepository.create(db, AnnotatorSessionCreate(gazetteer=gazetteer))
    DocumentRepository.create_from_text_files(
        db, files, session.id, spacy_model, apply_spacy=False
    )
    return RedirectResponse(
        url=request.app.url_path_for("annotate", session_id=session.id, doc_index=0),
        status_code=status.HTTP_302_FOUND,
    )


@router.post("/session/read/legacy-files", tags=["session"])
def create_from_legacy_files(
    db: t.Annotated[DBSession, Depends(get_db)],
) -> LegacyFilesResponse:
    legacy_files = list(db_location.parent.glob("*.json"))
    if not legacy_files:
        return LegacyFilesResponse()
    files_loaded = 0
    files_failed = []
    for legacy_file in legacy_files:
        try:
            with open(legacy_file, encoding="utf-8") as infile:
                content = infile.read()
            SessionRepository.create_from_json(db, content, keep_id=True)
        except (InvalidUploadException, UnicodeDecodeError):
            files_failed.append(legacy_file.name)
        else:
            legacy_file.unlink()
            files_loaded += 1
    return LegacyFilesResponse(
        files_found=len(legacy_files),
        files_loaded=files_loaded,
        files_failed=files_failed,
    )


@router.post("/session/continue/cached", tags=["session"])
def continue_session_cached(
    request: Request,
    db: t.Annotated[DBSession, Depends(get_db)],
    session_id: t.Annotated[uuid.UUID, Form()],
) -> RedirectResponse:
    try:
        session = get_session(db, session_id)
    except SessionNotFoundException:
        return RedirectResponse(
            request.app.url_path_for("continue_session"),
            status_code=status.HTTP_302_FOUND,
        )
    return RedirectResponse(
        request.app.url_path_for("annotate", session_id=session.id, doc_index=0),
        status_code=status.HTTP_302_FOUND,
    )


@router.post("/session/continue/file", tags=["session"])
def continue_session_file(
    request: Request,
    db: t.Annotated[DBSession, Depends(get_db)],
    session_file: UploadFile | None = None,
) -> RedirectResponse:
    if session_file and session_file.filename:
        try:
            session_content = session_file.file.read().decode("utf-8")
        except UnicodeDecodeError as error:
            raise InvalidUploadException(
                "Session file must be valid UTF-8 JSON."
            ) from error
        session = SessionRepository.create_from_json(db, session_content, keep_id=False)
        return RedirectResponse(
            request.app.url_path_for("annotate", session_id=session.id, doc_index=0),
            status_code=status.HTTP_302_FOUND,
        )
    return RedirectResponse(
        request.app.url_path_for("continue_session"),
        status_code=status.HTTP_302_FOUND,
    )


@router.delete("/session/{session_id}", tags=["session"])
def delete_session(
    db: t.Annotated[DBSession, Depends(get_db)],
    session: t.Annotated[AnnotatorSession, Depends(get_session)],
) -> BaseResponse:
    SessionRepository.delete(db, session.id)
    return BaseResponse()
