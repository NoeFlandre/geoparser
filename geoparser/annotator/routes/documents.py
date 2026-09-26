"""Document operations for the annotator API."""

import typing as t

from fastapi import APIRouter, Depends, Form, Response, UploadFile, status
from sqlmodel import Session as DBSession

from geoparser.annotator.db.crud import DocumentRepository
from geoparser.annotator.db.db import get_db
from geoparser.annotator.db.models import AnnotatorDocument, AnnotatorSession
from geoparser.annotator.dependencies import get_document, get_session
from geoparser.annotator.models.api import (
    BaseResponse,
    ParsingResponse,
    PreAnnotatedTextResponse,
    ProgressResponse,
)

router = APIRouter()


@router.post("/session/{session_id}/documents", tags=["document"])
def add_documents(
    response: Response,
    db: t.Annotated[DBSession, Depends(get_db)],
    session: t.Annotated[AnnotatorSession, Depends(get_session)],
    spacy_model: t.Annotated[str, Form()],
    files: list[UploadFile] | None = None,
) -> BaseResponse:
    if not files:
        response.status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
        return BaseResponse(message="No files selected.", status="error")
    DocumentRepository.create_from_text_files(
        db, files, session.id, spacy_model, apply_spacy=False
    )
    return BaseResponse()


@router.get("/session/{session_id}/documents", tags=["document"])
def get_documents(
    session: t.Annotated[AnnotatorSession, Depends(get_session)],
) -> list[AnnotatorDocument]:
    return session.documents


@router.post("/session/{session_id}/document/{doc_index}/parse", tags=["document"])
def parse_document(
    db: t.Annotated[DBSession, Depends(get_db)],
    doc: t.Annotated[AnnotatorDocument, Depends(get_document)],
) -> ParsingResponse:
    if not doc.spacy_applied:
        DocumentRepository.parse(db, doc.id)
        return ParsingResponse(parsed=True)
    return ParsingResponse(parsed=False)


@router.get("/session/{session_id}/document/{doc_index}/progress", tags=["document"])
def get_document_progress(
    db: t.Annotated[DBSession, Depends(get_db)],
    doc: t.Annotated[AnnotatorDocument, Depends(get_document)],
) -> ProgressResponse:
    return ProgressResponse(**DocumentRepository.get_document_progress(db, doc.id))


@router.get("/session/{session_id}/document/{doc_index}/text", tags=["document"])
def get_document_text(
    db: t.Annotated[DBSession, Depends(get_db)],
    doc: t.Annotated[AnnotatorDocument, Depends(get_document)],
) -> PreAnnotatedTextResponse:
    return PreAnnotatedTextResponse(
        pre_annotated_text=DocumentRepository.get_pre_annotated_text(db, doc.id)
    )


@router.delete(
    "/session/{session_id}/document/{doc_index}",
    tags=["document"],
    dependencies=[Depends(get_document)],
)
def delete_document(
    db: t.Annotated[DBSession, Depends(get_db)],
    session: t.Annotated[AnnotatorSession, Depends(get_session)],
    doc_index: int,
) -> BaseResponse:
    DocumentRepository.delete(db, session.documents[doc_index].id)
    return BaseResponse()
