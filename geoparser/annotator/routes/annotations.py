"""Candidate lookup and annotation routes."""

import json
import typing as t
from datetime import datetime
from io import StringIO

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlmodel import Session as DBSession

from geoparser.annotator.db.crud import SessionRepository, ToponymRepository
from geoparser.annotator.db.db import get_db
from geoparser.annotator.db.models import (
    AnnotatorDocument,
    AnnotatorSession,
    AnnotatorSessionUpdate,
    AnnotatorToponymBase,
    AnnotatorToponymCreate,
    AnnotatorToponymUpdate,
)
from geoparser.annotator.dependencies import get_document, get_session
from geoparser.annotator.exceptions import ToponymNotFoundException
from geoparser.annotator.models.api import AnnotationEdit, BaseResponse, CandidatesGet

router = APIRouter()


@router.post(
    "/session/{session_id}/document/{doc_index}/get_candidates", tags=["candidates"]
)
def get_candidates(
    session: t.Annotated[AnnotatorSession, Depends(get_session)],
    doc: t.Annotated[AnnotatorDocument, Depends(get_document)],
    candidates_request: CandidatesGet,
) -> dict[str, t.Any]:
    return ToponymRepository.get_candidates(doc, session.gazetteer, candidates_request)


@router.post(
    "/session/{session_id}/document/{doc_index}/annotation", tags=["annotation"]
)
def create_annotation(
    db: t.Annotated[DBSession, Depends(get_db)],
    session: t.Annotated[AnnotatorSession, Depends(get_session)],
    doc: t.Annotated[AnnotatorDocument, Depends(get_document)],
    annotation: AnnotatorToponymBase,
) -> BaseResponse:
    ToponymRepository.create(
        db,
        AnnotatorToponymCreate(
            text=annotation.text, start=annotation.start, end=annotation.end
        ),
        additional={"document_id": doc.id},
    )
    SessionRepository.update(
        db, AnnotatorSessionUpdate(id=session.id, last_updated=datetime.now())
    )
    return BaseResponse()


@router.get("/session/{session_id}/annotations/download", tags=["annotation"])
def download_annotations(
    db: t.Annotated[DBSession, Depends(get_db)],
    session: t.Annotated[AnnotatorSession, Depends(get_session)],
) -> StreamingResponse:
    file_content = json.dumps(
        SessionRepository.read_to_json(db, session.id), ensure_ascii=False, indent=4
    )
    return StreamingResponse(
        StringIO(file_content),
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="annotations_{session.id}.json"'
        },
    )


@router.put(
    "/session/{session_id}/document/{doc_index}/annotation", tags=["annotation"]
)
def overwrite_annotation(
    db: t.Annotated[DBSession, Depends(get_db)],
    session: t.Annotated[AnnotatorSession, Depends(get_session)],
    doc: t.Annotated[AnnotatorDocument, Depends(get_document)],
    annotation: AnnotatorToponymBase,
) -> BaseResponse:
    ToponymRepository.annotate_many(db, doc, annotation)
    SessionRepository.update(
        db, AnnotatorSessionUpdate(id=session.id, last_updated=datetime.now())
    )
    return BaseResponse()


@router.patch(
    "/session/{session_id}/document/{doc_index}/annotation", tags=["annotation"]
)
def update_annotation(
    db: t.Annotated[DBSession, Depends(get_db)],
    session: t.Annotated[AnnotatorSession, Depends(get_session)],
    doc: t.Annotated[AnnotatorDocument, Depends(get_document)],
    annotation: AnnotationEdit,
) -> BaseResponse:
    toponym = ToponymRepository.get_toponym(
        doc, annotation.old_start, annotation.old_end
    )
    if toponym is None:
        raise ToponymNotFoundException
    ToponymRepository.update(
        db,
        AnnotatorToponymUpdate(
            id=toponym.id,
            start=annotation.new_start,
            end=annotation.new_end,
            text=annotation.new_text,
        ),
        document_id=doc.id,
    )
    SessionRepository.update(
        db, AnnotatorSessionUpdate(id=session.id, last_updated=datetime.now())
    )
    return BaseResponse()


@router.delete(
    "/session/{session_id}/document/{doc_index}/annotation", tags=["annotation"]
)
def delete_annotation(
    db: t.Annotated[DBSession, Depends(get_db)],
    session: t.Annotated[AnnotatorSession, Depends(get_session)],
    doc: t.Annotated[AnnotatorDocument, Depends(get_document)],
    start: int,
    end: int,
) -> BaseResponse:
    toponym = ToponymRepository.get_toponym(doc, start, end)
    if toponym is None:
        raise ToponymNotFoundException
    ToponymRepository.delete(db, toponym.id)
    SessionRepository.update(
        db, AnnotatorSessionUpdate(id=session.id, last_updated=datetime.now())
    )
    return BaseResponse()
