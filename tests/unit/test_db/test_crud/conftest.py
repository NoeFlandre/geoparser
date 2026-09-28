import pytest
from sqlmodel import Session

from geoparser.db.crud.reference import ReferenceRepository
from geoparser.db.models import (
    Document,
    DocumentCreate,
    Project,
    ProjectCreate,
    Recognizer,
    RecognizerCreate,
    ReferenceCreate,
    Resolver,
    ResolverCreate,
)


@pytest.fixture
def session(test_session: Session) -> Session:
    """Use the shared isolated database for CRUD tests."""
    return test_session


@pytest.fixture
def add_row(session: Session):
    """Persist one database row and return it refreshed."""

    def persist(instance):
        session.add(instance)
        session.commit()
        session.refresh(instance)
        return instance

    return persist


@pytest.fixture
def world(session: Session, add_row):
    """Two documents, references, recognizers, and resolvers."""
    project = add_row(Project(**ProjectCreate(name="demo").model_dump()))
    documents = [
        add_row(
            Document(**DocumentCreate(text=text, project_id=project.id).model_dump())
        )
        for text in ("Paris and Berlin", "Rome and Milan")
    ]
    recognizers = [
        add_row(
            Recognizer(
                **RecognizerCreate(id=identifier, name="R", config={}).model_dump()
            )
        )
        for identifier in ("rec-a", "rec-b")
    ]
    resolvers = [
        add_row(
            Resolver(**ResolverCreate(id=identifier, name="S", config={}).model_dump())
        )
        for identifier in ("res-a", "res-b")
    ]
    references = [
        ReferenceRepository.create(
            session,
            ReferenceCreate(
                start=0,
                end=5,
                document_id=document.id,
                recognizer_id=recognizers[0].id,
            ),
        )
        for document in documents
    ]
    return {
        "documents": documents,
        "recognizers": recognizers,
        "resolvers": resolvers,
        "references": references,
    }
