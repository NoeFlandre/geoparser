import uuid

import pytest

from geoparser.project.project import Project


@pytest.mark.unit
def test_create_documents_returns_ordered_client_ids_that_select_the_documents():
    project = Project("bulk insert ID order")

    ids = project.create_documents(["First", "Second"])

    assert all(isinstance(document_id, uuid.UUID) for document_id in ids)
    documents = project.get_documents(ids=ids)
    assert [document.text for document in documents] == ["First", "Second"]
