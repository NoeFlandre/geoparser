from typing import Any

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from geoparser.modules.recognizers.manual import ManualRecognizer
from geoparser.modules.resolvers.manual import ManualResolver
from geoparser.project import Project

pytestmark = pytest.mark.acceptance
scenarios("features/project_persistence.feature")


@pytest.fixture
def project_state() -> dict[str, Any]:
    return {}


@given(parsers.parse('I parsed "{text}" in project "{name}" with the manual pipeline'))
def parsed_project(
    project_state: dict[str, Any], andorra_gazetteer, text: str, name: str
) -> None:
    span = (0, len("Andorra la Vella"))
    project = Project(name)
    project.create_documents([text])
    project.run_recognizer(
        ManualRecognizer(label="persisted", texts=[text], references=[[span]])
    )
    project.run_resolver(
        ManualResolver(
            label="persisted",
            texts=[text],
            references=[[span]],
            referents=[[("andorranames", "3041563")]],
        )
    )


@when(parsers.parse('I open project "{name}" again'))
def reopen(project_state: dict[str, Any], name: str) -> None:
    project_state["documents"] = Project(name).get_documents()


@then(
    parsers.parse(
        'the document still has the place "{place}" resolved to "{identifier}"'
    )
)
def still_resolved(project_state: dict[str, Any], place: str, identifier: str) -> None:
    (document,) = project_state["documents"]
    (toponym,) = document.toponyms
    assert toponym.text == place
    assert toponym.location is not None
    assert toponym.location.identifier == identifier
