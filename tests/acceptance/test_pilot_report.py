from types import SimpleNamespace

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from geoparser.evaluation import Annotation
from scripts.pilot import PilotCase, PilotSpan, build_report, collect_predictions

pytestmark = pytest.mark.acceptance
scenarios("features/pilot_report.feature")


@pytest.fixture
def pilot_state() -> dict[str, object]:
    return {}


@given(parsers.parse('the pilot observed "{text}"'))
def observed_text(pilot_state: dict[str, object], text: str) -> None:
    pilot_state["text"] = text


@given(parsers.parse('the gold place is "{place}" with identifier "{identifier}"'))
def gold_place(pilot_state: dict[str, object], place: str, identifier: str) -> None:
    text = pilot_state["text"]
    assert isinstance(text, str)
    start = text.index(place)
    pilot_state["gold"] = PilotSpan(start, start + len(place), identifier)


@given(parsers.parse('the predicted place is "{place}" with identifier "{identifier}"'))
def predicted_place(
    pilot_state: dict[str, object], place: str, identifier: str
) -> None:
    text = pilot_state["text"]
    assert isinstance(text, str)
    start = text.index(place)
    pilot_state["predicted"] = Annotation(start, start + len(place), identifier)


@when("I build the pilot report")
def build_pilot_report(pilot_state: dict[str, object]) -> None:
    text = pilot_state["text"]
    gold = pilot_state["gold"]
    predicted = pilot_state["predicted"]
    assert isinstance(text, str)
    assert isinstance(gold, PilotSpan)
    assert isinstance(predicted, Annotation)

    pilot_state["report"] = build_report(
        (PilotCase("acceptance", text, (gold,)),),
        ((predicted,),),
        (0.0,),
        models={"recognizer": "acceptance"},
        configuration={"gazetteer": "andorranames"},
    )


@then("the report recognition F1 is 1.0")
def report_recognition_f1_is_perfect(pilot_state: dict[str, object]) -> None:
    report = pilot_state["report"]
    assert report["aggregate"]["recognition"]["f1"] == 1.0


@then("the report resolution accuracy is 1.0")
def report_resolution_is_perfect(pilot_state: dict[str, object]) -> None:
    report = pilot_state["report"]
    assert report["aggregate"]["resolution"]["accuracy"] == 1.0


@then(parsers.parse('the report preserves the span text "{place}"'))
def report_preserves_span_text(pilot_state: dict[str, object], place: str) -> None:
    report = pilot_state["report"]
    assert report["documents"][0]["predicted"][0]["text"] == place


@given("two pilot documents whose gold places share the same offsets")
def two_documents_sharing_offsets(pilot_state: dict[str, object]) -> None:
    pilot_state["cases"] = (
        PilotCase("encamp", "Encamp welcomes hikers.", (PilotSpan(0, 6, "3040686"),)),
        PilotCase("ordino", "Ordino keeps its paths.", (PilotSpan(0, 6, "3039678"),)),
    )


@given("only the first document is recognized and resolved")
def only_the_first_document_is_predicted(pilot_state: dict[str, object]) -> None:
    pilot_state["predictions"] = ((Annotation(0, 6, "3040686"),), ())


@when("I build the pilot report for both documents")
def build_pilot_report_for_both_documents(pilot_state: dict[str, object]) -> None:
    cases = pilot_state["cases"]
    predictions = pilot_state["predictions"]
    assert isinstance(cases, tuple)
    assert isinstance(predictions, tuple)

    pilot_state["report"] = build_report(
        cases,
        predictions,
        (0.0,) * len(cases),
        models={"recognizer": "acceptance"},
        configuration={"gazetteer": "andorranames"},
    )


@then(parsers.parse("the aggregate recognition recall is {recall:f}"))
def aggregate_recognition_recall(pilot_state: dict[str, object], recall: float) -> None:
    report = pilot_state["report"]
    assert report["aggregate"]["recognition"]["recall"] == recall


@then(parsers.parse("the aggregate resolution accuracy is {accuracy:f}"))
def aggregate_resolution_accuracy(
    pilot_state: dict[str, object], accuracy: float
) -> None:
    report = pilot_state["report"]
    assert report["aggregate"]["resolution"]["accuracy"] == accuracy


@then(parsers.parse("the aggregate counts {count:d} gold annotations"))
def aggregate_counts_gold_annotations(
    pilot_state: dict[str, object], count: int
) -> None:
    report = pilot_state["report"]
    assert report["aggregate"]["gold_annotation_count"] == count


@given("two pilot documents read back in reverse order")
def documents_read_back_in_reverse(pilot_state: dict[str, object]) -> None:
    documents = {
        "encamp": SimpleNamespace(
            toponyms=[
                SimpleNamespace(
                    start=0, end=6, location=SimpleNamespace(identifier="3040686")
                )
            ]
        ),
        "route": SimpleNamespace(
            toponyms=[
                SimpleNamespace(
                    start=15, end=21, location=SimpleNamespace(identifier="3040686")
                )
            ]
        ),
    }

    class ReversingProject:
        def get_documents(self, ids=None):
            if ids is None:
                return list(reversed(list(documents.values())))
            return [documents[id] for id in ids]

    pilot_state["project"] = ReversingProject()
    pilot_state["document_ids"] = ["encamp", "route"]


@when("I collect the pilot predictions by document ID")
def collect_pilot_predictions(pilot_state: dict[str, object]) -> None:
    pilot_state["predictions"] = collect_predictions(
        pilot_state["project"], pilot_state["document_ids"]
    )


@then("each document keeps its own predicted span")
def each_document_keeps_its_span(pilot_state: dict[str, object]) -> None:
    assert pilot_state["predictions"] == [
        [Annotation(0, 6, "3040686")],
        [Annotation(15, 21, "3040686")],
    ]
