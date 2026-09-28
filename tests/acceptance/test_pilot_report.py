import json
import sys
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

import scripts.pilot as pilot
from geoparser.evaluation import Annotation
from scripts.pilot import (
    PILOT_CASES,
    PilotCase,
    PilotSpan,
    build_report,
    collect_predictions,
    combine_timings_ms,
    write_report,
)

pytestmark = pytest.mark.acceptance
scenarios("features/pilot_report.feature")


@pytest.fixture
def pilot_state() -> dict[str, Any]:
    return {}


@given(parsers.parse('the pilot observed "{text}"'))
def observed_text(pilot_state: dict[str, Any], text: str) -> None:
    pilot_state["text"] = text


@given(parsers.parse('the gold place is "{place}" with identifier "{identifier}"'))
def gold_place(pilot_state: dict[str, Any], place: str, identifier: str) -> None:
    text = pilot_state["text"]
    assert isinstance(text, str)
    start = text.index(place)
    pilot_state["gold"] = PilotSpan(start, start + len(place), identifier)


@given(parsers.parse('the predicted place is "{place}" with identifier "{identifier}"'))
def predicted_place(pilot_state: dict[str, Any], place: str, identifier: str) -> None:
    text = pilot_state["text"]
    assert isinstance(text, str)
    start = text.index(place)
    pilot_state["predicted"] = Annotation(start, start + len(place), identifier)


@when("I build the pilot report")
def build_pilot_report(pilot_state: dict[str, Any]) -> None:
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
def report_recognition_f1_is_perfect(pilot_state: dict[str, Any]) -> None:
    report = pilot_state["report"]
    assert report["aggregate"]["recognition"]["f1"] == 1.0


@then("the report resolution accuracy is 1.0")
def report_resolution_is_perfect(pilot_state: dict[str, Any]) -> None:
    report = pilot_state["report"]
    assert report["aggregate"]["resolution"]["accuracy"] == 1.0


@then(parsers.parse('the report preserves the span text "{place}"'))
def report_preserves_span_text(pilot_state: dict[str, Any], place: str) -> None:
    report = pilot_state["report"]
    assert report["documents"][0]["predicted"][0]["text"] == place


@given("two pilot documents whose gold places share the same offsets")
def two_documents_sharing_offsets(pilot_state: dict[str, Any]) -> None:
    pilot_state["cases"] = (
        PilotCase("encamp", "Encamp welcomes hikers.", (PilotSpan(0, 6, "3040686"),)),
        PilotCase("ordino", "Ordino keeps its paths.", (PilotSpan(0, 6, "3039678"),)),
    )


@given("only the first document is recognized and resolved")
def only_the_first_document_is_predicted(pilot_state: dict[str, Any]) -> None:
    pilot_state["predictions"] = ((Annotation(0, 6, "3040686"),), ())


@when("I build the pilot report for both documents")
def build_pilot_report_for_both_documents(pilot_state: dict[str, Any]) -> None:
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
def aggregate_recognition_recall(pilot_state: dict[str, Any], recall: float) -> None:
    report = pilot_state["report"]
    assert report["aggregate"]["recognition"]["recall"] == recall


@then(parsers.parse("the aggregate resolution accuracy is {accuracy:f}"))
def aggregate_resolution_accuracy(pilot_state: dict[str, Any], accuracy: float) -> None:
    report = pilot_state["report"]
    assert report["aggregate"]["resolution"]["accuracy"] == accuracy


@then(parsers.parse("the aggregate counts {count:d} gold annotations"))
def aggregate_counts_gold_annotations(pilot_state: dict[str, Any], count: int) -> None:
    report = pilot_state["report"]
    assert report["aggregate"]["gold_annotation_count"] == count


@given("two pilot documents read back in reverse order")
def documents_read_back_in_reverse(pilot_state: dict[str, Any]) -> None:
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
def collect_pilot_predictions(pilot_state: dict[str, Any]) -> None:
    pilot_state["predictions"] = collect_predictions(
        pilot_state["project"], pilot_state["document_ids"]
    )


@then("each document keeps its own predicted span")
def each_document_keeps_its_span(pilot_state: dict[str, Any]) -> None:
    assert pilot_state["predictions"] == [
        [Annotation(0, 6, "3040686")],
        [Annotation(15, 21, "3040686")],
    ]


def test_run_pilot_caps_context_and_uses_cpu_reranker(monkeypatch, tmp_path):
    class FakeModel:
        def __init__(self):
            self.converted = False

        def float(self):
            self.converted = True
            return self

        def parameters(self):
            return iter([SimpleNamespace(dtype="float32")])

    class FakeTransformer(FakeModel):
        def __init__(self):
            super().__init__()
            self.max_seq_length = 8192

    class FakeRecognizer:
        def __init__(self):
            self.model_name = "recognizer-test"

    class FakeResolver:
        def __init__(self, *args, **kwargs):
            self.transformer = FakeTransformer()
            self.reranker = FakeModel()
            self.model_name = "embedding-test"
            self.reranker_name = "reranker-test"
            self.rerank_top_k = 1

    class FakeGazetteerBuilder:
        def build(self, config_path):
            return tmp_path / "andorranames.gazetteer"

    class FakeProject:
        instances: ClassVar[list] = []

        def __init__(self, name):
            self.name = name
            self.documents = []
            self.instances.append(self)

        def create_documents(self, texts):
            self.documents = [SimpleNamespace(toponyms=[]) for _ in texts]
            return list(range(len(texts)))

        def run_recognizer(self, recognizer):
            self.recognizer = recognizer

        def run_resolver(self, resolver):
            self.resolver = resolver

        def get_documents(self, ids=None):
            if ids is None:
                return self.documents
            return [self.documents[index] for index in ids]

        def delete(self):
            self.deleted = True

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(set_num_threads=lambda count: None, float32="float32"),
    )
    monkeypatch.setitem(
        sys.modules,
        "geoparser.gazetteer.build",
        SimpleNamespace(GazetteerBuilder=FakeGazetteerBuilder),
    )
    monkeypatch.setitem(
        sys.modules,
        "geoparser.modules",
        SimpleNamespace(GLiNER2Recognizer=FakeRecognizer, JinaResolver=FakeResolver),
    )
    monkeypatch.setitem(
        sys.modules, "geoparser.project", SimpleNamespace(Project=FakeProject)
    )
    monkeypatch.setattr(pilot, "_configure_runtime", lambda *args, **kwargs: None)

    pilot.run_pilot(
        config_path=tmp_path / "andorranames.yaml", output_dir=tmp_path / "output"
    )

    resolver = FakeProject.instances[-1].resolver
    assert resolver.transformer.max_seq_length == 128
    assert not resolver.transformer.converted
    assert resolver.reranker.converted


def test_pilot_cases_have_valid_hand_written_gold_spans():
    assert len(PILOT_CASES) == 13
    assert sum(len(case.gold) for case in PILOT_CASES) == 15

    for case in PILOT_CASES:
        for span in case.gold:
            assert 0 <= span.start < span.end <= len(case.text)
            assert case.text[span.start : span.end].strip()
            assert span.identifier.isdigit()


def test_build_report_records_annotations_metrics_and_timings():
    cases = (
        PilotCase(
            "capital",
            "Andorra la Vella is the capital.",
            (PilotSpan(0, 16, "3041563"),),
        ),
        PilotCase("empty", "Nothing is named here.", ()),
    )
    predictions = ((Annotation(0, 16, "3041563"),), ())

    report = build_report(
        cases,
        predictions,
        (12.34567, 2.0),
        models={"recognizer": "recognizer-test"},
        configuration={"gazetteer": "andorranames"},
    )

    assert report["schema_version"] == 1
    assert report["models"] == {"recognizer": "recognizer-test"}
    assert report["configuration"] == {"gazetteer": "andorranames"}
    assert report["aggregate"] == {
        "document_count": 2,
        "gold_annotation_count": 1,
        "predicted_annotation_count": 1,
        "recognition": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
        "resolution": {"accuracy": 1.0},
    }
    assert report["documents"][0] == {
        "id": "capital",
        "text": "Andorra la Vella is the capital.",
        "gold": [
            {
                "start": 0,
                "end": 16,
                "text": "Andorra la Vella",
                "identifier": "3041563",
            }
        ],
        "predicted": [
            {
                "start": 0,
                "end": 16,
                "text": "Andorra la Vella",
                "identifier": "3041563",
            }
        ],
        "metrics": {
            "recognition": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
            "resolution": {"accuracy": 1.0},
        },
        "elapsed_ms": 12.346,
    }


def test_aggregate_counts_are_the_metric_denominators():
    cases = (
        PilotCase("encamp", "Encamp welcomes hikers.", (PilotSpan(0, 6, "3040686"),)),
    )
    predictions = ((Annotation(0, 6, "3040686"), Annotation(0, 6, "3040686")),)

    aggregate = build_report(cases, predictions, (1.0,), models={})["aggregate"]

    assert aggregate["predicted_annotation_count"] == 1
    assert aggregate["recognition"]["precision"] == 1.0


def test_combine_timings_ms_matches_documents_by_text():
    cases = (
        PilotCase("first", "Encamp.", ()),
        PilotCase("second", "Canillo.", ()),
    )
    recognition_ms = {"Canillo.": 2.0, "Encamp.": 1.0}
    resolution_ms = {"Encamp.": 0.5, "Canillo.": 0.25}

    assert combine_timings_ms(cases, recognition_ms, resolution_ms) == [1.5, 2.25]


def test_combine_timings_ms_treats_an_unobserved_document_as_untimed():
    cases = (PilotCase("first", "Encamp.", ()),)

    assert combine_timings_ms(cases, {}, {}) == [0.0]


def test_pilot_case_texts_are_unique():
    texts = [case.text for case in PILOT_CASES]

    assert len(set(texts)) == len(texts)


def test_build_report_rejects_misaligned_inputs():
    case = PilotCase("one", "One.", ())

    with pytest.raises(ValueError, match="same length"):
        build_report((case,), (), (), models={})


def test_write_report_emits_json_and_markdown(tmp_path):
    report = build_report(
        (PilotCase("empty", "Nothing.", ()),),
        ((),),
        (0.5,),
        models={"recognizer": "recognizer-test"},
        configuration={"gazetteer": "andorranames"},
    )

    json_path, markdown_path = write_report(report, tmp_path)

    assert json.loads(json_path.read_text()) == report
    markdown = markdown_path.read_text()
    assert "# Geoparsing pilot" in markdown
    assert "recognizer-test" in markdown
    assert "| empty |" in markdown
