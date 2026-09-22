import json
from types import SimpleNamespace

import pytest

from geoparser.evaluation import Annotation
from scripts.pilot import (
    MAX_SEQUENCE_LENGTH,
    PILOT_CASES,
    PilotCase,
    PilotSpan,
    _limit_model_context,
    _use_cpu_float32,
    build_report,
    collect_predictions,
    combine_timings_ms,
    write_report,
)


def test_pilot_bounds_context_length_for_short_fixture_documents() -> None:
    resolver = SimpleNamespace(transformer=SimpleNamespace(max_seq_length=8192))

    _limit_model_context(resolver)

    assert MAX_SEQUENCE_LENGTH == 128
    assert resolver.transformer.max_seq_length == MAX_SEQUENCE_LENGTH


def test_pilot_uses_float32_for_cpu_model_inference() -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.converted = False

        def float(self):
            self.converted = True
            return self

    resolver = SimpleNamespace(transformer=FakeModel(), reranker=FakeModel())

    _use_cpu_float32(resolver)

    assert not resolver.transformer.converted
    assert resolver.reranker.converted


def test_pilot_cases_have_valid_hand_written_gold_spans() -> None:
    assert len(PILOT_CASES) == 13
    assert sum(len(case.gold) for case in PILOT_CASES) == 15

    for case in PILOT_CASES:
        for span in case.gold:
            assert 0 <= span.start < span.end <= len(case.text)
            assert case.text[span.start : span.end].strip()
            assert span.identifier.isdigit()


def test_build_report_records_annotations_metrics_and_timings() -> None:
    cases = (
        PilotCase(
            "capital",
            "Andorra la Vella is the capital.",
            (PilotSpan(0, 16, "3041563"),),
        ),
        PilotCase("empty", "Nothing is named here.", ()),
    )
    predictions = (
        (Annotation(0, 16, "3041563"),),
        (),
    )

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


def test_aggregate_keeps_documents_with_identical_spans_apart() -> None:
    cases = (
        PilotCase("encamp", "Encamp welcomes hikers.", (PilotSpan(0, 6, "3040686"),)),
        PilotCase("canillo", "Canillo lies north.", (PilotSpan(0, 7, "3041204"),)),
        PilotCase("route", "Encamp and Canillo.", (PilotSpan(0, 6, "3040686"),)),
    )
    # Only the first document is recognized and resolved. Without document
    # identity the third document's gold span collapses onto the first one's,
    # hiding a missed annotation behind a perfect recall.
    predictions = ((Annotation(0, 6, "3040686"),), (), ())

    aggregate = build_report(cases, predictions, (1.0, 1.0, 1.0), models={})[
        "aggregate"
    ]

    assert aggregate["gold_annotation_count"] == 3
    assert aggregate["predicted_annotation_count"] == 1
    assert aggregate["recognition"]["precision"] == 1.0
    assert aggregate["recognition"]["recall"] == pytest.approx(1 / 3)
    assert aggregate["resolution"]["accuracy"] == pytest.approx(1 / 3)


def test_aggregate_counts_are_the_metric_denominators() -> None:
    # A repeated span within one document is one annotation to the metrics, so
    # the reported counts have to describe that same population.
    cases = (
        PilotCase("encamp", "Encamp welcomes hikers.", (PilotSpan(0, 6, "3040686"),)),
    )
    predictions = ((Annotation(0, 6, "3040686"), Annotation(0, 6, "3040686")),)

    aggregate = build_report(cases, predictions, (1.0,), models={})["aggregate"]

    assert aggregate["predicted_annotation_count"] == 1
    assert aggregate["recognition"]["precision"] == 1.0


def test_collect_predictions_follows_the_requested_document_order() -> None:
    class FakeDocument:
        def __init__(self, identifier: str, start: int) -> None:
            self.id = identifier
            self.toponyms = [
                SimpleNamespace(
                    start=start,
                    end=start + 6,
                    location=SimpleNamespace(identifier="3040686"),
                )
            ]

    documents = {"a": FakeDocument("a", 0), "b": FakeDocument("b", 10)}

    class FakeProject:
        def get_documents(self, ids=None):
            # The database has no ordering contract, so an unfiltered read is
            # free to hand documents back in any order at all.
            if ids is None:
                return list(reversed(list(documents.values())))
            return [documents[id] for id in ids]

    predictions = collect_predictions(FakeProject(), ["a", "b"])

    assert predictions == [
        [Annotation(0, 6, "3040686")],
        [Annotation(10, 16, "3040686")],
    ]


def test_combine_timings_ms_matches_documents_by_text() -> None:
    cases = (
        PilotCase("first", "Encamp.", ()),
        PilotCase("second", "Canillo.", ()),
    )
    # Both phases observe the documents in their own order.
    recognition_ms = {"Canillo.": 2.0, "Encamp.": 1.0}
    resolution_ms = {"Encamp.": 0.5, "Canillo.": 0.25}

    assert combine_timings_ms(cases, recognition_ms, resolution_ms) == [1.5, 2.25]


def test_combine_timings_ms_treats_an_unobserved_document_as_untimed() -> None:
    cases = (PilotCase("first", "Encamp.", ()),)

    assert combine_timings_ms(cases, {}, {}) == [0.0]


def test_pilot_case_texts_are_unique() -> None:
    # Timings are matched back to their case by text, which only identifies a
    # document while the texts stay distinct.
    texts = [case.text for case in PILOT_CASES]

    assert len(set(texts)) == len(texts)


def test_build_report_rejects_misaligned_inputs() -> None:
    case = PilotCase("one", "One.", ())

    with pytest.raises(ValueError, match="same length"):
        build_report((case,), (), (), models={})


def test_write_report_emits_json_and_markdown(tmp_path) -> None:
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
