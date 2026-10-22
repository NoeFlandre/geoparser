"""Per-example inspection includes both phases without article text."""

import json

from scripts.benchmark import checkpoint, runner
from scripts.benchmark.corpus import Document, GoldSpan


def test_inspection_record_contains_gold_both_predictions_and_failure_state(tmp_path):
    document = Document(
        "doc-1",
        "Paris is here.",
        (GoldSpan(0, 5, "Paris", 48.8566, 2.3522),),
    )
    identity = checkpoint.RunIdentity(
        pipeline="gliner-minilm",
        corpus_digest="corpus-pin",
        gazetteer="geonames",
        min_similarity=0.0,
        limit=None,
        commit="code-pin",
    )
    state = checkpoint.Checkpoint(identity)
    state.record(
        runner.RECOGNITION,
        "doc-1",
        [{"start": 0, "end": 5, "identifier": None}],
    )
    state.record(
        runner.RESOLUTION,
        "doc-1",
        [
            {
                "start": 0,
                "end": 5,
                "identifier": "geonames:2988507",
                "latitude": 48.8566,
                "longitude": 2.3522,
            }
        ],
    )
    state.failures[runner.RECOGNITION] = {
        "doc-1": {
            "kind": "invalid_output",
            "reason": "invalid label",
            "message": "model echoed the source sentence",
            "raw_output": [{"text": "Paris"}],
        }
    }

    records = runner.inspection_records(
        [document],
        state,
        pipeline="gliner-minilm",
        scoring_view="strict-raw-v1",
    )
    path = tmp_path / "inspection.jsonl"
    runner.write_inspection(
        path,
        [document],
        state,
        pipeline="gliner-minilm",
        scoring_view="strict-raw-v1",
    )

    record = records[0]
    payload = json.loads(path.read_text(encoding="utf-8"))
    serialized = path.read_text(encoding="utf-8")
    assert (
        len(records),
        record["reference"],
        record["prediction"][runner.RECOGNITION][0]["end"],
        record["prediction"][runner.RESOLUTION][0]["identifier"],
        record["resolution_input"],
        record["phase_status"],
        "raw_output" in record["failures"][runner.RECOGNITION],
        "message" in record["failures"][runner.RECOGNITION],
        payload == record,
        "Paris is here." in serialized,
    ) == (
        1,
        [
            {
                "start": 0,
                "end": 5,
                "name": "Paris",
                "latitude": 48.8566,
                "longitude": 2.3522,
            }
        ],
        5,
        "geonames:2988507",
        "gold_reference",
        {runner.RECOGNITION: "failed", runner.RESOLUTION: "complete"},
        False,
        False,
        True,
        False,
    )
