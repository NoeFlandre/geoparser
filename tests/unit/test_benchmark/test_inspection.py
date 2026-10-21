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

    assert len(records) == 1
    assert records[0]["reference"] == [
        {
            "start": 0,
            "end": 5,
            "name": "Paris",
            "latitude": 48.8566,
            "longitude": 2.3522,
        }
    ]
    assert records[0]["prediction"][runner.RECOGNITION][0]["end"] == 5
    assert records[0]["prediction"][runner.RESOLUTION][0]["identifier"] == (
        "geonames:2988507"
    )
    assert records[0]["resolution_input"] == "gold_reference"
    assert records[0]["phase_status"] == {
        runner.RECOGNITION: "failed",
        runner.RESOLUTION: "complete",
    }
    assert "raw_output" not in records[0]["failures"][runner.RECOGNITION]
    assert "message" not in records[0]["failures"][runner.RECOGNITION]
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == records[0]
    assert "Paris is here." not in path.read_text(encoding="utf-8")
