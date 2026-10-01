import json

from scripts.panx_benchmark.report import (
    _language_rows,
    _model_summary,
    render_markdown,
    write_reports,
)


def _feasibility_result():
    return {
        "evaluation_kind": "bounded_feasibility_sample",
        "repository_commit": "abc123",
        "checkpoint_snapshot_id": "snapshot-123",
        "seed": 0,
        "hardware": {
            "platform": "linux",
            "torch_threads": 4,
            "cuda_available": False,
        },
        "dataset": {
            "id": "unimelb-nlp/wikiann",
            "revision": "f0a3",
            "split": "test",
            "eligible_languages": ["en"],
            "missing_target_languages": ["ha"],
            "source_test_examples": 10,
            "evaluated_examples": 1,
            "data_load_seconds": 0.1,
        },
        "evaluation": {"batch_size": 8},
        "language_list": {
            "source_revision": "c6b5039",
            "source_path": "docs/sentence-splitting.md",
            "source_url": "https://example.test/source",
        },
        "models": [],
        "estimated_full_matrix_inference_seconds": 12.0,
        "full_matrix_estimate_note": "Linear estimate from a small sample.",
    }


def test_feasibility_markdown_clearly_distinguishes_sample_from_full_quality():
    markdown = render_markdown(_feasibility_result())

    assert "Feasibility sample only" in markdown
    assert "not a full-test quality comparison" in markdown
    assert "Immutable run snapshot: `snapshot-123`" in markdown


def test_feasibility_markdown_names_missing_target_languages():
    markdown = render_markdown(_feasibility_result())

    assert "`ha`" in markdown


def test_feasibility_report_round_trips_both_local_files(tmp_path):
    result = _feasibility_result()
    markdown = render_markdown(result)
    json_path, markdown_path = write_reports(tmp_path, result)

    assert json.loads(json_path.read_text()) == result
    assert markdown_path.read_text() == markdown


def test_language_rows_distinguish_untested_and_measured_languages():
    rows = _language_rows(
        {
            "per_language": {
                "en": {
                    "status": "not_evaluated_english_only",
                    "documented_support": False,
                    "evaluated_examples": 0,
                    "metrics": None,
                },
                "fr": {
                    "status": "evaluated",
                    "documented_support": "evaluated_multilingual_claim",
                    "evaluated_examples": 1,
                    "metrics": {
                        "gold_spans": 1,
                        "precision": 1.0,
                        "recall": 1.0,
                        "f1": 1.0,
                    },
                },
            },
        }
    )

    assert len(rows) == 2
    assert (
        "| `en` | not_evaluated_english_only | False | — | — | — | — | — |" in rows[0]
    )
    assert (
        "| `fr` | evaluated | evaluated_multilingual_claim | 1 | 1 | 1.0000 |"
        in rows[1]
    )


def test_unspecified_multilingual_card_does_not_render_as_zero_languages():
    row = _model_summary(
        {
            "model_id": "fastino/gliner2.5-multi-v1",
            "documented_languages": [],
            "evaluated_examples": 1,
            "macro": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "steady_examples_per_second": 1.0,
            "checkpoint_download_seconds": 1.0,
            "model_load_seconds": 1.0,
        }
    )

    assert "unspecified" in row
