import json

from scripts.panx_benchmark.report import (
    _aggregate_lines,
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
        "evaluation": {"batch_sizes_by_model": {"gliner2_multi": 1}},
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


def test_spacy_transfer_report_does_not_claim_native_multilingual_support():
    lines = _aggregate_lines(
        {
            "evaluation": {
                "selected_model_keys": ["spacy_en"],
                "complete_model_matrix": False,
                "spacy_cross_lingual_transfer": True,
            },
            "models": [
                {
                    "key": "spacy_en",
                    "macro": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                    "micro": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                }
            ],
            "estimated_full_matrix_inference_seconds": 7.0,
            "full_matrix_estimate_note": "Measured on all test languages.",
        }
    )

    assert any("selected model subset (`spacy_en`)" in line for line in lines)
    assert any("cross-lingual transfer results" in line for line in lines)
    assert any("not evidence of native multilingual support" in line for line in lines)


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
                        "invalid_prediction_spans": 1,
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
        "| `en` | not_evaluated_english_only | False | — | — | — | — | — | — |"
        in rows[0]
    )
    assert (
        "| `fr` | evaluated | evaluated_multilingual_claim | 1 | 1 | 1 | "
        "1.0000 | 1.0000 | 1.0000 |" in rows[1]
    )


def test_unspecified_multilingual_card_does_not_render_as_zero_languages():
    row = _model_summary(
        {
            "model_id": "fastino/gliner2.5-multi-v1",
            "batch_size": 1,
            "documented_languages": [],
            "evaluated_examples": 1,
            "macro": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "steady_examples_per_second": 1.0,
            "checkpoint_download_seconds": 1.0,
            "model_load_seconds": 1.0,
        }
    )

    assert "unspecified" in row
    assert "| 1 |" in row


def test_historical_sample_discloses_unavailable_source_revision():
    import json
    from pathlib import Path

    directory = (
        Path(__file__).resolve().parents[3]
        / "benchmark-evidence/panx/feasibility-2026-09-30"
    )
    report = json.loads((directory / "report.json").read_text())
    audit = report["provenance_audit"]
    assert audit["source_commit_retrievable"] is False
    assert audit["usable_for_pipeline_selection"] is False
    assert audit["status"] == "unverified_source_unavailable"
    assert "not reproducible evidence" in (directory / "report.md").read_text()
