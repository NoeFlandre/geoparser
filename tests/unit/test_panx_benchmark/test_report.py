import json

from scripts.panx_benchmark.report import _model_summary, render_markdown, write_reports


def test_feasibility_report_clearly_distinguishes_sample_from_full_quality(tmp_path):
    result = {
        "evaluation_kind": "bounded_feasibility_sample",
        "repository_commit": "abc123",
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

    markdown = render_markdown(result)
    json_path, markdown_path = write_reports(tmp_path, result)

    assert "Feasibility sample only" in markdown
    assert "not a full-test quality comparison" in markdown
    assert "`ha`" in markdown
    assert json.loads(json_path.read_text()) == result
    assert markdown_path.read_text() == markdown


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
