"""Offline contract for the opt-in Otter arm plan and reviewed source pins."""

import hashlib
import json
from pathlib import Path

import pytest

from geoparser import modules
from scripts.panx_benchmark.constants import GLINER_ENTITY_LABELS, MODELS

ROOT = Path(__file__).resolve().parents[3]


def test_otter_is_available_through_the_existing_lazy_module_exports():
    assert modules.OtterRecognizer.NAME == "OtterRecognizer"


def test_otter_plan_preserves_existing_baselines_and_location_policy():
    plan = json.loads((ROOT / "docs/examples/otter-recognition-arms.json").read_text())
    assert plan["status"] == "planned"
    assert plan["adapter"] == "geoparser.modules.recognizers.otter.OtterRecognizer"
    assert plan["benchmark_interface"] == "predict_batch"


def test_otter_plan_uses_the_existing_location_labels():
    plan = json.loads((ROOT / "docs/examples/otter-recognition-arms.json").read_text())
    assert plan["label_mapping"] == {"city": "LOC", "country": "LOC", "location": "LOC"}
    assert tuple(plan["arms"][0]["entity_types"]) == GLINER_ENTITY_LABELS
    assert plan["arms"][1]["entity_types"] == ["city", "country", "location"]


def test_otter_plan_does_not_change_the_existing_default_arms():
    plan = json.loads((ROOT / "docs/examples/otter-recognition-arms.json").read_text())
    assert [spec.key for spec in MODELS] == [
        "spacy_en",
        "gliner2_multi",
        "xlmr_ner_hrl",
    ]
    assert plan["baselines"]["keys"] == ["gliner2_multi", "xlmr_ner_hrl"]
    assert plan["baselines"]["change_existing_defaults"] is False


def test_otter_plan_has_distinct_pinned_checkpoints_and_explicit_thresholds():
    plan = json.loads((ROOT / "docs/examples/otter-recognition-arms.json").read_text())
    cross, bi = plan["arms"]
    assert cross["model"] == {
        "identifier": "whoisjones/otter-cross-mmbert",
        "revision": "8729188e4f5fc7948d0e9dfd7d7e6d36c2e7270d",
        "sha256": "8987080bfc3e6672a75fb19ffe904d39e79ad804eabfda8247a62c347fb024b2",
    }
    assert bi["model"] == {
        "identifier": "whoisjones/otter-bi-mmbert",
        "revision": "53e10a09bc71a2e45980a7a257233a28305a5777",
        "sha256": "05c4f718fb9e5871d66b8eb68fc40e17d0d8611c5b8e6252e371662bc0f78c91",
    }
    assert (cross["threshold"], bi["threshold"]) == (0.5, 0.2)
    assert (cross["expected_architecture"], bi["expected_architecture"]) == (
        "cross_encoder",
        "bi_encoder",
    )


def test_otter_plan_does_not_claim_executed_results_or_embedding_caching():
    plan = json.loads((ROOT / "docs/examples/otter-recognition-arms.json").read_text())
    assert plan["execution"] == {
        "weights_downloaded": False,
        "custom_code_executed": False,
        "inference_run": False,
        "results": None,
    }
    assert [arm["label_embedding_cache"] for arm in plan["arms"]] == ["none", "none"]


def test_otter_plan_does_not_overstate_language_support_or_test_tuning():
    plan = json.loads((ROOT / "docs/examples/otter-recognition-arms.json").read_text())
    assert [arm["language_support"] for arm in plan["arms"]] == [
        "unspecified",
        "unspecified",
    ]
    assert [arm["training_overlap"] for arm in plan["arms"]] == ["unknown", "unknown"]
    assert plan["long_text_policy"] == "reject_before_prediction"


def test_otter_plan_freezes_development_thresholds_before_test():
    plan = json.loads((ROOT / "docs/examples/otter-recognition-arms.json").read_text())
    assert plan["threshold_selection"]["stage"] == "development_only"
    assert plan["threshold_selection"]["freeze_before_held_out"] is True


@pytest.mark.parametrize("artifact", ["source_manifest", "source_review"])
def test_otter_source_review_and_manifest_are_content_pinned(artifact):
    plan = json.loads((ROOT / "docs/examples/otter-recognition-arms.json").read_text())
    assert (
        hashlib.sha256((ROOT / plan[artifact]).read_bytes()).hexdigest()
        == plan[artifact + "_sha256"]
    )


def test_otter_review_names_the_import_closure_without_execution():
    plan = json.loads((ROOT / "docs/examples/otter-recognition-arms.json").read_text())
    manifest = json.loads((ROOT / plan["source_manifest"]).read_text())
    assert manifest["inference_imports"] == [
        "configuration_otter.py",
        "modeling_otter.py",
        "loss.py",
        "masks.py",
        "metrics.py",
    ]
    assert manifest["execution_status"] == "not_run"
    assert manifest["weights_downloaded"] is False
