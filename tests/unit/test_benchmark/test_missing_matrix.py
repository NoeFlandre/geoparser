"""The first-five-corpora comparison is a fixed, reviewed experiment."""

from scripts.benchmark import missing_matrix


def test_missing_matrix_is_the_exact_twenty_slice_scope():
    assert [
        (cell.corpus, cell.language, cell.documents, cell.gold_toponyms)
        for cell in missing_matrix.MATRIX
    ] == [
        ("geovirus", "en", 229, 2167),
        ("hipe2020", "de", 48, 558),
        ("hipe2020", "en", 38, 158),
        ("hipe2020", "fr", 43, 800),
        ("newseye", "de", 8, 628),
        ("newseye", "fi", 18, 214),
        ("newseye", "fr", 33, 665),
        ("newseye", "sv", 18, 265),
        ("topres19th", "en", 110, 880),
        ("newsli", "ar", 500, 3319),
        ("newsli", "de", 500, 1019),
        ("newsli", "es", 500, 1270),
        ("newsli", "fa", 71, 326),
        ("newsli", "ja", 500, 1577),
        ("newsli", "pl", 186, 196),
        ("newsli", "ro", 226, 385),
        ("newsli", "sr", 500, 985),
        ("newsli", "ta", 500, 1578),
        ("newsli", "tr", 500, 1578),
        ("newsli", "uk", 164, 258),
    ]
    assert sum(cell.documents for cell in missing_matrix.MATRIX) == 4692
    assert sum(cell.gold_toponyms for cell in missing_matrix.MATRIX) == 18826


def test_missing_pipelines_and_model_pins_are_explicit():
    assert missing_matrix.MISSING_PIPELINES == (
        "swapped",
        "xlm-minilm",
        "otter-cross-minilm",
    )
    assert missing_matrix.MODEL_PINS == {
        "gliner": (
            "fastino/gliner2.5-multi-v1",
            "cf5593a5d45e3bbf204b9df621b13b1c0cf25ee3",
        ),
        "jina_embeddings": (
            "jinaai/jina-embeddings-v5-text-small",
            "dd76d535f5447ca3897a9c893fb1e612ead98192",
        ),
        "jina_reranker": (
            "jinaai/jina-reranker-v3.5",
            "e8a93f33f0b22108f8c2364f8484ce3422552fbc",
        ),
        "minilm": (
            "dguzh/geo-all-MiniLM-L6-v2",
            "d678769f195c194ffdc6f3735a9360118a855b43",
        ),
        "xlm_r": (
            "Davlan/xlm-roberta-base-ner-hrl",
            "253f557bd8249b8515114cfd7f71974fe5fa4d2f",
        ),
        "otter_cross": (
            "whoisjones/otter-cross-mmbert",
            "8729188e4f5fc7948d0e9dfd7d7e6d36c2e7270d",
        ),
    }


def test_xlm_support_distinguishes_finetuning_from_transfer():
    assert missing_matrix.language_coverage("xlm_r", "ar") == "fine_tuned"
    assert missing_matrix.language_coverage("xlm_r", "fi") == "cross_lingual_transfer"
    assert missing_matrix.language_coverage("xlm_r", "ta") == "cross_lingual_transfer"
    assert missing_matrix.language_coverage("otter_cross", "ta") == "unspecified"


def test_minilm_gold_span_resolution_is_shareable_only_for_matching_pins():
    assert missing_matrix.resolver_family("xlm-minilm") == "minilm"
    assert missing_matrix.resolver_family("otter-cross-minilm") == "minilm"
    assert missing_matrix.resolver_family("swapped") == "jina"
    assert missing_matrix.resolver_pins("xlm-minilm") == missing_matrix.resolver_pins(
        "otter-cross-minilm"
    )


def test_model_metadata_records_license_and_cell_specific_language_coverage():
    arabic = missing_matrix.pipeline_model_metadata("xlm-minilm", "ar")
    finnish = missing_matrix.pipeline_model_metadata("xlm-minilm", "fi")
    jina = missing_matrix.pipeline_model_metadata("swapped", "en")
    otter = missing_matrix.pipeline_model_metadata("otter-cross-minilm", "ta")

    assert arabic["recognizer"]["language_coverage"] == "fine_tuned"
    assert finnish["recognizer"]["language_coverage"] == "cross_lingual_transfer"
    assert jina["reranker"]["license"] == "CC-BY-NC-4.0"
    assert jina["resolver"]["language_support"].startswith("119+ languages")
    assert otter["recognizer"]["language_coverage"] == "unspecified"
