"""Hand-checked support of the #98 arms on the twelve MultiCoNER II languages."""

import pytest

from scripts.multiconer_benchmark.arms import arm_keys, language_support


def test_arms_are_the_three_pinned_panx_models_in_order():
    assert arm_keys() == ("spacy_en", "gliner2_multi", "xlmr_ner_hrl")


@pytest.mark.parametrize(
    "language", ["bn", "de", "en", "es", "fa", "fr", "hi", "it", "pt", "sv", "uk", "zh"]
)
def test_english_only_spacy_is_documented_for_english_and_unsupported_elsewhere(
    language,
):
    expected = "documented" if language == "en" else "unsupported"
    assert language_support("spacy_en", language) == expected


@pytest.mark.parametrize(
    "language", ["bn", "de", "en", "es", "fa", "fr", "hi", "it", "pt", "sv", "uk", "zh"]
)
def test_gliner_without_a_language_list_is_unspecified_everywhere(language):
    assert language_support("gliner2_multi", language) == "unspecified"


def test_xlmr_documented_languages_and_transfer_languages_are_split():
    documented = {"de", "en", "es", "fr", "it", "pt", "zh"}
    transfer = {"bn", "fa", "hi", "sv", "uk"}
    for language in documented:
        assert language_support("xlmr_ner_hrl", language) == "documented"
    for language in transfer:
        assert language_support("xlmr_ner_hrl", language) == "transfer"
