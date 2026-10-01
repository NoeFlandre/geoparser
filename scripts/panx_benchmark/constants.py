"""Pinned inputs and fixed evaluation choices for PAN-X #98."""

from dataclasses import dataclass

DATASET_ID = "unimelb-nlp/wikiann"
DATASET_REVISION = "f0a3be6dc5564c0cc4150bb660144800a1f539d4"
DATASET_SPLIT = "test"
SEED = 0
BATCH_SIZE = 8
GLINER_THRESHOLD = 0.5
GLINER_ENTITY_LABELS = ("city", "country", "location")
SPACY_LOCATION_LABELS = frozenset({"FAC", "GPE", "LOC"})
LOCATION_LABEL = "LOC"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Pinned model and its known language/training-data provenance."""

    key: str
    model_id: str
    revision: str | None
    documented_languages: tuple[str, ...] | None
    coverage_note: str
    training_data_note: str
    overlap_note: str
    model_card_url: str
    batch_size: int = BATCH_SIZE


SPACY_MODEL_LANGUAGES = ("en",)
XLMR_FINETUNED_LANGUAGES = (
    "ar",
    "de",
    "en",
    "es",
    "fr",
    "it",
    "lv",
    "nl",
    "pt",
    "zh",
)

MODELS = (
    ModelSpec(
        key="spacy_en",
        model_id="en_core_web_sm",
        revision="3.8.0",
        documented_languages=SPACY_MODEL_LANGUAGES,
        coverage_note="English-only upstream reference model.",
        training_data_note=(
            "spaCy English small model 3.8.0; its model metadata identifies "
            "OntoNotes 5 as its training source."
        ),
        overlap_note=(
            "The model metadata does not report WikiANN training. Possible "
            "text overlap with Wikipedia was not audited."
        ),
        model_card_url=("https://spacy.io/models/en#en_core_web_sm-accuracy"),
    ),
    ModelSpec(
        key="gliner2_multi",
        model_id="fastino/gliner2.5-multi-v1",
        revision="2ca71aafb3446d9014e1c55c7ff51c9bc7209c47",
        documented_languages=None,
        coverage_note=(
            "Model card describes the checkpoint as multilingual but does not "
            "publish an exhaustive language list."
        ),
        training_data_note=(
            "The pinned model card does not identify the training corpora."
        ),
        overlap_note=(
            "Whether training included WikiANN/PAN-X or overlapping examples "
            "is unknown from the pinned model card."
        ),
        model_card_url=("https://huggingface.co/fastino/gliner2.5-multi-v1"),
        batch_size=1,
    ),
    ModelSpec(
        key="xlmr_ner_hrl",
        model_id="Davlan/xlm-roberta-base-ner-hrl",
        revision="253f557bd8249b8515114cfd7f71974fe5fa4d2f",
        documented_languages=XLMR_FINETUNED_LANGUAGES,
        coverage_note=(
            "Fine-tuned for Arabic, German, English, Spanish, French, Italian, "
            "Latvian, Dutch, Portuguese, and Chinese. Other evaluated languages "
            "are cross-lingual transfer."
        ),
        training_data_note=(
            "Its pinned model card lists ANERcorp, CoNLL 2002/2003, Europeana "
            "Newspapers, Italian I-CAB, Latvian NER, Paramopama/Second HAREM, "
            "and MSRA for those ten languages."
        ),
        overlap_note=(
            "The model card does not list WikiANN/PAN-X among its fine-tuning "
            "corpora. Pretraining/text-level overlap was not audited."
        ),
        model_card_url=("https://huggingface.co/Davlan/xlm-roberta-base-ner-hrl"),
    ),
)
