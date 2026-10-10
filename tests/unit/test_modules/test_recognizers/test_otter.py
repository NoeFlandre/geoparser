"""Offline contracts for Otter's prepared-model adapter."""

from types import SimpleNamespace

import pytest

from geoparser.modules.recognizers.otter import OtterRecognizer, PredictionOutputError


class CharacterTokenizer:
    """A deterministic tokenizer with one token per Python character."""

    model_max_length = 512

    def __call__(self, texts, **kwargs):
        assert kwargs == {
            "padding": False,
            "truncation": False,
            "add_special_tokens": True,
        }
        return {"input_ids": [[0, *range(len(text)), 1] for text in texts]}


class PreparedModel:
    """A ready model double; this test never imports checkpoint code."""

    def __init__(self, predictions, *, architecture="cross_encoder"):
        self.config = SimpleNamespace(
            architecture=architecture,
            max_seq_length=1024,
            max_span_length=30,
            prediction_threshold=0.5 if architecture == "cross_encoder" else 0.2,
        )
        self.type_config = SimpleNamespace(max_position_embeddings=512)
        self._tokenizer = CharacterTokenizer()
        self._token_tokenizer = CharacterTokenizer()
        self._type_tokenizer = CharacterTokenizer()
        self.predictions = predictions
        self.calls = []

    @staticmethod
    def build_prompt(labels):
        return "[LABEL] " + " [LABEL] ".join(labels) + " [SEP] "

    def predict(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        return self.predictions


def entity(text, start, end, *, label="city", score=0.9):
    return {"text": text, "start": start, "end": end, "label": label, "score": score}


@pytest.mark.unit
@pytest.mark.parametrize("architecture", ["cross_encoder", "bi_encoder"])
def test_preserves_unicode_offsets_duplicates_and_empty_document_positions(
    architecture,
):
    model = PreparedModel(
        [
            [entity("Paris", 9, 14), entity("Paris", 2, 7), entity("Paris", 2, 7)],
            [entity("Zürich", 4, 10)],
            [entity("Paris", 2, 7)],
        ],
        architecture=architecture,
    )
    recognizer = OtterRecognizer(model, source="local-test-snapshot", batch_size=2)
    texts = ["", "😀 Paris, Paris.", " \n", "e\u0301: Zürich", "😀 Paris, Paris."]

    assert recognizer.predict(texts) == [
        [],
        [(2, 7), (9, 14)],
        [],
        [(4, 10)],
        [(2, 7)],
    ]
    assert model.calls == [
        (
            ["😀 Paris, Paris.", "e\u0301: Zürich", "😀 Paris, Paris."],
            {
                "labels": ["city", "country", "location"],
                "threshold": 0.5 if architecture == "cross_encoder" else 0.2,
                "batch_size": 2,
                "max_seq_length": 1024,
            },
        )
    ]
    assert recognizer.last_raw_predictions == [
        [],
        model.predictions[0],
        [],
        model.predictions[1],
        model.predictions[2],
    ]


@pytest.mark.unit
def test_predict_batch_preserves_the_set_based_benchmark_contract():
    recognizer = OtterRecognizer(
        PreparedModel([[entity("Paris", 0, 5)]]), source="local-test-snapshot"
    )

    assert recognizer.predict_batch(["Paris"]) == [{(0, 5)}]
    assert recognizer.predict([]) == []
    assert recognizer.last_raw_predictions == []


@pytest.mark.unit
def test_blank_documents_preserve_alignment_and_reset_previous_evidence():
    model = PreparedModel([[entity("Paris", 0, 5)]])
    recognizer = OtterRecognizer(model, source="test")
    assert recognizer.last_raw_predictions == []
    assert recognizer.predict(["Paris"]) == [[(0, 5)]]
    assert model.calls == [
        (
            ["Paris"],
            {
                "labels": ["city", "country", "location"],
                "threshold": 0.5,
                "batch_size": 8,
                "max_seq_length": 1024,
            },
        )
    ]

    assert (
        recognizer.predict(["", " \n\t"]),
        recognizer.last_raw_predictions,
        len(model.calls),
    ) == ([[], []], [[], []], 1)


@pytest.mark.unit
def test_only_scores_strictly_above_the_threshold_survive():
    model = PreparedModel(
        [[entity("Paris", 0, 5, score=0.5), entity("Tokyo", 6, 11, score=0.6)]]
    )
    recognizer = OtterRecognizer(model, source="test", threshold=0.5)

    assert recognizer.predict(["Paris Tokyo"]) == [[(6, 11)]]
    assert recognizer.last_raw_predictions == [
        [entity("Paris", 0, 5, score=0.5), entity("Tokyo", 6, 11, score=0.6)]
    ]


@pytest.mark.unit
def test_configuration_is_serializable_and_identifies_effective_settings():
    model = PreparedModel([[entity("Paris", 0, 5, label="capital")]])
    recognizer = OtterRecognizer(
        model, source="reviewed-snapshot@123", entity_types=["capital"], batch_size=3
    )

    assert recognizer.predict(["Paris"]) == [[(0, 5)]]
    assert recognizer.config == {
        "architecture": "cross_encoder",
        "batch_size": 3,
        "entity_types": ["capital"],
        "max_seq_length": 1024,
        "max_span_length": 30,
        "provenance": "caller-supplied",
        "source": "reviewed-snapshot@123",
        "threshold": 0.5,
        "type_max_seq_length": None,
    }
    same = OtterRecognizer(
        model,
        source="reviewed-snapshot@123",
        entity_types=("capital",),
        batch_size=3,
        threshold=0.5,
    )
    other = OtterRecognizer(
        model, source="different-snapshot", entity_types=["capital"]
    )
    assert recognizer.id == same.id
    assert recognizer.id != other.id


@pytest.mark.unit
@pytest.mark.parametrize("limit", ["max_seq_length", "max_span_length"])
def test_model_limits_change_the_recorded_recognizer_identity(limit):
    model = PreparedModel([[entity("Paris", 0, 5)]])
    original = OtterRecognizer(model, source="same-artifact")
    setattr(model.config, limit, getattr(model.config, limit) + 1)
    changed = OtterRecognizer(model, source="same-artifact")

    assert changed.predict(["Paris"]) == [[(0, 5)]]
    assert changed.id != original.id
    assert changed.config[limit] == original.config[limit] + 1


@pytest.mark.unit
def test_bi_label_limit_changes_the_recorded_recognizer_identity():
    model = PreparedModel([[entity("Paris", 0, 5)]], architecture="bi_encoder")
    original = OtterRecognizer(model, source="same-artifact")
    model._type_tokenizer.model_max_length = 128
    changed = OtterRecognizer(model, source="same-artifact")

    assert changed.predict(["Paris"]) == [[(0, 5)]]
    assert original.config["type_max_seq_length"] == 512
    assert changed.config["type_max_seq_length"] == 128
    assert changed.id != original.id


@pytest.mark.unit
@pytest.mark.parametrize("threshold", [0, 1])
def test_probability_endpoints_are_valid_and_use_strict_comparison(threshold):
    model = PreparedModel(
        [[entity("Paris", 0, 5, score=0), entity("Tokyo", 6, 11, score=1)]]
    )
    recognizer = OtterRecognizer(model, source="test", threshold=threshold)

    expected = [[(6, 11)]] if threshold == 0 else [[]]
    assert recognizer.predict(["Paris Tokyo"]) == expected
    assert recognizer.last_raw_predictions == [
        [entity("Paris", 0, 5, score=0), entity("Tokyo", 6, 11, score=1)]
    ]


@pytest.mark.unit
@pytest.mark.parametrize(
    "threshold", [-0.1, 1.1, float("nan"), float("inf"), "0.5", True]
)
def test_rejects_invalid_thresholds(threshold):
    with pytest.raises((TypeError, ValueError)) as captured:
        OtterRecognizer(PreparedModel([]), source="test", threshold=threshold)
    assert str(captured.value) == "threshold must be a finite number between 0 and 1."


@pytest.mark.unit
@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True])
def test_rejects_invalid_batch_sizes(batch_size):
    with pytest.raises(ValueError) as captured:
        OtterRecognizer(PreparedModel([]), source="test", batch_size=batch_size)
    assert str(captured.value) == "batch_size must be a positive integer."


@pytest.mark.unit
@pytest.mark.parametrize("source", ["", " \n", None, 3])
def test_requires_a_caller_supplied_source_identity(source):
    with pytest.raises(ValueError) as captured:
        OtterRecognizer(PreparedModel([]), source=source)
    assert str(captured.value) == "source must be a nonblank string."


@pytest.mark.unit
@pytest.mark.parametrize(
    ("labels", "diagnostic"),
    [
        ([], "entity_types must contain at least one distinct entity name."),
        (
            ["city", "city"],
            "entity_types must contain at least one distinct entity name.",
        ),
        ([" "], "entity type must be a nonblank string."),
        ([3], "entity type must be a nonblank string."),
        ("city", "entity_types must be a sequence of distinct entity names."),
    ],
)
def test_rejects_empty_ambiguous_or_invalid_entity_types(labels, diagnostic):
    with pytest.raises((TypeError, ValueError)) as captured:
        OtterRecognizer(PreparedModel([]), source="test", entity_types=labels)
    assert str(captured.value) == diagnostic


@pytest.mark.unit
def test_rejects_unknown_architecture():
    with pytest.raises(ValueError) as captured:
        OtterRecognizer(PreparedModel([], architecture="other"), source="test")
    assert str(captured.value) == "Unsupported Otter architecture: 'other'."


@pytest.mark.unit
@pytest.mark.parametrize(
    ("architecture", "cache"),
    [
        ("cross_encoder", "_tokenizer"),
        ("bi_encoder", "_token_tokenizer"),
        ("bi_encoder", "_type_tokenizer"),
    ],
)
def test_requires_ready_tokenizers_without_invoking_the_lazy_properties(
    architecture, cache
):
    class LazyModel(PreparedModel):
        @property
        def tokenizer(self):
            pytest.fail("The lazy tokenizer loader must never run.")

        @property
        def token_tokenizer(self):
            pytest.fail("The lazy tokenizer loader must never run.")

        @property
        def type_tokenizer(self):
            pytest.fail("The lazy tokenizer loader must never run.")

    model = LazyModel([], architecture=architecture)
    setattr(model, cache, None)
    with pytest.raises(TypeError) as captured:
        OtterRecognizer(model, source="test")
    assert str(captured.value) == (
        f"Prepare Otter's {cache} tokenizer from the reviewed local "
        "checkpoint before constructing the recognizer."
    )


@pytest.mark.unit
@pytest.mark.parametrize("limit", ["max_seq_length", "max_span_length"])
def test_rejects_invalid_model_limits(limit):
    model = PreparedModel([])
    setattr(model.config, limit, 0)
    with pytest.raises(ValueError) as captured:
        OtterRecognizer(model, source="test")
    assert str(captured.value) == f"{limit} must be a positive integer."


@pytest.mark.unit
@pytest.mark.parametrize("architecture", ["cross_encoder", "bi_encoder"])
def test_preflights_exact_limit_and_reports_original_overlength_index(architecture):
    model = PreparedModel([[entity("Paris", 0, 5)]], architecture=architecture)
    # Cross input is '[LABEL] city [SEP] Paris', which has 24 characters,
    # plus two special tokens. Bi input needs seven tokens.
    model.config.max_seq_length = 26 if architecture == "cross_encoder" else 7
    recognizer = OtterRecognizer(model, source="test", entity_types=["city"])
    assert recognizer.predict(["Paris"]) == [[(0, 5)]]

    with pytest.raises(ValueError) as captured:
        recognizer.predict(["", "Paris", "Paris!"])
    expected = (
        "Otter document at index 2 has 27 tokens, exceeding the limit of 26; truncation is not supported."
        if architecture == "cross_encoder"
        else "Otter document at index 2 has 8 tokens, exceeding the limit of 7; truncation is not supported."
    )
    assert (
        str(captured.value),
        len(model.calls),
        recognizer.last_raw_predictions,
    ) == (expected, 1, [])


@pytest.mark.unit
def test_prompt_is_counted_with_the_text_in_the_same_tokenization():
    model = PreparedModel([[entity("Paris", 0, 5)]])

    class BoundaryTokenizer(CharacterTokenizer):
        def __call__(self, texts, **kwargs):
            assert texts == ["[LABEL] city [SEP] Paris"]
            return super().__call__(texts, **kwargs)

    model._tokenizer = BoundaryTokenizer()
    recognizer = OtterRecognizer(model, source="test", entity_types=["city"])
    assert recognizer.predict(["Paris"]) == [[(0, 5)]]


@pytest.mark.unit
@pytest.mark.parametrize("limiting_attribute", ["tokenizer", "encoder"])
def test_bi_encoder_rejects_overlength_labels_before_inference(limiting_attribute):
    model = PreparedModel([[entity("Paris", 0, 5)]], architecture="bi_encoder")
    if limiting_attribute == "tokenizer":
        model._type_tokenizer.model_max_length = 6
    else:
        model.type_config.max_position_embeddings = 6
    recognizer = OtterRecognizer(model, source="test", entity_types=["city"])
    assert recognizer.predict(["Paris"]) == [[(0, 5)]]

    recognizer = OtterRecognizer(model, source="test", entity_types=["country"])
    with pytest.raises(ValueError) as captured:
        recognizer.predict(["Paris"])
    assert str(captured.value) == (
        "Otter entity type at index 0 has 9 tokens, "
        "exceeding the limit of 6; truncation is not supported."
    )
    assert len(model.calls) == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    ("texts", "diagnostic"),
    [
        ("Paris", "texts must be a list of strings."),
        (["Paris", None], "Every document in texts must be a string."),
        ([3], "Every document in texts must be a string."),
    ],
)
def test_rejects_non_string_documents(texts, diagnostic):
    recognizer = OtterRecognizer(PreparedModel([]), source="test")
    with pytest.raises(TypeError) as captured:
        recognizer.predict(texts)
    assert str(captured.value) == diagnostic


@pytest.mark.unit
@pytest.mark.parametrize("raw", [None, {}, [], [[], []]])
def test_bad_batch_shape_retains_the_entire_raw_response(raw):
    recognizer = OtterRecognizer(PreparedModel(raw), source="test")
    with pytest.raises(PredictionOutputError) as captured:
        recognizer.predict(["", "Paris"])

    assert (
        str(captured.value)
        == "Otter must return one result list for each input index [1]."
    )
    assert captured.value.document_index is None
    assert captured.value.raw_output is raw
    assert recognizer.last_raw_predictions == []


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw", "diagnostic"),
    [
        (None, "Each document result must be a list of entities."),
        ({}, "Each document result must be a list of entities."),
        ([None], "Each entity must be a dictionary."),
        ([{}], "'start'"),
        (
            [entity("Paris", True, 5)],
            "Entity offsets must be integers, excluding booleans.",
        ),
        (
            [entity("Paris", 0, False)],
            "Entity offsets must be integers, excluding booleans.",
        ),
        (
            [entity("Paris", 0.0, 5)],
            "Entity offsets must be integers, excluding booleans.",
        ),
        (
            [entity("Paris", 0, 5.0)],
            "Entity offsets must be integers, excluding booleans.",
        ),
        (
            [entity("Paris", -1, 5)],
            "Entity offsets must select a nonempty span in the original text.",
        ),
        (
            [entity("", 0, 0)],
            "Entity offsets must select a nonempty span in the original text.",
        ),
        (
            [entity("", 5, 1)],
            "Entity offsets must select a nonempty span in the original text.",
        ),
        (
            [entity("Paris", 0, 6)],
            "Entity offsets must select a nonempty span in the original text.",
        ),
        (
            [entity("Lyon", 0, 5)],
            "Entity text does not match its offsets in the original text.",
        ),
        ([entity("Paris", 0, 5, label="person")], "Entity label was not requested."),
        (
            [entity("Paris", 0, 5, score=float("nan"))],
            "entity score must be a finite number between 0 and 1.",
        ),
        (
            [entity("Paris", 0, 5, score=float("inf"))],
            "entity score must be a finite number between 0 and 1.",
        ),
        (
            [entity("Paris", 0, 5, score=-0.1)],
            "entity score must be a finite number between 0 and 1.",
        ),
        (
            [entity("Paris", 0, 5, score=1.1)],
            "entity score must be a finite number between 0 and 1.",
        ),
        (
            [entity("Paris", 0, 5, score="0.9")],
            "entity score must be a finite number between 0 and 1.",
        ),
        (
            [entity("Paris", 0, 5, score=True)],
            "entity score must be a finite number between 0 and 1.",
        ),
    ],
)
def test_invalid_entities_retain_original_index_and_raw_document(raw, diagnostic):
    recognizer = OtterRecognizer(PreparedModel([raw]), source="test")

    with pytest.raises(PredictionOutputError, match="document at index 1") as captured:
        recognizer.predict(["", "Paris"])

    assert (
        str(captured.value)
        == f"Invalid Otter output for document at index 1: {diagnostic}"
    )
    assert captured.value.document_index == 1
    assert captured.value.raw_output is raw
    assert recognizer.last_raw_predictions == [[], raw]


@pytest.mark.unit
def test_invalid_below_threshold_spans_are_not_hidden_as_empty_results():
    raw = [entity("London", 0, 5, score=0.1)]
    recognizer = OtterRecognizer(PreparedModel([raw]), source="test")
    with pytest.raises(PredictionOutputError, match="original text") as captured:
        recognizer.predict(["Paris"])
    assert captured.value.raw_output == [entity("London", 0, 5, score=0.1)]


@pytest.mark.unit
def test_rejects_overlapping_spans_but_keeps_adjacent_ones():
    model = PreparedModel([[entity("Paris", 0, 5), entity("Tokyo", 5, 10)]])
    recognizer = OtterRecognizer(model, source="test")
    assert recognizer.predict(["ParisTokyo"]) == [[(0, 5), (5, 10)]]

    model.predictions = [[entity("Paris", 0, 5), entity("ParisTokyo", 0, 10)]]
    with pytest.raises(PredictionOutputError, match="overlapping") as captured:
        recognizer.predict(["ParisTokyo"])
    assert str(captured.value) == (
        "Invalid Otter output for document at index 0: "
        "Otter returned distinct overlapping entity spans."
    )
    assert captured.value.raw_output == [
        entity("Paris", 0, 5),
        entity("ParisTokyo", 0, 10),
    ]


@pytest.mark.unit
def test_model_failures_propagate_and_clear_old_raw_evidence(monkeypatch):
    model = PreparedModel([[entity("Paris", 0, 5)]])
    recognizer = OtterRecognizer(model, source="test")
    assert recognizer.predict(["Paris"]) == [[(0, 5)]]

    def fail(*args, **kwargs):
        raise RuntimeError("device failed")

    monkeypatch.setattr(model, "predict", fail)
    with pytest.raises(RuntimeError, match="device failed"):
        recognizer.predict(["Paris"])
    assert recognizer.last_raw_predictions == []


@pytest.mark.unit
def test_batch_size_one_accepts_and_preserves_multiple_documents():
    model = PreparedModel([[entity("Paris", 0, 5)], [entity("Tokyo", 0, 5)]])
    recognizer = OtterRecognizer(model, source="test", batch_size=1)

    assert recognizer.predict(["Paris", "Tokyo"]) == [[(0, 5)], [(0, 5)]]
    assert model.calls == [
        (
            ["Paris", "Tokyo"],
            {
                "labels": ["city", "country", "location"],
                "threshold": 0.5,
                "batch_size": 1,
                "max_seq_length": 1024,
            },
        )
    ]


@pytest.mark.unit
@pytest.mark.parametrize("length", [0, 3])
def test_malformed_tokenizer_row_count_is_rejected_before_inference(length):
    class MalformedTokenizer(CharacterTokenizer):
        def __call__(self, texts, **kwargs):
            return {"input_ids": [[0, 1]] * length}

    model = PreparedModel([[entity("Paris", 0, 5)], [entity("Tokyo", 0, 5)]])
    recognizer = OtterRecognizer(model, source="test", batch_size=2)
    assert recognizer.predict(["Paris", "Tokyo"]) == [[(0, 5)], [(0, 5)]]

    model._tokenizer = MalformedTokenizer()
    malformed = OtterRecognizer(model, source="test", batch_size=2)
    with pytest.raises(ValueError, match="zip"):
        malformed.predict(["Paris", "Tokyo"])
    assert model.calls == [
        (
            ["Paris", "Tokyo"],
            {
                "labels": ["city", "country", "location"],
                "threshold": 0.5,
                "batch_size": 2,
                "max_seq_length": 1024,
            },
        )
    ]


@pytest.mark.unit
def test_preflight_processes_each_tokenizer_batch_once_and_preserves_order():
    class RecordingTokenizer(CharacterTokenizer):
        def __init__(self):
            self.batches = []

        def __call__(self, texts, **kwargs):
            self.batches.append(texts)
            return super().__call__(texts, **kwargs)

    model = PreparedModel(
        [[entity("Paris", 0, 5)], [entity("Tokyo", 0, 5)], [entity("Lima", 0, 4)]],
        architecture="bi_encoder",
    )
    tokenizer = RecordingTokenizer()
    model._token_tokenizer = tokenizer
    recognizer = OtterRecognizer(model, source="test", batch_size=2)

    assert recognizer.predict(["Paris", "Tokyo", "Lima"]) == [
        [(0, 5)],
        [(0, 5)],
        [(0, 4)],
    ]
    assert tokenizer.batches == [["Paris", "Tokyo"], ["Lima"]]


@pytest.mark.unit
def test_first_overlength_document_cannot_skip_preflight():
    model = PreparedModel([[entity("Paris", 0, 5)]], architecture="bi_encoder")
    model.config.max_seq_length = 7
    recognizer = OtterRecognizer(model, source="test")
    assert recognizer.predict(["Paris"]) == [[(0, 5)]]

    with pytest.raises(ValueError) as captured:
        recognizer.predict(["Paris!"])
    assert str(captured.value) == (
        "Otter document at index 0 has 8 tokens, "
        "exceeding the limit of 7; truncation is not supported."
    )
    assert len(model.calls) == 1


@pytest.mark.unit
@pytest.mark.parametrize("limit", ["tokenizer", "encoder"])
def test_invalid_bi_label_limits_name_the_failing_component(limit):
    model = PreparedModel([], architecture="bi_encoder")
    if limit == "tokenizer":
        model._type_tokenizer.model_max_length = 0
    else:
        model.type_config.max_position_embeddings = 0

    with pytest.raises(ValueError) as captured:
        OtterRecognizer(model, source="test")
    assert str(captured.value) == f"type {limit} limit must be a positive integer."
