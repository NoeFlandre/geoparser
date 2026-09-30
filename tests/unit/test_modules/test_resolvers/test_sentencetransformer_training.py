"""SentenceTransformerResolver training data (resolvers/_training.py)."""

from unittest.mock import Mock, patch

import pytest

from geoparser.modules.resolvers import _training
from geoparser.modules.resolvers._training import TrainingMixin


@pytest.mark.unit
class TestSentenceTransformerResolverPrepareTrainingData:
    """Test SentenceTransformerResolver _prepare_training_data method."""

    def test_search_uses_the_training_candidate_limit(self):
        """Training data uses an explicit stable limit for each exact search."""
        from geoparser.modules.resolvers.sentencetransformer import (
            SentenceTransformerResolver,
        )

        resolver = object.__new__(SentenceTransformerResolver)
        candidate = Mock(identifier="123")
        resolver._extract_context = Mock(return_value="Paris is beautiful")
        resolver._candidate_description = Mock(return_value="Paris (city)")
        resolver._search_candidates = Mock(return_value=[candidate])

        resolver._prepare_training_data(
            ["Paris is beautiful."], [[(0, 5)]], [[("geonames", "123")]]
        )

        resolver._search_candidates.assert_called_once_with(
            "Paris", "exact", tiers=1, limit=10000
        )


@pytest.mark.unit
class TestTrainingMixinFit:
    """Test the public model fine-tuning workflow without external training."""

    def test_trains_from_prepared_examples_and_saves_the_model(
        self, monkeypatch, tmp_path
    ):
        transformer = Mock()

        class Resolver(TrainingMixin):
            def __init__(self):
                self.transformer = transformer

        resolver = Resolver()
        texts = ["Paris"]
        references = [[(0, 5)]]
        referents = [[("geonames", "123")]]
        training_data = {
            "sentence1": [f"context {index}" for index in range(220)],
            "sentence2": [f"candidate {index}" for index in range(220)],
            "label": [float(index % 2) for index in range(220)],
        }
        resolver._prepare_training_data = Mock(return_value=training_data)
        dataset = object()
        dataset_factory = Mock(return_value=dataset)
        monkeypatch.setattr(_training.Dataset, "from_dict", dataset_factory)
        loss = object()
        loss_factory = Mock(return_value=loss)
        monkeypatch.setattr(_training, "ContrastiveLoss", loss_factory)
        args = object()
        args_factory = Mock(return_value=args)
        monkeypatch.setattr(
            _training, "SentenceTransformerTrainingArguments", args_factory
        )
        trainer = Mock()
        trainer_factory = Mock(return_value=trainer)
        monkeypatch.setattr(_training, "SentenceTransformerTrainer", trainer_factory)

        output_path = tmp_path / "fine-tuned-model"
        resolver.fit(
            texts,
            references,
            referents,
            output_path,
            epochs=2,
            batch_size=2,
            learning_rate=3e-5,
            warmup_ratio=0.15,
            save_strategy="steps",
        )

        resolver._prepare_training_data.assert_called_once_with(
            texts, references, referents
        )
        dataset_factory.assert_called_once_with(training_data)
        loss_factory.assert_called_once_with(resolver.transformer)
        args_factory.assert_called_once_with(
            output_dir=str(output_path),
            num_train_epochs=2,
            per_device_train_batch_size=2,
            learning_rate=3e-5,
            warmup_ratio=0.15,
            save_strategy="steps",
            logging_strategy="steps",
            logging_steps=11,
            eval_strategy="no",
            save_total_limit=2,
            load_best_model_at_end=False,
        )
        logging_steps = args_factory.call_args.kwargs["logging_steps"]
        assert isinstance(logging_steps, int) and not isinstance(logging_steps, bool)
        trainer_factory.assert_called_once_with(
            model=resolver.transformer,
            args=args,
            train_dataset=dataset,
            loss=loss,
        )
        trainer.train.assert_called_once_with()
        transformer.save_pretrained.assert_called_once_with(str(output_path))

    def test_uses_default_training_settings(self, monkeypatch, tmp_path):
        transformer = Mock()

        class Resolver(TrainingMixin):
            def __init__(self):
                self.transformer = transformer

        resolver = Resolver()
        training_data = {
            "sentence1": [f"context {index}" for index in range(9)],
            "sentence2": [f"candidate {index}" for index in range(9)],
            "label": [float(index % 2) for index in range(9)],
        }
        resolver._prepare_training_data = Mock(return_value=training_data)
        dataset = object()
        monkeypatch.setattr(_training.Dataset, "from_dict", Mock(return_value=dataset))
        monkeypatch.setattr(_training, "ContrastiveLoss", Mock())
        args_factory = Mock(return_value=object())
        monkeypatch.setattr(
            _training, "SentenceTransformerTrainingArguments", args_factory
        )
        trainer = Mock()
        monkeypatch.setattr(
            _training, "SentenceTransformerTrainer", Mock(return_value=trainer)
        )
        output_path = tmp_path / "fine-tuned-model"

        resolver.fit(["Paris"], [[(0, 5)]], [[("geonames", "123")]], output_path)

        assert args_factory.call_args.kwargs == {
            "output_dir": str(output_path),
            "num_train_epochs": 1,
            "per_device_train_batch_size": 8,
            "learning_rate": 2e-5,
            "warmup_ratio": 0.1,
            "save_strategy": "epoch",
            "logging_strategy": "steps",
            "logging_steps": 1,
            "eval_strategy": "no",
            "save_total_limit": 2,
            "load_best_model_at_end": False,
        }
        trainer.train.assert_called_once_with()
        transformer.save_pretrained.assert_called_once_with(str(output_path))

    def test_rejects_empty_training_data(self, tmp_path):
        class Resolver(TrainingMixin):
            def __init__(self):
                self.transformer = Mock()

        resolver = Resolver()
        resolver._prepare_training_data = Mock(return_value={"sentence1": []})

        with pytest.raises(ValueError) as error:
            resolver.fit([], [], [], tmp_path / "unused")

        assert str(error.value).strip() not in {"", "None"}

    @patch("geoparser.modules.resolvers.sentencetransformer.load_spacy_model")
    @patch(
        "geoparser.modules.resolvers.sentencetransformer.AutoTokenizer.from_pretrained"
    )
    @patch("geoparser.modules.resolvers.sentencetransformer.SentenceTransformer")
    @patch("geoparser.modules.resolvers.sentencetransformer.Gazetteer")
    def test_creates_training_examples_from_referents(
        self, mock_gazetteer_class, mock_transformer, mock_tokenizer, mock_spacy_load
    ):
        """Test that _prepare_training_data creates training examples from referents."""
        # Arrange
        from geoparser.modules.resolvers.sentencetransformer import (
            SentenceTransformerResolver,
        )

        resolver = SentenceTransformerResolver(gazetteer_name="geonames")

        # Mock gazetteer search to return candidates
        mock_gazetteer_instance = mock_gazetteer_class.return_value
        mock_candidate1 = Mock()
        mock_candidate1.identifier = "123"
        mock_candidate1.data = {"name": "Paris", "feature_name": "city"}
        mock_candidate2 = Mock()
        mock_candidate2.identifier = "456"
        mock_candidate2.data = {"name": "Paris", "feature_name": "region"}
        mock_gazetteer_instance.search.return_value = [mock_candidate1, mock_candidate2]

        # Mock _extract_context
        with patch.object(
            resolver, "_extract_context", return_value="Paris is beautiful"
        ):
            texts = ["Paris is beautiful."]
            references = [[(0, 5)]]  # "Paris"
            referents = [[("geonames", "123")]]  # Matches candidate1

            # Act
            training_data = resolver._prepare_training_data(
                texts, references, referents
            )

            # Assert
            assert (
                set(training_data),
                all(
                    len(training_data[field]) > 0
                    for field in ("sentence1", "sentence2", "label")
                ),
            ) == ({"sentence1", "sentence2", "label"}, True)

    @patch("geoparser.modules.resolvers.sentencetransformer.load_spacy_model")
    @patch(
        "geoparser.modules.resolvers.sentencetransformer.AutoTokenizer.from_pretrained"
    )
    @patch("geoparser.modules.resolvers.sentencetransformer.SentenceTransformer")
    @patch("geoparser.modules.resolvers.sentencetransformer.Gazetteer")
    def test_creates_positive_and_negative_examples(
        self, mock_gazetteer_class, mock_transformer, mock_tokenizer, mock_spacy_load
    ):
        """Test that _prepare_training_data creates both positive and negative examples."""
        # Arrange
        from geoparser.modules.resolvers.sentencetransformer import (
            SentenceTransformerResolver,
        )

        resolver = SentenceTransformerResolver(gazetteer_name="geonames")

        # Mock gazetteer search to return multiple candidates
        mock_gazetteer_instance = mock_gazetteer_class.return_value
        mock_candidate1 = Mock()
        mock_candidate1.identifier = "123"
        mock_candidate1.data = {"name": "Paris", "feature_name": "city"}
        mock_candidate2 = Mock()
        mock_candidate2.identifier = "456"
        mock_candidate2.data = {"name": "Paris", "feature_name": "region"}
        mock_gazetteer_instance.search.return_value = [mock_candidate1, mock_candidate2]

        with patch.object(
            resolver, "_extract_context", return_value="Paris is beautiful"
        ):
            texts = ["Paris is beautiful."]
            references = [[(0, 5)]]
            referents = [[("geonames", "123")]]  # Only matches candidate1

            # Act
            training_data = resolver._prepare_training_data(
                texts, references, referents
            )

            # Assert
            # Should have 2 examples: 1 positive (label=1) and 1 negative (label=0)
            assert len(training_data["label"]) == 2
            assert 1 in training_data["label"]  # Positive example
            assert 0 in training_data["label"]  # Negative example

    @patch("geoparser.modules.resolvers.sentencetransformer.load_spacy_model")
    @patch(
        "geoparser.modules.resolvers.sentencetransformer.AutoTokenizer.from_pretrained"
    )
    @patch("geoparser.modules.resolvers.sentencetransformer.SentenceTransformer")
    @patch("geoparser.modules.resolvers.sentencetransformer.Gazetteer")
    def test_handles_multiple_references_in_document(
        self, mock_gazetteer_class, mock_transformer, mock_tokenizer, mock_spacy_load
    ):
        """Test that _prepare_training_data handles multiple references per document."""
        # Arrange
        from geoparser.modules.resolvers.sentencetransformer import (
            SentenceTransformerResolver,
        )

        resolver = SentenceTransformerResolver(gazetteer_name="geonames")

        # Mock gazetteer search
        mock_gazetteer_instance = mock_gazetteer_class.return_value
        mock_candidate = Mock()
        mock_candidate.identifier = "123"
        mock_candidate.data = {"name": "City", "feature_name": "city"}
        mock_gazetteer_instance.search.return_value = [mock_candidate]

        with patch.object(resolver, "_extract_context", return_value="Context"):
            texts = ["Paris and London"]
            references = [[(0, 5), (10, 16)]]  # "Paris", "London"
            referents = [[("geonames", "123"), ("geonames", "123")]]

            # Act
            training_data = resolver._prepare_training_data(
                texts, references, referents
            )

            # Assert
            # Should have examples for both references
            assert len(training_data["sentence1"]) >= 2

    @patch("geoparser.modules.resolvers.sentencetransformer.load_spacy_model")
    @patch(
        "geoparser.modules.resolvers.sentencetransformer.AutoTokenizer.from_pretrained"
    )
    @patch("geoparser.modules.resolvers.sentencetransformer.SentenceTransformer")
    @patch("geoparser.modules.resolvers.sentencetransformer.Gazetteer")
    def test_handles_multiple_documents(
        self, mock_gazetteer_class, mock_transformer, mock_tokenizer, mock_spacy_load
    ):
        """Test that _prepare_training_data handles multiple documents."""
        # Arrange
        from geoparser.modules.resolvers.sentencetransformer import (
            SentenceTransformerResolver,
        )

        resolver = SentenceTransformerResolver(gazetteer_name="geonames")

        # Mock gazetteer search
        mock_gazetteer_instance = mock_gazetteer_class.return_value
        mock_candidate = Mock()
        mock_candidate.identifier = "123"
        mock_candidate.data = {"name": "City", "feature_name": "city"}
        mock_gazetteer_instance.search.return_value = [mock_candidate]

        with patch.object(resolver, "_extract_context", return_value="Context"):
            texts = ["Paris is beautiful.", "London is historic."]
            references = [[(0, 5)], [(0, 6)]]  # "Paris", "London"
            referents = [[("geonames", "123")], [("geonames", "123")]]

            # Act
            training_data = resolver._prepare_training_data(
                texts, references, referents
            )

            # Assert
            # Should have examples from both documents
            assert len(training_data["sentence1"]) >= 2

    @patch("geoparser.modules.resolvers.sentencetransformer.load_spacy_model")
    @patch(
        "geoparser.modules.resolvers.sentencetransformer.AutoTokenizer.from_pretrained"
    )
    @patch("geoparser.modules.resolvers.sentencetransformer.SentenceTransformer")
    @patch("geoparser.modules.resolvers.sentencetransformer.Gazetteer")
    def test_extracts_context_for_each_reference(
        self, mock_gazetteer_class, mock_transformer, mock_tokenizer, mock_spacy_load
    ):
        """Test that _prepare_training_data extracts context for each reference."""
        # Arrange
        from geoparser.modules.resolvers.sentencetransformer import (
            SentenceTransformerResolver,
        )

        resolver = SentenceTransformerResolver(gazetteer_name="geonames")

        # Mock gazetteer search
        mock_gazetteer_instance = mock_gazetteer_class.return_value
        mock_candidate = Mock()
        mock_candidate.identifier = "123"
        mock_candidate.data = {"name": "City", "feature_name": "city"}
        mock_gazetteer_instance.search.return_value = [mock_candidate]

        with patch.object(
            resolver, "_extract_context", return_value="Extracted context"
        ) as mock_extract:
            texts = ["Paris is beautiful."]
            references = [[(0, 5)]]
            referents = [[("geonames", "123")]]

            # Act
            training_data = resolver._prepare_training_data(
                texts, references, referents
            )

            # Assert
            # _extract_context should have been called
            mock_extract.assert_called()
            # All sentence1 entries should be the extracted context
            assert all(s1 == "Extracted context" for s1 in training_data["sentence1"])

    @patch("geoparser.modules.resolvers.sentencetransformer.load_spacy_model")
    @patch(
        "geoparser.modules.resolvers.sentencetransformer.AutoTokenizer.from_pretrained"
    )
    @patch("geoparser.modules.resolvers.sentencetransformer.SentenceTransformer")
    @patch("geoparser.modules.resolvers.sentencetransformer.Gazetteer")
    def test_generates_descriptions_for_candidates(
        self, mock_gazetteer_class, mock_transformer, mock_tokenizer, mock_spacy_load
    ):
        """Test that _prepare_training_data generates descriptions for candidates."""
        # Arrange
        from geoparser.modules.resolvers.sentencetransformer import (
            SentenceTransformerResolver,
        )

        resolver = SentenceTransformerResolver(gazetteer_name="geonames")

        # Mock gazetteer search
        mock_gazetteer_instance = mock_gazetteer_class.return_value
        mock_candidate = Mock()
        mock_candidate.identifier = "123"
        mock_candidate.data = {"name": "Paris", "feature_name": "city"}
        mock_gazetteer_instance.search.return_value = [mock_candidate]

        with (
            patch.object(resolver, "_extract_context", return_value="Context"),
            patch.object(
                resolver, "_generate_description", return_value="Paris (city)"
            ) as mock_generate,
        ):
            texts = ["Paris is beautiful."]
            references = [[(0, 5)]]
            referents = [[("geonames", "123")]]

            # Act
            training_data = resolver._prepare_training_data(
                texts, references, referents
            )

            # Assert
            # _generate_description should have been called
            mock_generate.assert_called()
            # All sentence2 entries should be the generated description
            assert all(s2 == "Paris (city)" for s2 in training_data["sentence2"])

    @patch("geoparser.modules.resolvers.sentencetransformer.load_spacy_model")
    @patch(
        "geoparser.modules.resolvers.sentencetransformer.AutoTokenizer.from_pretrained"
    )
    @patch("geoparser.modules.resolvers.sentencetransformer.SentenceTransformer")
    @patch("geoparser.modules.resolvers.sentencetransformer.Gazetteer")
    def test_returns_correct_data_structure(
        self, mock_gazetteer_class, mock_transformer, mock_tokenizer, mock_spacy_load
    ):
        """Test that _prepare_training_data returns correct data structure."""
        # Arrange
        from geoparser.modules.resolvers.sentencetransformer import (
            SentenceTransformerResolver,
        )

        resolver = SentenceTransformerResolver(gazetteer_name="geonames")

        # Mock gazetteer search
        mock_gazetteer_instance = mock_gazetteer_class.return_value
        mock_candidate = Mock()
        mock_candidate.identifier = "123"
        mock_candidate.data = {"name": "City", "feature_name": "city"}
        mock_gazetteer_instance.search.return_value = [mock_candidate]

        with patch.object(resolver, "_extract_context", return_value="Context"):
            texts = ["Paris"]
            references = [[(0, 5)]]
            referents = [[("geonames", "123")]]

            # Act
            training_data = resolver._prepare_training_data(
                texts, references, referents
            )

            # Assert
            # Should be a dict with three keys
            assert (
                isinstance(training_data, dict),
                set(training_data),
                len(training_data["sentence1"])
                == len(training_data["sentence2"])
                == len(training_data["label"]),
                all(label in (0, 1) for label in training_data["label"]),
            ) == (True, {"sentence1", "sentence2", "label"}, True, True)
