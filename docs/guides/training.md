# Training Modules

This guide explains how to train modules and fine-tune them on annotated data. Training improves performance for specific domains, languages, or use cases.

## Overview

The Irchel Geoparser can train and fine-tune modules on annotated data. You can train any module that has a `fit()` method with the correct interface. Training improves performance on texts that are different from the data of the original training. It also adds support for new languages and for specialized geographic contexts.

The built-in `SpacyRecognizer` and `SentenceTransformerResolver` modules have a `fit()` method. It trains the underlying models on annotated examples. Training needs documents with ground-truth annotations. A recognizer needs the positions of place names in text. A resolver needs the positions of place names and also their correct links to gazetteer entries. If you do not have such annotations yet, refer to [annotating](annotating.md) to make them. Training uses the project-level methods (`project.train_recognizer()` and `project.train_resolver()`). These methods automatically collect the training data from the annotated documents in the project. Then they call the `fit()` method of the module.

## Training SpacyRecognizer

The `SpacyRecognizer` uses the named entity recognition framework of spaCy. Training fine-tunes an existing spaCy model, so that it recognizes place names better in your domain or language.

### Preparing Training Data

Training data is texts and the positions of the place names in those texts. You can make annotations by hand or load them from files:

``` python
from geoparser import Project

project = Project("training_corpus")

# Option 1: Create annotations manually
texts = ["The summit was held in Geneva."]

# The documents must exist before they can be annotated
project.create_documents(texts)

project.create_references(
    texts=texts,
    references=[[(23, 29)]],  # Position of "Geneva"
    tag="gold",
)

# Option 2: Load from JSON file
project.load_annotations(path="annotations.json", tag="gold", create_documents=True)
```

The tag that you select here is important. It is the name that you use later to refer to this set of annotations. Training reads its examples from one tag. Thus you must train on the tag under which you annotated. This guide uses `"gold"` in all examples.

> [!WARNING]
> `create_references()` and `create_referents()` annotate documents that are already in the project. They do not create the documents. They match annotations to documents by **exact text equality**. If the text of an annotation matches no stored document, the project skips the annotation and gives no error. Therefore call `create_documents()` first. Give exactly the same strings to both methods. `load_annotations()` is different. It has a `create_documents` flag. Thus Option 2 does not need a separate call.

Refer to the [projects](projects.md) guide for more information about annotations.

### Training the Recognizer

When the project has annotated documents, training a recognizer is simple:

``` python
from geoparser import Project
from geoparser.modules import SpacyRecognizer

project = Project("training_corpus")

# Create the recognizer to train
recognizer = SpacyRecognizer(model_name="en_core_web_sm")

# Train on documents tagged as "gold"
project.train_recognizer(
    recognizer,
    tag="gold",
    output_path="models/trained_recognizer",
    epochs=10,
    batch_size=8,
    dropout=0.1,
    learning_rate=0.001,
)
```

The `train_recognizer()` method retrieves all documents from the project that have reference annotations for the specified tag. It extracts the texts and the reference positions. Then it calls the `fit()` method of the recognizer, which does the training. The method saves the trained model to the output path that you specified.

> [!NOTE]
> Assume that the tag you train on has no annotations. This is most often because it is not the tag under which you annotated. Training then fails with `ValueError: No training examples found. Ensure documents contain reference annotations.` It does not train on nothing. First check the tag with `len(project.get_documents(tag="gold")[0].toponyms)`.

The training parameters control how the model learns. The `epochs` parameter sets how many times the training algorithm iterates over the dataset. More epochs can improve performance. But if you have little training data, they can cause overfitting. The `batch_size` parameter sets how many examples the model processes together in each training step. Larger batches give more stable gradients, but they need more memory. The `dropout` rate adds regularization. It randomly drops neural network connections during training, which helps to prevent overfitting. The `learning_rate` sets how fast the model changes its parameters during training.

### Using the Trained Model

After training, specify the path of the trained model when you create a recognizer:

``` python
from geoparser import Project
from geoparser.modules import SpacyRecognizer

# Load the trained model
recognizer = SpacyRecognizer(model_name="models/trained_recognizer")

project = Project("evaluation_corpus")
project.run_recognizer(recognizer)
```

You can use the trained recognizer in any workflow, in the same way as the pre-trained models. The model path becomes part of the configuration of the recognizer. Thus the project tracks the results of the trained model separately from the results of the base model.

## Training SentenceTransformerResolver

The `SentenceTransformerResolver` uses a SentenceTransformer model to calculate embeddings of contexts and location descriptions. Training fine-tunes this model. The model then better captures the semantic relationship between the way your texts mention places and the way the resolver must describe them to disambiguate them.

### Preparing Training Data

Training a resolver needs the positions of place names and their correct resolutions. Each place name must link to a specific feature in the gazetteer:

``` python
from geoparser import Project

project = Project("training_corpus")

# Option 1: Create annotations manually
texts = ["The summit was held in Geneva."]
references = [[(23, 29)]]

project.create_documents(texts)

# The place names themselves...
project.create_references(texts, references, tag="gold")

# ...and the gazetteer features they refer to
project.create_referents(
    texts,
    references,
    referents=[[("geonames", "2660646")]],  # the city of Geneva
    tag="gold",
)

# Option 2: Load from JSON file
project.load_annotations(path="annotations.json", tag="gold", create_documents=True)
```

You need both calls. `create_referents()` records which feature each place name refers to. It does not record the place names. If you use only this call, there is nothing to train on, and `train_resolver()` fails with `ValueError: No training examples found. Ensure documents contain references with referent annotations.` Give the same `references` to both calls. As for the recognizer, annotate and train under the same tag, and create the documents first.

Note that the referent is a specific gazetteer feature. It is not an abstract place. Thus check which feature you selected. GeoNames has different features for the city of Geneva (`2660646`) and the canton of the same name (`2660645`). For this sentence, the city is correct. If you train on the canton, the resolver learns the wrong association. Use `Gazetteer("geonames").find("2660646")` to confirm an identifier quickly before you use it.

Refer to the [projects](projects.md) guide for more information about annotations.

### Training the Resolver

Training a resolver through a project is similar to training a recognizer:

``` python
from geoparser import Project
from geoparser.modules import SentenceTransformerResolver

project = Project("training_corpus")

# Create the resolver to train
resolver = SentenceTransformerResolver(
    model_name="dguzh/geo-all-MiniLM-L6-v2", gazetteer_name="geonames"
)

# Train on documents tagged as "gold"
project.train_resolver(
    resolver,
    tag="gold",
    output_path="models/trained_resolver",
    epochs=1,
    batch_size=8,
    learning_rate=2e-5,
    warmup_ratio=0.1,
)
```

The `train_resolver()` method retrieves the documents that have reference annotations and referent annotations. For each reference that has a referent, it extracts the context and generates candidate descriptions from the gazetteer. Then it makes training examples that teach the model which candidate description matches the context.

The training parameters are slightly different from those of the recognizer. Transformer models usually need fewer epochs. One epoch is often sufficient for fine-tuning. Set the `learning_rate` low (2e-5 is a common default), so that you do not destroy the knowledge that the model already has. The `warmup_ratio` sets how slowly the learning rate increases from zero at the start of training. This helps to stabilize the training.

### Using the Trained Model

After training, specify the path of the trained resolver to use it:

``` python
from geoparser import Project
from geoparser.modules import SpacyRecognizer, SentenceTransformerResolver

recognizer = SpacyRecognizer()
resolver = SentenceTransformerResolver(
    model_name="models/trained_resolver", gazetteer_name="geonames"
)

project = Project("evaluation_corpus")
project.run_recognizer(recognizer)
project.run_resolver(resolver)
```

Remember that a trained transformer model is specific to the gazetteer that you used for training. A model that you trained with GeoNames does not work well with SwissNames3D, because the location descriptions have different formats and attributes. If you must support more than one gazetteer, train a separate model for each gazetteer.

## Evaluation

After training, evaluate your models on held-out test data that you did not use in training. Create a separate project with test annotations and run your trained modules on it:

``` python
from geoparser import Project
from geoparser.modules import SpacyRecognizer, SentenceTransformerResolver

# Load test data
test_project = Project("test_corpus")
test_project.load_annotations(
    path="test_annotations.json", tag="gold", create_documents=True
)

# Run trained modules
recognizer = SpacyRecognizer(model_name="models/trained_recognizer")
resolver = SentenceTransformerResolver(model_name="models/trained_resolver")

test_project.run_recognizer(recognizer, tag="predicted")
test_project.run_resolver(resolver, tag="predicted")

# Retrieve both gold and predicted results for comparison
gold_docs = test_project.get_documents(tag="gold")
pred_docs = test_project.get_documents(tag="predicted")

# Calculate evaluation metrics
# (You'll need to implement your own comparison logic)
gold_count = sum(len(doc.toponyms) for doc in gold_docs)
pred_count = sum(len(doc.toponyms) for doc in pred_docs)

print(f"Gold standard: {gold_count} toponyms")
print(f"Predictions: {pred_count} toponyms")
```

For a more thorough evaluation, calculate the precision, the recall, and the F1 score for recognition. Calculate accuracy metrics for resolution. To do this, align the predicted toponyms with the gold-standard annotations by position. Then check if the resolved locations match.
