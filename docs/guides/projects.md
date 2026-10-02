# Managing Projects

A project is a workspace that remembers. The project stores the documents that you add and the results that you make under a name. Thus you can return to them in a later session. You can also run different pipelines on the same corpus and keep all the result sets side by side.

## When to Use One

`Geoparser.parse()` is the correct tool for analysis that you do in one session. You give text and get results. Nothing is stored. Most work is of this kind. The [quickstart](../quickstart.md) describes it.

Use a project in these cases:

- **The corpus is so large that reprocessing it is a problem.** Parse it one time. Then query the results as often as you want.
- **You want to compare configurations.** Run two recognizers, three similarity thresholds, or a different gazetteer on the same documents. Keep each set of results separate. This makes the comparison meaningful.
- **You work with annotations.** Training and evaluation need human-annotated data. The project stores this data together with the model output, so that you can compare the two.
- **The work continues over more than one session.** The results remain when you close Python.

The cost is bookkeeping. You must name things and keep a record of what each name holds. A project records what the modules processed, which module processed it, and which configuration the module used. It files each result under a **tag**, which identifies the pipeline that made the result. You must understand tags well. A later section describes them. The [Project API](../api/project.md) reference gives the full signatures of all methods.

> [!NOTE]
> `Geoparser` is the interface that we expect to remain stable. The database-backed project layer will probably become one option of several. It will not be the foundation of everything. Where either one is suitable, use `Geoparser`.

## Creating and Loading Projects

A name identifies a project. When you create a `Project` instance, it creates a new project in the database or loads the existing project with that name:

``` python
from geoparser import Project

# Create a new project or load an existing one
project = Project("research_corpus")
```

A project stays in the database until you delete it. You can close your Python session and return later. Load the same project with the same name.

## Adding Documents

After you create a project, add documents to it with the `create_documents()` method. The method takes a list of text strings, one for each document:

``` python
from geoparser import Project

project = Project("news_analysis")

# Add a single document
project.create_documents(["The summit took place in Geneva."])

# Add multiple documents
texts = [
    "London hosted the Olympic Games in 2012.",
    "The conference was held in Barcelona.",
    "Researchers gathered in Vienna to discuss the findings.",
]
project.create_documents(texts)
```

You can add more documents to the same project at any time. They accumulate in the collection of the project.

The project stores each document under a unique identifier. `create_documents()` returns these identifiers in the order of the texts that you gave. Assume that the texts come from a collection that you already have, such as a database of articles or a set of files. Store the returned IDs with those records. Then you can match the parsed results to the originals:

``` python
articles = load_articles()  # your own records, whatever shape they have

document_ids = project.create_documents([article["body"] for article in articles])

for article, document_id in zip(articles, document_ids):
    article["document_id"] = document_id
```

Give those IDs to `get_documents()` to retrieve exactly those documents, in the order that you asked for:

``` python
# Results for one specific article
documents = project.get_documents(ids=articles[0]["document_id"])

# Or for a subset of your material, in your own order
recent = [article["document_id"] for article in articles if article["year"] >= 2020]
documents = project.get_documents(ids=recent)
```

With this method, you can work with parts of a corpus separately. For example, you can compare how a pipeline performs on older material and on newer material. You do not need to reprocess anything.

## Running Processing Modules

When the project has documents, you can run recognition modules and resolution modules to find and resolve place names. The project controls the execution and stores the results:

``` python
from geoparser import Project
from geoparser.modules import SpacyRecognizer, SentenceTransformerResolver

project = Project("news_analysis")

# Create module instances
recognizer = SpacyRecognizer()
resolver = SentenceTransformerResolver()

# Run the recognizer to identify place names
project.run_recognizer(recognizer)

# Run the resolver to link place names to locations
project.run_resolver(resolver)
```

The `run_recognizer()` method processes all documents in the project that this recognizer has not yet processed. In the same way, `run_resolver()` processes all references that this resolver has not yet resolved. Thus you can safely call these methods many times. They handle only the items that are not yet processed.

## Retrieving Results

After you run modules on your project, retrieve the processed documents with the `get_documents()` method:

``` python
from geoparser import Project

project = Project("news_analysis")

# Get all documents with their results
documents = project.get_documents()

# Or only specific ones, using the IDs from create_documents()
documents = project.get_documents(ids=document_ids)

# Access the results
for doc in documents:
    print(f"Document: {doc.text}")
    for toponym in doc.toponyms:
        print(f"  - {toponym.text}", end="")
        location = toponym.location
        if location:
            data = location.data
            print(
                f" → {data.get('name')} ({data.get('latitude')}, {data.get('longitude')})"
            )
        else:
            print(" (unresolved)")
    print()
```

The `toponyms` property of a document returns only the references that the recognizer of the current tag found. In the same way, the `location` property of a reference returns the feature that the resolver of that tag selected. If you do not specify a tag, the system uses the `"latest"` tag. This tag points to the recognizer and the resolver that ran most recently with the `"latest"` tag. If you run modules with a different tag, the `"latest"` tag does not change. It continues to point to the modules that ran last with `"latest"`. Tags control this filtering. The next section explains tags.

## Understanding Tags

Tags let you keep many result sets in the same project. When you run a recognizer or a resolver, you can specify a tag for the results. The default tag is `"latest"`.

``` python
from geoparser import Project
from geoparser.modules import SpacyRecognizer, SentenceTransformerResolver

project = Project("comparison_study")

# Add documents
project.create_documents(
    [
        "Scientists met in Copenhagen to discuss climate change.",
        "The treaty was signed in Kyoto.",
    ]
)

# Run with default spaCy model and tag as "baseline"
recognizer_baseline = SpacyRecognizer(model_name="en_core_web_sm")
resolver_baseline = SentenceTransformerResolver()

project.run_recognizer(recognizer_baseline, tag="baseline")
project.run_resolver(resolver_baseline, tag="baseline")

# Run with transformer-based model and tag as "transformer"
recognizer_trf = SpacyRecognizer(model_name="en_core_web_trf")
resolver_trf = SentenceTransformerResolver(model_name="dguzh/geo-all-distilroberta-v1")

project.run_recognizer(recognizer_trf, tag="transformer")
project.run_resolver(resolver_trf, tag="transformer")

# Compare results from different configurations
baseline_docs = project.get_documents(tag="baseline")
transformer_docs = project.get_documents(tag="transformer")

print("Baseline Results:")
for doc in baseline_docs:
    print(f"  Found {len(doc.toponyms)} toponyms")

print("\nTransformer Results:")
for doc in transformer_docs:
    print(f"  Found {len(doc.toponyms)} toponyms")
```

The `en_core_web_trf` pipeline needs the `spacy-curated-transformers` plugin. Install the compatible pinned line with `pip install "spacy-curated-transformers>=0.3.1,<1"`. The plugin has no Python 3.14 release yet. On that interpreter, use a non-transformer model for the requested language, for example `en_core_web_lg` for English.

Tags let you run many recognition strategies and resolution strategies on the same corpus and compare their performance. Each tag keeps its own pointer to the recognizer and the resolver that ran. When you call `get_documents(tag="baseline")`, you see only the results of the modules for that tag. A tag represents a complete processing pipeline. It does not represent a single module. When you use tags, always run a recognizer and a resolver with the same tag. The resolution results depend on the recognition results. If you use different tags for recognition and resolution in the same pipeline, the results are invalid or incomplete.

## Comparative Workflows

Projects are very useful when you compare processing configurations. You can examine how different recognizers, resolvers, or parameters change the geoparsing results:

``` python
from geoparser import Project
from geoparser.modules import SpacyRecognizer, SentenceTransformerResolver

project = Project("parameter_study")
project.create_documents(
    [
        "The delegation traveled from Brussels to Amsterdam.",
        "Trade routes connected Venice, Constantinople, and Alexandria.",
    ]
)

# Test different similarity thresholds
recognizer = SpacyRecognizer()

for threshold in [0.5, 0.6, 0.7, 0.8]:
    resolver = SentenceTransformerResolver(min_similarity=threshold)
    tag = f"threshold_{threshold}"

    project.run_recognizer(recognizer, tag=tag)
    project.run_resolver(resolver, tag=tag)

    docs = project.get_documents(tag=tag)
    resolved_count = sum(
        1 for doc in docs for toponym in doc.toponyms if toponym.location is not None
    )
    print(f"Threshold {threshold}: {resolved_count} resolved toponyms")
```

With this method, you can do reproducible experiments. You control exactly which processing configuration made which results, and you can document it.

## Working with Annotations

Projects can manage manually annotated data. You need this data to train custom models and to evaluate system performance. Create annotations in code with the `create_references()` and `create_referents()` methods:

``` python
from geoparser import Project

project = Project("manual_annotations")

texts = ["Paris is the capital of France."]
project.create_documents(texts)

# Create references (identified place names)
references = [[(0, 5), (24, 30)]]  # "Paris" and "France"
project.create_references(texts, references, tag="manual")

# Create referents (resolved locations)
referents = [[("geonames", "2988507"), ("geonames", "3017382")]]
project.create_referents(texts, references, referents, tag="manual")
```

The positions are character offsets into the document text. `end` is exclusive, as `text[start:end]` expects. Check the offsets before you annotate in bulk. `"Paris is the capital of France."[24:30]` is `'France'`, but `[23:29]` gives `' Franc'`. The project calculates the text of a reference again from the offsets. It does not use any text that you give. Thus an off-by-one error is stored without a warning.

These methods store the annotations in the database. They use internal recognizer and resolver modules. You can then use the annotations for training or evaluation.

You can also load annotations from JSON files that you exported from annotation tools:

``` python
from geoparser import Project

project = Project("annotated_corpus")

# Load annotations from a JSON file
# Set create_documents=True if documents aren't already in the project
project.load_annotations(
    path="annotations.json", tag="annotator", create_documents=True
)

# Access the manually annotated toponyms
documents = project.get_documents(tag="annotator")
for doc in documents:
    print(f"Document: {doc.text}")
    print(f"  Annotated toponyms: {len(doc.toponyms)}")
```

The JSON file must have this structure:

``` json
{
    "gazetteer": "geonames",
    "documents": [
        {
            "text": "Paris is the capital of France.",
            "toponyms": [
                {
                    "start": 0,
                    "end": 5,
                    "text": "Paris",
                    "loc_id": "2988507"
                },
                {
                    "start": 24,
                    "end": 30,
                    "text": "France",
                    "loc_id": "3017382"
                }
            ]
        }
    ]
}
```

The `loc_id` field must contain the identifier from the specified gazetteer. For toponyms that you did not link to locations, use an empty string or null. The project reads only `start` and `end` of each toponym. The `"text"` field is documentation for the reader. The project calculates the stored reference text again from the offsets.

The `annotator <annotating>` exports this format. Thus you can load annotations that you made there without conversion.

## Deleting Projects

When you finish with a project and you want to free database space, delete it with the `delete()` method:

``` python
from geoparser import Project

project = Project("temporary_analysis")
# ... work with the project ...

# Delete the project and all associated data
project.delete()
```

This removes the project and all its documents, references, and referents from the database. The deletion is permanent. You cannot undo it. Be careful when you use this method.
