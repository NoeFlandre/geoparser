# Annotating Data

Two tasks need annotated data: training a module on your own material and measuring how well a pipeline performs. Both tasks need the same thing: texts in which a person marked the place names and the places to which they refer.

The library has a web application that makes this data. This guide describes what the application does, how to run it, and how its output goes back into a project.

> [!NOTE]
> The annotator is a standalone tool. It is not an integrated part of the library. The developers built it for an earlier architecture and adapted it so that annotation continues to work after the redesign. It has its own database and does not share projects with the rest of the library. Annotations move between the two as a JSON file. Depend on this file. The developers will probably rework or replace the application.

## Do You Need It?

You do not need annotations to geoparse text. The built-in modules are pre-trained and work immediately.

You need annotations if you want to do one of these tasks:

- **Fine-tune a module** for a language, a period, or a domain that the defaults handle poorly. Refer to [training](training.md).
- **Measure performance** on your own material. Do not trust the figures from the corpus of another person.
- **Build a gold-standard corpus** as a research output.

You do not need the application if you already have annotations in another format. `create_references()` and `create_referents()` take spans and identifiers directly. Refer to [projects](projects.md). Use the annotator to make annotations by hand.

## Starting the Application

``` bash
geoparser annotator
```

The command starts a local web server and opens `http://127.0.0.1:5000/` in your browser. Stop it with `Ctrl-C`.

By default, the server listens only on `127.0.0.1`. Thus only your own machine can reach it. The command has these options:

| Option | Default | Effect |
|---|---|---|
| `--host HOST` | `127.0.0.1` | The interface to bind to. |
| `--port PORT` | `5000` | The port to listen on. |
| `--no-browser` | off | Does not open a browser, for example on a headless server. |
| `--reload` | off | Restarts the server when source files change (for development). |

> [!WARNING]
> The annotator has no authentication. With `--host 0.0.0.0`, it listens on every network interface. Anyone who can reach your machine on that port can read and edit your annotations. Use this option only on a trusted network.

The annotator offers **GeoNames** and **SwissNames3D**, if you installed them. If you did not install one yet, install it first. Refer to [installation](../installation.md).

## The Workflow

**1. Start a session.** A session is one annotation job. It has a set of documents, a gazetteer that you selected, and the annotations made so far. Upload one or more plain text files and select the gazetteer for the place links. The gazetteer is fixed for the session, because an identifier has meaning only for its gazetteer.

**2. The recognizer proposes spans.** When you open a document for the first time, the recognizer runs on it and marks the place names that it finds. Thus you correct proposals and you do not mark each name by hand. This occurs automatically, and you cannot skip it in the interface. The recognizer uses the spaCy model that you selected when you started the session.

**3. Correct the place names.** Select a span of text to add a toponym that the recognizer missed. Remove the toponyms that the recognizer marked in error. Spans must not overlap. The annotator rejects an annotation that overlaps an existing toponym, because overlapping spans have no consistent interpretation in later steps.

**4. Link each name to a place.** For a marked toponym, the annotator searches the gazetteer. It shows the candidates with their attributes and coordinates, so that you can tell one Springfield from another. Select the correct candidate. You can leave a toponym unlinked if you cannot resolve it, for example a fictional place or a place that is not in the gazetteer. This records that a person also could not resolve it.

**5. Download the annotations.** The session exports as a JSON file. The annotator also stores sessions in its own database. Thus you can close the browser and continue later.

Two session settings are important. *Auto-close annotation modal* moves you to the next item after each choice. It is faster when you know the interface. *One sense per discourse* applies your choice to the other occurrences of the same name in the same document. It assumes that a repeated name in one text usually means the same place. This is often true, but not always. Examine the result.

## The Annotation Format

The export has this format:

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

`start` and `end` are character offsets into `text`. `end` is exclusive. `loc_id` is the identifier in the named gazetteer. For a toponym that you did not link, it is an empty string or `null`.

The format is plain JSON. Thus you can also generate it from an existing annotated corpus and not use the application. Note these two points if you do this. First, the project reads only `start` and `end` of each toponym. The `"text"` field is for human readers. The project calculates the stored text again from the offsets. Second, the document `text` must be exactly the same as the text in your project, because the project matches annotations to documents by text equality.

## Loading Annotations into a Project

Put the exported file into a project with `load_annotations()`:

``` python
from geoparser import Project

project = Project("annotated_corpus")

project.load_annotations(
    path="annotations_5f3a.json",
    tag="gold",
    create_documents=True,
)

for document in project.get_documents(tag="gold"):
    print(f"{len(document.toponyms)} annotated toponyms")
```

Two arguments set the behavior.

`create_documents=True` adds the annotated texts to the project as documents. Use it when the project is empty. Set it to `False` when the documents are already in the project, for example from an earlier `create_documents()`, and the annotations must attach to them. In that case, the texts must match exactly. If they do not, the project skips the annotation and gives no error.

`tag` is the name of this annotation set in the project. Use it to refer to the annotations afterwards. It must be the same tag that you use later for training or evaluation. A mismatch is the most frequent reason that training reports no examples. Always use different tags for human annotations and model output. Then you can compare `get_documents(tag="gold")` and `get_documents(tag="predicted")`.

``` python
# Human annotations
project.load_annotations(path="gold.json", tag="gold", create_documents=True)

# Model output over the same documents
project.run_recognizer(SpacyRecognizer(), tag="predicted")
project.run_resolver(SentenceTransformerResolver(), tag="predicted")

gold = project.get_documents(tag="gold")
predicted = project.get_documents(tag="predicted")
```

Now the annotations are ordinary project data. You can use them to train a module (refer to [training](training.md)). You can also use them as the reference set for an evaluation.
