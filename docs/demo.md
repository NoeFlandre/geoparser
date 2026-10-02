# Demo

The demo shows every populated place in the novel *Around the World in Eighty Days* by Jules Verne. The library extracted 74 places from the 37 chapters and mapped them. The size of a marker shows how often the novel mentions the place. Hover over a marker to read the passages where the place appears.

<iframe src="_static/map.html" width="100%" height="550" frameborder="0"></iframe>

The book is from [Project Gutenberg](https://www.gutenberg.org/ebooks/103). The demo uses the library as the documentation describes it. The sections below explain the geoparsing parts. The other parts download the book, split it into chapters, and draw the map.

## What the Pipeline Does Here

The demo splits the novel into its 37 chapters. It parses each chapter as one document. It then aggregates the resolved places from all the chapters. Two choices are important. Both are different from the defaults:

``` python
from geoparser import Geoparser
from geoparser.modules import SentenceTransformerResolver, SpacyRecognizer

# A transformer model finds more place names in literary prose
recognizer = SpacyRecognizer(model_name="en_core_web_trf")

# A higher threshold: precision matters more than recall on a map
resolver = SentenceTransformerResolver(min_similarity=0.7)

geoparser = Geoparser(recognizer=recognizer, resolver=resolver)
```

The demo uses `en_core_web_trf` and not the default `en_core_web_sm`. Narrative prose of the nineteenth century is not like the news text that trained the small model. The small model misses many more names in it. The demo uses `min_similarity=0.7` and not `0.6`. A wrong marker on a map does more damage than a missing marker. A mistake is visible and gives wrong information. An omission is only absent. The code does not name the gazetteer. `SentenceTransformerResolver` uses GeoNames unless you give a different gazetteer.

The `en_core_web_trf` pipeline needs the `spacy-curated-transformers` plugin. To install the pinned compatible line, use `pip install "spacy-curated-transformers>=0.3.1,<1"`. The plugin has no release for Python 3.14 yet. On that interpreter, use a non-transformer model for the language that you need. For English, use `en_core_web_lg`.

This is the general method to tune a pipeline. The defaults are a good start. The correct values depend on your material. They also depend on which type of error costs more for you.

## Parsing and Aggregating

Each chapter goes in as a document. The results come back in the same order:

``` python
chapter_texts = [chapter["text"] for chapter in chapters]
documents = geoparser.parse(chapter_texts)
```

Then the code groups the mentions by the place that they resolved to. Two details in this step are good to copy:

``` python
from collections import defaultdict

places = defaultdict(lambda: {"location": None, "mentions": [], "count": 0})

for document, chapter in zip(documents, chapters):
    for toponym in document.toponyms:
        if toponym.location is None:
            continue

        # Populated places only — skip regions, seas, and mountains
        if toponym.location.data.get("feature_class") != "P":
            continue

        # Group by identifier, never by name
        entry = places[toponym.location.identifier]
        entry["location"] = toponym.location
        entry["count"] += 1

        # Keep the surrounding text, for the hover popups
        start = max(0, toponym.start - 30)
        end = min(len(document.text), toponym.end + 30)
        entry["mentions"].append((chapter["number"], document.text[start:end]))
```

**Group by** `identifier`, **not by name.** This is the important detail. Different places can have the same name. GeoNames has 122 places with the name Paris. If you group by `data["name"]`, you merge different places into one marker.

**Filter on** `feature_class` to keep the map readable. GeoNames classifies each feature. `"P"` means a populated place. Without the filter, "Europe", "the Atlantic", and "the Rocky Mountains" each become a single point. This is misleading. The full list of classes is in [querying gazetteers](guides/gazetteers.md).

The code keeps the character offsets. This makes the popups possible. The library gives you positions in the original text. To get the passage around a mention, you need only one slice.

The rest is an ordinary plot. Read `latitude` and `longitude` from the `data` of each feature. Set the size of the markers with `count`. Give the data to a plotting library. The notebook uses Plotly.

## Run It Yourself

The complete notebook is in the repository at [demo/demo.ipynb](https://github.com/NoeFlandre/geoparser/blob/main/demo/demo.ipynb). It downloads the book and splits the chapters. It runs the pipeline and builds the map above.

To run it in your own environment, you need the library and the `geonames` gazetteer. [Installation](installation.md) describes them. You also need two extra packages:

``` bash
pip install jupyter plotly
```

``` bash
geoparser install geonames
```

``` bash
jupyter lab demo/demo.ipynb
```

The parse takes a few minutes. The transformer recognizer is slow on a CPU, and the book has 37 chapters. A GPU makes the parse much faster. Read [Using a GPU](installation.md#using-a-gpu).

You can also run the notebook in Docker. Docker builds the demo image locally from this checkout and its lockfile. The demo image does not contain the gazetteer. You install the gazetteer once into a persistent volume. The runtime image shares this volume:

``` bash
export JUPYTER_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe())')"
docker compose --profile demo build demo
docker compose --profile install run --rm install
docker compose --profile demo up demo
```

Then open `http://localhost:8888/lab/tree/demo.ipynb`. Enter the value of `JUPYTER_TOKEN`. For details, read `demo/README.md` in the repository.
