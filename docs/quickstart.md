# Quickstart

This page is one worked example. It builds up one step at a time. At the end, you will have parsed a text and read the results. You will also know how to handle names that do not resolve and how to change the pipeline.

Before you start, complete the [installation](installation.md). You must have the package and a gazetteer. The examples use `geonames`.

## Building a Geoparser

A geoparser has two modules. You must provide both of them explicitly. A **recognizer** finds place names in text. A **resolver** links the names to a gazetteer.

``` python
from geoparser import Geoparser
from geoparser.modules import GLiNER2Recognizer, PriorResolver

geoparser = Geoparser(
    recognizer=GLiNER2Recognizer(),
    resolver=PriorResolver(gazetteer_name="geonames"),
)
```

Both arguments are required. There are no defaults. If you omit one, the library raises a `TypeError`. To skip a stage, pass `None`. For example, `resolver=None` gives you recognition only. You must do this explicitly.

The first time that you run this code, it downloads the models that the two modules need. This happens once. GLiNER2 recognizes places in many languages. The default MiniLM resolver was fine-tuned on English news. It works best with GeoNames. `PriorResolver` uses a small population prior to break ties between close candidates. For measured comparisons and alternatives, read the [benchmark results](https://huggingface.co/datasets/NoeFlandre/geoparser-benchmark-results) and [Changing the Pipeline](#changing-the-pipeline).

## Parsing a Text

Give a text to `parse()`. It returns a **document**. The document has the text and the place names found in it.

``` python
text = (
    "Heavy rain caused flooding across northern England this week. The worst "
    "damage was reported in Manchester and Leeds, where rivers burst their banks "
    "overnight. Emergency services in Sheffield said they had received hundreds "
    "of calls."
)

document = geoparser.parse(text)
print(f"{len(document.toponyms)} toponyms found")
```

``` text
4 toponyms found
```

The example uses a paragraph and not one sentence. This is deliberate. The resolver reads the words around a name to decide between places with the same name. It works much better on a few sentences of connected prose than on one short sentence. Read [Why Context Matters](quickstart.md#why-context-matters).

## Reading the Results

Each toponym has its text and its position in the text:

``` python
for toponym in document.toponyms:
    print(f"{toponym.text!r} at characters {toponym.start}-{toponym.end}")
```

``` text
'England' at characters 43-50
'Manchester' at characters 95-105
'Leeds' at characters 110-115
'Sheffield' at characters 181-190
```

The offsets are indexes into `document.text`. `document.text[toponym.start:toponym.end]` is the toponym. A wider slice gives you the text around it.

The geographic information is in `toponym.location`:

``` python
for toponym in document.toponyms:
    location = toponym.location
    print(f"{toponym.text}:")
    print(f"  Name:        {location.data.get('name')}")
    print(f"  Type:        {location.data.get('feature_name')}")
    print(
        f"  Coordinates: {location.data.get('latitude')}, {location.data.get('longitude')}"
    )
```

``` text
England:
  Name:        England
  Type:        first-order administrative division
  Coordinates: 52.16045, -0.70312
Manchester:
  Name:        Manchester
  Type:        seat of a second-order administrative division
  Coordinates: 53.48095, -2.23743
Leeds:
  Name:        Leeds
  Type:        seat of a second-order administrative division
  Coordinates: 53.79648, -1.54785
Sheffield:
  Name:        Sheffield
  Type:        seat of a second-order administrative division
  Coordinates: 53.38297, -1.4659
```

`location` is a **feature**. It is one entry in the gazetteer. Its `data` is a dictionary of the information that the gazetteer records. This information is different between gazetteers. It can also be different between sources in one gazetteer. Therefore, read it with `.get()`. Do not use `data["name"]`. The page [working with results](guides/results.md) describes the three objects and their attributes in full.

The name of a place in the gazetteer can be different from the name in the text. If you ask the same pipeline about Vienna, the feature comes back as `Wien`. To group or count places, use `location.identifier`. This is the stable id of the gazetteer for that place. For Manchester it is `2643123`. Names are ambiguous. Identifiers are not.

A feature also has a `geometry`. It is a Shapely object. You can map it or measure it:

``` python
point = document.toponyms[1].location.geometry
print(point, point.x, point.y)
```

``` text
POINT (-2.23743 53.48095) -2.23743 53.48095
```

## Handling What Is Missing

The code above works only because every name resolved and every attribute was present. The library does not guarantee this. Write this version in your own code:

``` python
text = (
    "The expedition began in Cape Town, where the crew loaded supplies before "
    "heading south. After three weeks at sea they reached South Georgia, a "
    "remote island in the southern Atlantic, and from there they pushed on "
    "toward Antarctica."
)

document = geoparser.parse(text)

for toponym in document.toponyms:
    if toponym.location is None:
        print(f"{toponym.text}: not resolved")
    else:
        print(f"{toponym.text}: {toponym.location.data.get('name')}")
```

``` text
Cape Town: Cape Town
South Georgia: not resolved
Atlantic: Atlantic Ocean
Antarctica: Antarctica
```

`location` is `None` when the recognizer found a name and the resolver did not resolve it. This occurred for South Georgia in the example. There are two possible reasons. The place can be absent from the gazetteer. Or no candidate passed the similarity threshold of the resolver. In both cases, check for `None` before you read attributes.

Individual attributes can also be missing. This is a separate problem:

``` python
for toponym in document.toponyms:
    if toponym.location:
        print(f"{toponym.text}: {toponym.location.data.get('country_name')!r}")
```

``` text
Cape Town: 'South Africa'
Atlantic: None
Antarctica: None
```

An ocean and a continent are not in a country. Therefore `country_name` is absent for them. The same is true for `geometry`. It is `None` for places that a gazetteer records by name and does not locate. Read attributes with `.get()`. Check `location` for `None`. This is the normal way to work with results.

## Parsing Several Texts

`parse()` accepts a list. It processes the list as a batch. This is much faster than a loop:

``` python
texts = [
    "Researchers in Nairobi and Mombasa collected samples along the Kenyan coast.",
    "The festival moved from Salzburg to Vienna after a dispute over funding.",
    "The conference was held in Zurich, with satellite events in Geneva and Basel.",
]

documents = geoparser.parse(texts)

for i, document in enumerate(documents, start=1):
    print(f"Document {i}:")
    for toponym in document.toponyms:
        name = toponym.location.data.get("name") if toponym.location else "unresolved"
        print(f"  {toponym.text} -> {name}")
```

``` text
Document 1:
  Nairobi -> Nairobi
  Mombasa -> Mombasa
Document 2:
  Salzburg -> Salzburg
  Vienna -> Wien
Document 3:
  Zurich -> Zürich
  Geneva -> Geneva
  Basel -> Basel
```

The result has the same shape as the input. If you pass a string, you get one document. If you pass a list, you get a list of documents **in the same order**. You use this order to relate the results to the source of the texts:

``` python
for text, document in zip(texts, documents):
    ...
```

For larger work or work that you keep for a long time, identifiers are a more robust link than position. Read [managing projects](guides/projects.md).

## Why Context Matters

The quantity of context in a text has a large effect on the results. Look at it directly. This is a short sentence:

``` python
document = geoparser.parse("She flew from Paris to Tokyo last spring.")
```

``` text
'Paris'  ->  not resolved
'Tokyo'  ->  Takeo, Japan
```

Tokyo became Takeo, a town in Kyushu. Paris did not resolve. Now parse the same two names with words around them:

``` python
document = geoparser.parse(
    "She flew from Paris to Tokyo last spring, changing planes twice. The trip "
    "was her first visit to Japan, and she spent a week in the city before "
    "returning to France."
)
```

``` text
'Paris'   ->  Paris, France
'Tokyo'   ->  Tokyo, Japan
'Japan'   ->  Japan
'France'  ->  Republic of France
```

Only the surrounding words changed. The resolver compares the context of a name with the descriptions of the candidate places. If a name has no context, the resolver has little basis to choose. The recognizer also depends on context. It can miss names in a bare sentence that it finds in a paragraph.

These are the practical consequences:

- **Parse whole paragraphs or documents.** Do not parse isolated sentences or bare lists of place names. If your data is a list of names, for example a spreadsheet column, a geoparser is the wrong tool. Use a plain gazetteer lookup ([querying gazetteers](guides/gazetteers.md)).
- **Compare the results with the text.** Both errors above are silent. The library raises no error. `Takeo` looks like a plausible answer until you compare it with the sentence.
- **The defaults are tuned for English news prose.** The default models trained on it. For historical, literary, or non-English material, expect worse results. Read the next section.

## Changing the Pipeline

Both modules have parameters. Use them to adapt the pipeline to your material:

``` python
from geoparser import Geoparser
from geoparser.modules import GLiNER2Recognizer, JinaResolver

geoparser = Geoparser(
    # Ask for the kinds of place your material actually contains
    recognizer=GLiNER2Recognizer(
        entity_types=["city", "country", "mountain", "valley"],
    ),
    # A different gazetteer, and a lower confidence threshold
    resolver=JinaResolver(
        gazetteer_name="swissnames3d",
        min_similarity=0.5,
    ),
)

document = geoparser.parse("Zurich is the largest city in Switzerland.")
```

Two parameters have the largest effect on the quantity of names that the library recognizes and resolves:

- `entity_types` on the recognizer. The default is `["city", "country", "location"]`. GLiNER2 matches these types zero-shot. They are ordinary words. They are not a fixed schema. Name the types of place that your material contains. This usually helps more than a bigger model. No later stage can recover a name that the recognizer did not find.
- `min_similarity` on the resolver. The default is `0.6`. It is the confidence that the embedding stage must have before a candidate is worth a rerank. Decrease it to resolve more names and to get more mistakes. Increase it for the opposite result. The default is calibrated for English news text against GeoNames. Other material generally needs a lower value.

The page [configuring modules](guides/modules.md) describes every parameter and the second pre-trained resolver model. It also describes how to write your own modules.

## Keeping the Results

`parse()` discards its work after it returns. To keep the work, use this code:

``` python
document = geoparser.parse("Berlin is the capital of Germany.", save=True)
```

``` text
Results saved under project name: a1b2c3d4
```

Use the printed name to get the results later with `Project("a1b2c3d4")`. If you know in advance that you want to keep the results, create a project with a name that you choose. Read [managing projects](guides/projects.md).

## From the Command Line

The same pipeline runs without Python code. `geoparser parse` reads each file that you name as one document. It reads standard input if you give `-` or no file. It writes one JSON record for each document to standard output:

``` bash
echo "The worst damage was reported in Manchester and Leeds." \
  | geoparser parse - --gazetteer geonames --recognizer gliner --resolver jina
```

``` text
{"source": "-", "text": "The worst damage ...", "toponyms": [{"start": 33, "end": 43, "text": "Manchester", "gazetteer": "geonames", "identifier": "2643123", "geometry": {"type": "Point", "coordinates": [...]}}, ...]}
```

`--recognizer` is `spacy` (the default) or `gliner`. `--resolver` is `prior` (the default), `sentencetransformer`, or `jina`. `--recognizer-model` and `--model` select a different recognizer checkpoint or resolver checkpoint. `--format` selects `jsonl` (the default), `json` (one array), or `geojson` (a FeatureCollection of the linked toponyms, in WGS 84). `--output PATH` writes to a file. Progress messages go to standard error. Thus you can pipe the output. `-q` shows only warnings. `-v` adds debug messages. If the gazetteer is not installed, the command exits with status 2 before it loads a model. It tells you to run `geoparser install`.

`geoparser list --json` prints the installed gazetteers as a JSON array, for scripts.
