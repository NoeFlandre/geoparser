# Working with Results

This guide explains what a pipeline returns and how to read it. It describes the three kinds of object that a parse makes and the attributes of each object. It also describes two cases that your code must handle: an unresolved name and a missing attribute.

## The Shape of the Output

A parse always returns **documents**. Each document holds the **toponyms** that the recognizer found in it. Each toponym can hold the **location** that the resolver selected.

``` python
from geoparser import Geoparser
from geoparser.modules import SentenceTransformerResolver, SpacyRecognizer

geoparser = Geoparser(
    recognizer=SpacyRecognizer(),
    resolver=SentenceTransformerResolver(gazetteer_name="geonames"),
)

document = geoparser.parse(
    "The conference was held in Zurich, with satellite events in Geneva and Basel."
)

for toponym in document.toponyms:
    print(toponym.text, toponym.start, toponym.end, toponym.location)
```

``` text
Zurich 27 33 Feature(geonames:2657896)
Geneva 60 66 Feature(geonames:2660646)
Basel 71 76 Feature(geonames:2661604)
```

The output has three levels. Each level answers a different question:

| Object                  | What you get from it                                                   |
|-------------------------|------------------------------------------------------------------------|
| `Document`              | `text`, `id`, and `toponyms`                                           |
| `Reference` (a toponym) | `text` as written, `start` and `end` character offsets, and `location` |
| `Feature` (a location)  | `identifier`, `data` attributes, `geometry`, `crs`, `names`, `source`  |

The [models API](../api/models.md) reference gives the full signatures of the three objects.

## Toponyms

A toponym is a place name as the recognizer found it in the text. `text` is the name as written in the text. It can be different from the name in the gazetteer. For example, the toponym above reads `Zurich`, but GeoNames names the city `Zürich`.

`start` and `end` are character offsets into `document.text`. `end` is exclusive. Thus `document.text[toponym.start:toponym.end]` returns the toponym. A wider slice returns the text around the toponym.

## Locations

A location is a `Feature`. It is one entry in the gazetteer that the resolver used.

`data` is a dictionary of the attributes that the gazetteer has for the place. The keys depend on the gazetteer. They can be different between sources in the same gazetteer. Use `.get()` to read them:

``` python
data = toponym.location.data
print(data.get("name"), data.get("country_name"), data.get("feature_name"))
```

`identifier` is the stable ID of the place in the gazetteer. Store it when you need a reference that lasts, for example in annotations or in exported data. Use it also to group or count places. Names are not unique. GeoNames has 122 places named Paris. If you count by `data["name"]`, you merge places that have the same name. If you count by `identifier`, you do not.

`geometry` is a Shapely object. It is usually a point. It can also be a line, a polygon, or a multi-part geometry. Its coordinate reference system is in `crs`. For both pre-configured gazetteers, `crs` is `EPSG:4326`. Because `geometry` is a Shapely object, you can measure it or transform it. You can also give it to any library that reads Shapely geometries.

`names` lists all the strings that you can use to search for the place. `source` gives the name of the gazetteer source of the feature. Use `source` to tell features of different kinds apart in the same gazetteer.

## What May Be Missing

Two kinds of data are often absent. Your code must handle them.

**A toponym can have no location.** `toponym.location` is `None` when the recognizer found a name but the resolver did not select a place. This occurs when the place is not in the gazetteer. It also occurs when no candidate passes the similarity threshold of the resolver.

``` python
for toponym in document.toponyms:
    if toponym.location is None:
        print(f"{toponym.text}: unresolved")
        continue
    print(f"{toponym.text}: {toponym.location.data.get('name')}")
```

**A location can lack an attribute or a geometry.** An attribute is absent when the gazetteer has no value for it. For example, the Pacific Ocean has no `country_name`. `geometry` can also be `None`. Every GeoNames entry has coordinates. But a gazetteer that you build yourself does not need a geometry for each place.

``` python
geometry = toponym.location.geometry
if geometry is not None:
    print(geometry.x, geometry.y)
```

These cases tell you about your setup. They do not show a defect in the library. Assume that a name occurs many times in your corpus and never resolves. Then the gazetteer probably does not cover that kind of place, or the threshold of the resolver is too high for your material. Refer to [modules](modules.md). Also parse whole paragraphs and not single sentences. The resolver uses the words around a name to select between places that have the same name. Refer to [Why Context Matters](../quickstart.md#why-context-matters).

## Taking Results Elsewhere

The results are ordinary Python data: strings, numbers, dictionaries, and Shapely geometries. To move them to another tool, walk the structure one time and keep the fields that you need:

``` python
documents = geoparser.parse(texts)

rows = []
for document in documents:
    for toponym in document.toponyms:
        location = toponym.location
        data = location.data if location else {}
        rows.append(
            {
                "document_id": str(document.id),
                "toponym": toponym.text,
                "start": toponym.start,
                "end": toponym.end,
                "gazetteer_id": location.identifier if location else None,
                "name": data.get("name"),
                "latitude": data.get("latitude"),
                "longitude": data.get("longitude"),
            }
        )
```

Most tabular tools and spatial tools expect one row for each toponym, as in the example. Note that `parse()` returns the same shape as its input. A single string gives one document and not a list. Put the string in a list before you iterate, or give a list to `parse()` at the start.

## Relating Results to Your Own Records

A parsed document is usually not the goal. You have records, such as articles, interviews, or files. You want to attach the geography to those records.

With `Geoparser.parse()`, the order links the results to the records. A list of texts returns a list of documents in the same order:

``` python
documents = geoparser.parse([article["body"] for article in articles])

for article, document in zip(articles, documents):
    article["places"] = [t.location.identifier for t in document.toponyms if t.location]
```

For a larger corpus, or for work that you want to do again, use a `project <projects>`. Keep the document identifiers that the project gives you. They last between sessions. The positional order does not.

## Keeping Results

`Geoparser.parse()` discards its work when it returns. To keep the work, set `save=True`:

``` python
document = geoparser.parse("Berlin is the capital of Germany.", save=True)
```

``` text
Results saved under project name: a1b2c3d4
```

Use the printed name to get the results again with `Project("a1b2c3d4")`. A random name is difficult to remember. If you know in advance that you want to keep the output, create a project with a name that you choose. Refer to [projects](projects.md).
