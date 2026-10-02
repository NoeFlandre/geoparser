# Irchel Geoparser

The **Irchel Geoparser** finds place names in text. It links the names to places in a geographic database.

Give it a sentence, a document, or a corpus. It returns the place names that it found. Where possible, it links each name to an entry in a **gazetteer**. A gazetteer is a database of places. The information in an entry depends on the gazetteer. It usually includes coordinates. It also includes attributes, such as the type of the place and the administrative units that contain it. Thus the library changes prose into data. You can map the data, count it, and join it to other data.

``` python
from geoparser import Geoparser
from geoparser.modules import GLiNER2Recognizer, PriorResolver

geoparser = Geoparser(
    recognizer=GLiNER2Recognizer(),
    resolver=PriorResolver(gazetteer_name="geonames"),
)

document = geoparser.parse(
    "The conference was held in Zurich, with satellite events in Geneva and Basel."
)

for toponym in document.toponyms:
    location = toponym.location  # None if the name could not be resolved
    print(
        f"{toponym.text} → {location.data['name']}, {location.data['country_name']} "
        f"({location.data['latitude']}, {location.data['longitude']})"
    )
```

``` text
Zurich → Zürich, Switzerland (47.36667, 8.55)
Geneva → Geneva, Switzerland (46.20222, 6.14569)
Basel → Basel, Switzerland (47.55839, 7.57327)
```

The library links each name to one specific entry in GeoNames. You get the name and the coordinates, as shown above. You also get a stable identifier for the place and the type of the place. You get the administrative units that contain the place. You also get a geometry. You can use the geometry to map, measure, or export the place.

First read the [installation](installation.md) page. Then parse your first text in the [quickstart](quickstart.md). The [demo](demo.md) maps every place in the book *Around the World in Eighty Days* by Jules Verne.

The [benchmark results](https://huggingface.co/datasets/NoeFlandre/geoparser-benchmark-results) show the measured resolver performance on English and multilingual corpora. The [glossary](glossary.md) defines the project terms.

## What It Does

Geoparsing has two stages. The library keeps the two stages separate. A **recognizer** finds the words in a text that are place names. A **resolver** then selects the place that each name refers to. The resolver selects the place from the entries of a gazetteer. The second stage is the difficult one. GeoNames has 122 places with the name Paris and 291 places with the name Springfield. For this reason, you must install a gazetteer to set up the library.

You supply the recognizer, the resolver, and the gazetteer. You can replace each of them. This is the central design decision of the library. Most of the features come from it. You can replace a recognizer that uses a statistical model with a recognizer that takes spans from you. You can point a module to a different model. You can fine-tune a module on your own annotated data for a language, a period, or a domain. You can also write your own module against a small interface. You can keep the results of several pipelines side by side for the same corpus and compare them.

The two pre-configured gazetteers cover the modern world and Switzerland in detail. Other geographic data can become a gazetteer with a YAML configuration file. Examples are a historical atlas, an excavation catalogue, a national register, or your own field data. You do not write code.

The [concepts](concepts.md) page explains these topics in more detail. It has no code.

## Project Status

The library is in active development. The architecture can still change. While the version is below 1.0, a minor release can include breaking changes. [ROADMAP.md](https://github.com/NoeFlandre/geoparser/blob/main/ROADMAP.md) describes the larger changes that we plan.

## Contributing

The Irchel Geoparser is open source. We welcome questions, bug reports, and ideas on the [issue tracker](https://github.com/NoeFlandre/geoparser/issues). We also welcome contributions. Read [CONTRIBUTING.md](https://github.com/NoeFlandre/geoparser/blob/main/CONTRIBUTING.md).

## Acknowledgments

Diego Gomes started the Irchel Geoparser as part of his Master's thesis. The [Department of Geography](https://www.geo.uzh.ch/) at the University of Zurich and the [Public Data Lab](https://publicdatalab.ch/) of the Digitalization Initiative of the Zurich Higher Education Institutions supported the further development. We thank Prof. Dr. Ross Purves. He gave us the opportunity to continue this work in a research project.

## License

The Irchel Geoparser has the [MIT License](https://github.com/NoeFlandre/geoparser/blob/main/LICENSE). It uses a number of third-party libraries. [pyproject.toml](https://github.com/NoeFlandre/geoparser/blob/main/pyproject.toml) lists them. Each library has its own license. pip installs each library with its license.
