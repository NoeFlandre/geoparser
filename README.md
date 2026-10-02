<p align="center">
  <img src="docs/_static/logo.png" alt="Irchel Geoparser" width="360">
</p>

<p align="center">
  <a href="https://github.com/NoeFlandre/geoparser/actions/workflows/test.yml?query=branch%3Amain+"><img src="https://img.shields.io/github/actions/workflow/status/NoeFlandre/geoparser/test.yml?branch=main&logo=github&label=CI" alt="CI"></a>
  <a href="https://coverage-badge.samuelcolvin.workers.dev/redirect/NoeFlandre/geoparser"><img src="https://coverage-badge.samuelcolvin.workers.dev/NoeFlandre/geoparser.svg" alt="Coverage"></a>
  <a href="https://pypi.org/project/geoparser"><img src="https://img.shields.io/pypi/v/geoparser.svg" alt="PyPI"></a>
  <a href="https://pepy.tech/projects/geoparser"><img src="https://static.pepy.tech/badge/geoparser" alt="Downloads"></a>
  <a href="https://pypi.org/project/geoparser"><img src="https://img.shields.io/pypi/pyversions/geoparser.svg" alt="Python"></a>
  <a href="https://github.com/NoeFlandre/geoparser/blob/main/LICENSE"><img src="https://img.shields.io/github/license/NoeFlandre/geoparser.svg" alt="License"></a>
</p>

A Python library that finds place names in text and links them to geographic locations.

Geoparsing has two stages. The library keeps the two stages separate. A *recognizer* finds the words that are place names. A *resolver* selects the place that each name refers to. The resolver selects the place from the entries of a *gazetteer*.

You supply the recognizer, the resolver, and the gazetteer. You can replace each of them. You can use a module that works in a different way. You can use a different model. You can fine-tune a module on your own annotated data. You can also write your own module against a small interface.

The library includes gazetteer configurations for the modern world and for Switzerland. You can change other geographic data into a gazetteer with a YAML configuration file. You do not write code.

## Installation

```bash
pip install geoparser
```

The library also needs a gazetteer. The package does not include a gazetteer. A gazetteer is the database of places that the library uses to resolve names.

```bash
geoparser install geonames
```

The library stores gazetteers and SQLite databases in the standard application data directory of the operating system. To use a different location, set `GEOPARSER_DATA_DIR`. The [installation guide](https://docs.geoparser.app/installation.html#where-data-is-stored) describes the directory layout.

Read the [installation guide](https://docs.geoparser.app/installation.html) for the environment setup, the available gazetteers, and their disk requirements.

## CLI

Use the module entry point to manage gazetteers and to start the annotator:

```bash
python -m geoparser --help
python -m geoparser install geonames
geoparser annotator --port 8000 --no-browser
```

To parse text, use the Python API in the next section.

## Data Paths

In Docker Compose, the named `geoparser-data` volume is mounted at `/data`. Gazetteers and SQLite databases use `/data/geoparser`. Hugging Face model files and cache files use `/data/hf`. Both paths stay available when the containers stop.

## Quick Start

```python
from geoparser import Geoparser
from geoparser.modules import GLiNER2Recognizer, PriorResolver

# Build a pipeline from a recognizer and a resolver
geoparser = Geoparser(
    recognizer=GLiNER2Recognizer(),
    resolver=PriorResolver(gazetteer_name="geonames"),
)

# Parse text
document = geoparser.parse(
    "The conference was held in Zurich, with satellite events in Geneva and Basel."
)

# Access results
for toponym in document.toponyms:
    location = toponym.location  # None if the name could not be resolved
    print(
        f"{toponym.text} -> {location.data['name']}, {location.data['country_name']} "
        f"({location.data['latitude']}, {location.data['longitude']})"
    )
```

```text
Zurich -> Zürich, Switzerland (47.36667, 8.55)
Geneva -> Geneva, Switzerland (46.20222, 6.14569)
Basel -> Basel, Switzerland (47.55839, 7.57327)
```

The library links each name to one specific entry in GeoNames. You get the name and the coordinates, as shown above. You also get a stable identifier for the place and the type of the place. You get the administrative units that contain the place. You also get a geometry. You can use the geometry to map, measure, or export the place.

## Outputs

`Geoparser.parse` returns a document. The document contains the detected `toponyms`. Each toponym keeps the source text and the span. If the resolver finds no match, `toponym.location` is `None`. If the resolver finds a match, `toponym.location` gives the data of the matched place. This data includes the name and the coordinates.

## Docker

The runtime image uses the checked-out source and `uv.lock`. Docker Compose gives the install service and the annotator service the same named data volume:

```bash
docker compose build
docker compose --profile install run --rm install
docker compose up annotator
```

The annotator is at [http://localhost:8000](http://localhost:8000). Set `HF_TOKEN` in the shell before you run Compose. Do this only if a Hugging Face resource that you use needs authentication. The volume keeps the installed gazetteer, the databases, and the Hugging Face cache when the containers stop. To build the notebook image locally, read the [demo setup guide](demo/README.md). A prebuilt demo image is not necessary.

## Documentation

The full documentation is at **[docs.geoparser.app](https://docs.geoparser.app)**. It includes the setup, the guides, and the API reference.

The [benchmark results](https://huggingface.co/datasets/NoeFlandre/geoparser-benchmark-results) compare the built-in resolver pipelines on English and multilingual corpora.

## Project Status

The library is in active development. The architecture can still change. While the version is below `1.0`, a minor release can include breaking changes. [ROADMAP.md](ROADMAP.md) describes the larger changes that we plan.

## Contributing

We welcome questions, bug reports, and ideas in the [issues](https://github.com/NoeFlandre/geoparser/issues). We also welcome pull requests. Read [CONTRIBUTING.md](CONTRIBUTING.md) for the local setup and the development guidelines.

## Acknowledgments

Diego Gomes started the Irchel Geoparser as part of his Master's thesis. The [Department of Geography](https://www.geo.uzh.ch/) at the University of Zurich and the [Public Data Lab](https://publicdatalab.ch/) of the Digitalization Initiative of the Zurich Higher Education Institutions supported the further development. We thank Prof. Dr. Ross Purves. He gave us the opportunity to continue this work in a research project.

## Citation

To cite this project in research or software, use the metadata in [CITATION.cff](CITATION.cff).

## License

This project has the MIT License. Read the [LICENSE](LICENSE) file for details.

Geoparser uses a number of third-party libraries. [pyproject.toml](pyproject.toml) lists them. Each library has its own license. pip installs each library with its license.
