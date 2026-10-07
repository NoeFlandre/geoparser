# Querying Gazetteers

A gazetteer is the list of places from which a resolver selects. This guide explains what a gazetteer contains and how the pre-configured gazetteers are different. It also explains how to query them directly from Python. The [installation](../installation.md) page explains how to install a gazetteer. The [custom gazetteers](custom-gazetteers.md) guide explains how to build one from your own data.

## What a Gazetteer Is

The Irchel Geoparser uses gazetteers as the authoritative source of geographic information for toponym resolution. A gazetteer stores information about places: their names, types, administrative hierarchies, and coordinates. Assume that a text mentions "Paris". The gazetteer has entries for Paris, France; Paris, Texas; Paris, Ontario; and many others. Each entry has the name and also attributes, such as coordinates, population, feature type, and administrative hierarchy. These attributes make it possible to tell one Paris from another.

The library keeps gazetteers separate from the processing modules. A resolver does not use SQL or file reads to access a gazetteer. It uses the `Gazetteer` class, which has a small set of search methods. This boundary lets any gazetteer work with any resolver.

Each installed gazetteer is one self-contained SQLite file, an *artifact*. All gazetteers have the same fixed schema. It has a `feature` table (identifier, source, data as JSON, geometry as WKB), a `name` table with full-text indexes and phonetic indexes for search, and a small `metadata` table. A pipeline builds the artifacts from declarative YAML configurations. The pipeline downloads the source files and puts them in a temporary analytical database (DuckDB). Then it projects them into the canonical schema. This includes joins, spatial joins, and deduplication. After the build, the pipeline discards the source files and the staging data. Only the artifact is installed. Nothing changes it afterwards.

Because all artifacts have the same schema, all gazetteers behave in the same way at query time. This is true even if their source data is very different. Each gazetteer stores its geometries in one coordinate reference system (EPSG:4326 by default). Any reprojection occurs one time, at build time.

## The Pre-configured Gazetteers

The library has ready-made configurations for two gazetteers. You can install them without a configuration of your own. Install them as the [installation](../installation.md) page describes. This section describes what you get.

### GeoNames

GeoNames is a comprehensive global gazetteer with more than 13 million place names. It covers countries, administrative divisions, cities, towns, and neighborhoods. It also covers natural features, such as mountains and rivers, and points of interest, such as buildings and monuments. The developers trained the default resolver models on this gazetteer. Use it for real work.

Its global scope gives uneven coverage. Some regions have more detail and more current data than others. Consider this before you draw conclusions from the number of resolved toponyms.

### SwissNames3D

SwissNames3D is the official Swiss placename register from Swisstopo, the Federal Office of Topography. In Switzerland, it is much richer than GeoNames. It has fine feature classifications and full geometries (points, lines, and polygons). Spatial joins at build time link each feature to its municipality, district, and canton.

Its attribute names are in German. The developers fine-tuned the pre-trained resolver models on English GeoNames descriptions. Lower `min_similarity` and, if possible, fine-tune the model. Refer to [modules](modules.md) and [training](training.md).

## Querying a Gazetteer

Resolvers use the `Gazetteer` class, and you can use it directly. Use it to examine what a gazetteer contains. Use it to check if a place is in the gazetteer before you blame the resolver. Use it also to write your own resolver.

### Opening a gazetteer

``` python
from geoparser import Gazetteer

gazetteer = Gazetteer("geonames")
```

Gazetteer names must contain only ASCII letters, digits, underscores, and
hyphens. Names cannot be empty or contain path separators. Invalid names raise
`ValueError` in the Python API. The CLI reports them with exit code 2.

The name must be the name of an installed gazetteer. If it is not, the call raises a `ValueError` that names the command to install the gazetteer.

### Searching by name

`search()` finds features by name. It has four methods. Each method trades precision for recall:

``` python
from geoparser import Gazetteer

gazetteer = Gazetteer("geonames")

features = gazetteer.search("Paris", method="exact")
print(f"Found {len(features)} features")

for feature in features[:5]:
    print(f"- {feature.data.get('name')}, {feature.data.get('country_name')}")
```

| Method      | Behavior                                                                                                                                              |
|-------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| `"exact"`   | Returns only features whose name is exactly the search string, ignoring case and diacritics. This is the fastest method. It misses names with a different spelling. |
| `"phrase"`  | Returns features whose name contains the search string as a complete phrase. It finds "New York City" for the search "New York". It is still restrictive. |
| `"partial"` | Returns features whose name contains any token of the search string. It handles articles and qualifiers that are added or omitted. It returns many more candidates. |
| `"fuzzy"`   | Uses approximate string matching. It tolerates spelling variation and typing errors. It is the most permissive method and the slowest. |

Two more arguments change the result. `tiers` sets how many rank tiers of results the search includes for the methods that are not exact. The search ranks results by match score: BM25 relevance for phrase and partial, and edit distance for fuzzy. It puts results with similar scores into the same bracket. A higher value includes matches of lower quality. `limit` sets the maximum number of results. The default is 10000. This matters mainly for common names with permissive methods.

``` python
# Only the top-ranked matches
features = gazetteer.search("London", method="partial", tiers=1)

# More permissive, including lower-ranked matches
features = gazetteer.search("London", method="partial", tiers=3)
```

An empty result has two possible causes. The name is not in the gazetteer, or the method was too strict. To find the cause quickly, go from `"exact"` to `"fuzzy"`.

### Looking up by identifier

If you know the identifier of a feature, `find()` retrieves the feature directly:

``` python
from geoparser import Gazetteer

gazetteer = Gazetteer("geonames")

# Paris, France, by its geonameid
feature = gazetteer.find("2988507")

if feature:
    print(f"Name: {feature.data.get('name')}")
    print(f"Country: {feature.data.get('country_name')}")
    print(f"Population: {feature.data.get('population')}")
```

The identifier scheme belongs to the gazetteer. It is the geonameid for GeoNames and a UUID for SwissNames3D. For a gazetteer of your own, it is the identifier that you chose. For an unknown identifier, `find()` returns `None` and does not raise an error. Check the result before you use it.

## Working with Features

`search()` and `find()` return `Feature` objects. There is one object for each place:

``` python
from geoparser import Gazetteer

gazetteer = Gazetteer("geonames")
features = gazetteer.search("Tokyo")

if features:
    feature = features[0]

    print(f"ID: {feature.identifier}")  # stable identifier
    print(f"Source: {feature.source}")  # which source it was built from
    print(f"Data: {feature.data}")  # attributes, as a dictionary
    print(f"Names: {feature.names}")  # every name it is searchable by
    print(f"Geometry: {feature.geometry}")  # Shapely geometry, or None
    print(f"CRS: {feature.crs}")  # e.g. EPSG:4326
```

The `identifier` is the value that you store to refer to this place, for example in annotations or in exported data. Use it wherever the reference must last. The `source` names the gazetteer source that the feature came from (`allCountries` in GeoNames). Use it to tell features of different kinds apart in the same gazetteer. `names` has all the strings that you can use to search for the place. It has no order and no labels. `data` has the attributes. `geometry` is a Shapely object in the coordinate system that `crs` gives. It is usually a point, but it can be a line, a polygon, or a multi-part geometry. It can also be `None`, because a gazetteer does not need a location for each place.

``` python
if feature.geometry is not None:
    print(feature.geometry.x, feature.geometry.y)
```

### Attributes by gazetteer

The keys in `data` depend on the gazetteer and on the source in the gazetteer. For GeoNames, the keys are:

- `name`, `asciiname`: The main name of the feature and its ASCII transliteration.
- `latitude` and `longitude`: The coordinates in decimal degrees.
- `feature_class`: The top-level category, as one letter. `P` is a populated place, `A` an administrative area, `H` water, `T` terrain, `S` a spot or building, `L` an area, `R` a road, `U` undersea, and `V` vegetation. Use it to filter results to one kind of place.
- `feature_code`, `feature_name`: The detailed type code and its readable form ("city", "mountain", "stream").
- `country_code`, `country_name`: The country that has the feature.
- `admin1_name`, `admin2_name`: The first-level and second-level administrative divisions. `admin1_code` to `admin4_code` give their codes.
- `population`: The number of inhabitants of an inhabited place.
- `elevation`, `dem`: The elevation in metres, as recorded and as sampled from a digital elevation model.
- `cc2`, `timezone`, `modification_date`: The alternate country codes, the timezone, and the date of the last change of the GeoNames record.

For SwissNames3D, the keys depend on the source, because the gazetteer is built from point datasets, line datasets, polygon datasets, and boundary datasets. All named features have these keys:

- `NAME`: The name of the feature.
- `OBJEKTART`: The detailed object type, in German.
- `GEMEINDE_NAME`, `BEZIRK_NAME`, `KANTON_NAME`: The municipality, the district, and the canton. A spatial join assigns them.

Point features also have `HOEHE` (elevation in metres). Line features also have `KUNSTBAUTE`. Polygon features also have `EINWOHNERK` and `ISCED`.

The keys can be different between sources in the same gazetteer. Therefore read them with `feature.data.get("key")` and not with a subscript. The configuration file of the gazetteer defines the exact keys for each source. A custom gazetteer has the keys that you gave it.

## Using a Gazetteer with a Resolver

You tell the resolver which gazetteer to use when you construct it:

``` python
from geoparser.modules import SentenceTransformerResolver

resolver = SentenceTransformerResolver(gazetteer_name="swissnames3d")
```

For GeoNames and SwissNames3D, this is all that you need. Some resolvers describe candidates in words. `SentenceTransformerResolver` is an example. For a gazetteer of your own, you must also tell such a resolver which attributes to use for the description. It cannot guess them. Refer to [modules](modules.md).
