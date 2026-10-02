# Building Custom Gazetteers

The pre-configured gazetteers cover the modern world and one country in detail. But no fixed set of gazetteers can cover every research question. Assume that you work on a region, a period, or a domain that the gazetteers do not cover. Examples are a national placename register, an excavation catalogue, a historical map index, or your own field data. You can turn that data into a gazetteer of your own. The library then uses it in the same way as a pre-configured gazetteer.

To do this, write a YAML configuration file. The file describes your source files and how their rows map to places. You do not write a plugin and you do not run code. You declare where the files are and which columns they have. You also declare which columns are the identifier, the names, the geometry, and the attributes of a place. The build pipeline does the rest.

This guide first goes through the process from start to end on a real dataset. Then it documents every configuration key.

## What You Are Building

Before you write a configuration, learn exactly what the build produces. Your configuration must describe it.

### The Canonical Feature Model

The build projects every gazetteer into one model, even if its sources are very different. A gazetteer is a set of **features**. Each feature has exactly five things:

| Field        | Meaning                                                                                                                                                                                                                                                  |
|--------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `identifier` | A string that identifies the place in this gazetteer and never changes. The library stores it in annotations. Thus it must be stable across rebuilds and unique in the whole gazetteer.                                                                  |
| `names`      | All strings by which a user must be able to find the place: its main name, historical spellings, transliterations, names in other languages, and abbreviations. Names have no rank and no label. They are a set of search keys.                         |
| `geometry`   | One geometry (point, line, polygon, or a multi-part combination) in the coordinate reference system of the gazetteer. It can be absent. A place that has a name but no location is a valid feature.                                                      |
| `data`       | A free-form dictionary of attributes: type, hierarchy, population, dates, links, descriptions, or other data that your source has and your work needs. There is no fixed schema. Different sources in the same gazetteer can store different keys.       |
| `source`     | The source of the configuration from which the feature came. The build sets it automatically. Use it to tell features of different kinds apart in one gazetteer.                                                                                         |

The finished gazetteer is one self-contained SQLite file (an *artifact*). It has these features and also the full-text indexes and phonetic indexes for search. Nothing else is installed. Nothing changes the artifact after the build.

Thus, to write a configuration, you answer five questions about your data. What is one place? What identifies it? What is it called? Where is it? What else must I know about it?

### How the Build Runs

The build reports three stages. If you know them, you can find the cause of an error message more easily:

1.  **Preparing sources.** The build downloads each source file or finds it on disk. If the file is a ZIP archive, the build extracts it. Then it loads the file into a table in a temporary analytical database (DuckDB). Errors in this stage concern files and columns. Examples are a missing file, a wrong column count, or a value that cannot convert to its declared type.
2.  **Compiling features.** For each `features` block, the build compiles the declared identifier, names, geometry, and data into SQL over the source of the block and its joins. Then it runs the SQL. Errors in this stage concern your expressions. Examples are an unknown column name, an invalid join clause, or an identifier that collides with the identifier of another block.
3.  **Building artifact.** The build writes the projected rows to a temporary SQLite file. It indexes, verifies, and compacts the file. Only then does it move the file into place. If a build fails, a previously installed artifact does not change.

The build discards the source files and the staging tables afterwards. It reads the sources again from the start for each build. Thus you can safely iterate on a configuration. Run it again, and the build atomically replaces the previous artifact.

## Worked Example: A Gazetteer of the Ancient World

This section builds a working gazetteer of the ancient world from the start. It adds one concern at a time. Each intermediate step is a valid configuration. You can install it and query it. Thus you can follow along and check your results.

The example uses two datasets. Together they use almost everything that a configuration can do:

- [Pleiades](https://pleiades.stoa.org/) is a community-built gazetteer of the ancient Mediterranean world. It is published as a ZIP archive of CSV exports from a relational database. It gives us places, their names in several scripts, and a controlled vocabulary of place types. The data is in separate files that you must join together.
- A [map of Roman provinces](https://urbesetorbis.com/) at the greatest extent of the empire is published as one GeoJSON file in Web Mercator. It gives us polygons in which we locate the places. Its coordinate system is different from that of all the other data.

The finished file is `pleiades.yaml <../examples/pleiades.yaml>`. The end of the walkthrough shows it in full. Thus you can compare your version with it at any point. It is an example and not a pre-configured gazetteer. You install it from the file, in the same way as you install a configuration of your own.

> [!NOTE]
> Every step below uses `name: pleiades`. Thus each build replaces the artifact of the previous step. This is what you want when you iterate.

The examples show every configuration key, also the keys that have a default. A comment states the default. Real configurations usually omit those lines. They are written here so that the file has no implicit parts.

### Step 1: Read the Data First

Do not start with the YAML file. First download the data and look at it. Each decision in the configuration depends on what is in the files.

``` bash
curl -O https://atlantides.org/downloads/pleiades/gis/pleiades_gis_data.zip
```

``` bash
unzip -l pleiades_gis_data.zip
```

The archive is about 35 MB. It expands to about 130 MB under `data/gis/`. It has seventeen CSV exports and a README that describes them. Four of the exports are relevant to us. A fifth file comes from the second dataset:

| File                     | Rows   | What it contributes                                                                                           |
|--------------------------|--------|---------------------------------------------------------------------------------------------------------------|
| `places.csv`             | 42,242 | One row for each place: title, description, a representative coordinate pair, a bounding box, and the Pleiades id |
| `names.csv`              | 43,708 | One row for each *name*, linked to a place: the attested form in its original script and up to three romanizations |
| `places_place_types.csv` | 52,511 | Which place-type keys apply to which place. It has more rows than places, because a place can have more than one type |
| `place_types.csv`        | 233    | The place-type vocabulary: key, readable term, definition                                                     |
| `empire2.geojson`        | 44     | (Separate download) One MultiPolygon for each Roman province                                                  |

Look at the actual bytes of each file that you plan to use. Do not rely only on the documentation:

``` bash
head -2 data/gis/places.csv
```

``` text
created,description,details,provenance,title,uri,id,representative_latitude,representative_longitude,bounding_box_wkt,location_precision
2021-11-14T03:44:08Z,"An ancient region covering a large part of southwestern Europe, ...",<p>The Barrington Atlas Directory notes: FRA</p>,Barrington Atlas: BAtlas 1 D1 Gallia,Gallia,https://pleiades.stoa.org/places/993,993,46.360953305773286,1.6706144893053327,"POLYGON ((9.6708805 31.937048, ...))",rough
```

These two lines already decide five parts of the configuration:

- There is a **header row**. The loader does not skip it automatically (`skip_rows: 1`).
- The fields are **separated by commas** and **quoted**. Some quoted fields contain commas and also line breaks. Thus quote handling must stay enabled (this is the default).
- `id` is a stable numeric identifier. It is the same number as the number in the `uri`. It is our `identifier`.
- `title` is the display name. `representative_latitude` and `representative_longitude` are our coordinates.
- The coordinates are plain decimal degrees. Thus the coordinate system of this source is EPSG:4326. This is the same system that the gazetteer stores. The provinces file is different.

This file does not have two answers. The alternate names are in `names.csv`. The place types are in `places_place_types.csv`. This is normal for data that was exported from a relational database. Most of the work below is to join those files back together.

Ask also what *counts* as a place here. Pleiades includes regions, rivers, roads, and ethnic groups together with settlements. About 7,500 of its places have no coordinates, because texts attest them but nobody has located them. We keep all of them. A place that is not located is still worth finding by name.

### Step 2: Get One Source to Build

Do not write the whole configuration at once. Start with one source, one identifier, and one name. Confirm that it builds. Then add one thing at a time. It is much easier to debug a small configuration that just broke than a large configuration that never worked.

Save this as `pleiades.yaml`:

``` yaml
name: pleiades
crs: EPSG:4326 # default

sources:
  - name: places
    url: https://atlantides.org/downloads/pleiades/gis/pleiades_gis_data.zip
    file: places.csv
    delimiter: ","
    quote: '"' # default
    skip_rows: 1 # header row; default is 0
    crs: EPSG:4326 # default (the gazetteer's crs)
    attributes:
      - name: "created"
        type: text
      - name: "description"
        type: text
      - name: "details"
        type: text
      - name: "provenance"
        type: text
      - name: "title"
        type: text
      - name: "uri"
        type: text
      - name: "id"
        type: integer
      - name: "representative_latitude"
        type: real
      - name: "representative_longitude"
        type: real
      - name: "bounding_box_wkt"
        type: text
      - name: "location_precision"
        type: text

features:
  - source: places
    identifier: "id"
    names:
      - "title"
```

Five parts of the source declaration need your attention. First attempts usually fail there:

- `url` points to the **archive**. `file` names the file to take **out of** the archive. The build downloads and unpacks the archive automatically. You do not unpack it yourself. You do not need to know where the file is in the archive. Later steps add more sources from the same archive. The build downloads it only one time for each build.
- You must declare every column of a delimited file **in the order of the file**, also the columns that you do not use. If you declare fewer columns than the file has, the build does not drop the extra columns. The file cannot be parsed. `created`, `details`, and `provenance` are declared here only to account for their position.
- Each column declares a `type` (`text`, `integer`, `real`, or `geometry`). If you are not sure, select `text`. If an `integer` column has a non-numeric value anywhere in the file, the build aborts.
- `quote` and `skip_rows` describe the *text* format of a delimited file. They exist only for such files. Both are written here for clarity. But `quote: '"'` is the default.
- `crs` names the coordinate system of the coordinates of this source. It is written here to show where it goes. It is the same as the system of the gazetteer. Thus it changes nothing. Step 8 adds a source for which it does.

> [!TIP]
> The build does not keep downloads between builds. Thus each step below fetches the 35 MB archive again. You already have the archive from step 1. While you iterate, point the sources at your local copy. Use `path: pleiades_gis_data.zip` in place of the `url:` line. The build resolves it relative to the configuration file. Change back to `url` when you finish. All other parts work in the same way.

Install it:

``` bash
geoparser install pleiades.yaml
```

``` text
─────────────────────────────────── pleiades ───────────────────────────────────

Prepared sources                          ━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:08
Compiled features                         ━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:01
Built artifact                            ━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:01

Summary

Features  42,242
Names     42,242
```

This is a real gazetteer that you can query:

``` python
from geoparser import Gazetteer

gazetteer = Gazetteer("pleiades")
feature = gazetteer.find("433032")

print(feature.names)  # ['Pompeii']
print(feature.data)  # {}
print(feature.geometry)  # None
```

The gazetteer has 42,242 features and exactly one name for each feature. This is the same as the row count of `places.csv`. At this stage, it is very useful to check that the counts are what you expect. If the feature count is wrong now, the problem is in the source declaration. It is not in anything that you add later.

### Step 3: Decide What Goes into `data`

`data` is the attribute dictionary of the feature. You control it fully. Write each entry as it appears in a SQL `SELECT` list. A column name stores that column under its own name. An optional `AS <alias>` at the end renames it.

``` yaml
features:
  - source: places
    identifier: "id"
    names:
      - "title"
    data:
      - "title"
      - "location_precision"
      - "description"
      - "uri"
```

``` python
{
    "title": "Pompeii",
    "location_precision": "precise",
    "description": "An ancient city of Campania destroyed by the volcanic eruption of Mt. Vesuvius in A.D. 79, ...",
    "uri": "https://pleiades.stoa.org/places/433032",
}
```

What to include is a judgement. Consider who reads the data. Attributes help a person or a resolver to tell two places with the same name apart. They also point back to the source record. Type, hierarchy, and dates do this. Internal revision timestamps and provenance notes usually do not. They only make the artifact bigger. `description` is worth its size here, because the descriptions in Pleiades are very informative. `uri` is worth including in almost any gazetteer. It lets anyone who uses your data go back to the original record.

### Step 4: Add Geometry

The geometry of a feature is one value. It is a geometry column of a spatial source or an expression that constructs a geometry. Here we build a point from the two coordinate columns:

``` yaml
features:
  - source: places
    identifier: "id"
    geometry: "ST_Point(representative_longitude, representative_latitude)"
    names:
      - "title"
    data:
      - "title"
      - "representative_latitude AS latitude"
      - "representative_longitude AS longitude"
      - "location_precision"
      - "description"
      - "uri"
```

> [!WARNING]
> `ST_Point` takes **longitude first** and then latitude. This is the most frequent mistake in a gazetteer configuration. It fails silently. The build succeeds, but the places are mirrored across the globe. Check one place that you know before you continue. Pompeii must be at about 14.49 E, 40.75 N and not at 40.75 E, 14.49 N.

The build now reports the same 42,242 features. Of these, 34,678 have a geometry. The other 7,564 are the places that are not located. Their coordinate columns are empty. Their geometry is `NULL`, and you can still search for them. It is redundant to also store the raw coordinates in `data`, because the geometry has them. But it is convenient for code that reads attributes and not geometry.

``` python
gazetteer = Gazetteer("pleiades")
feature = gazetteer.find("433032")

print(feature.geometry)  # POINT (14.485429 40.74941)
print(feature.crs)  # EPSG:4326
```

### Step 5: Add Names from a Second File

A gazetteer is only as good as its names. Until now, each place has exactly one name. The real names are in `names.csv`. It has one row for each name. Each row points to a place through `place_id`:

``` none
place_id  title      language_tag  attested_form  romanized_form_1
433032    Pompeii    la                           Pompeii
433032    Pompeia    grc           Πομπηία        Pompeia
433032    Pompei     it            Pompei         Pompei
433032    Pompei     la            Pompei         Pompei
433032    Colonia …  la            Colonia …      Colonia …
```

To get to them, the feature block *joins* that file. A join is a raw SQL join clause that the build appends to the source of the block. The whole joined table becomes available. You select what you need from it afterwards. Declare `names` as a second source, with all nineteen columns in the order of the file, exactly as for `places`. Then join it:

``` yaml
sources:
  - name: places
    # ... as before ...

  - name: names
    url: https://atlantides.org/downloads/pleiades/gis/pleiades_gis_data.zip
    file: names.csv
    delimiter: ","
    quote: '"' # default
    skip_rows: 1
    crs: EPSG:4326 # default
    attributes:
      - name: "created"
        type: text
      - name: "description"
        type: text
      - name: "details"
        type: text
      - name: "provenance"
        type: text
      - name: "title"
        type: text
      - name: "uri"
        type: text
      - name: "id"
        type: text
      - name: "place_id"
        type: integer
      - name: "name_type"
        type: text
      - name: "language_tag"
        type: text
      - name: "attested_form"
        type: text
      - name: "romanized_form_1"
        type: text
      - name: "romanized_form_2"
        type: text
      - name: "romanized_form_3"
        type: text
      - name: "association_certainty"
        type: text
      - name: "transcription_accuracy"
        type: text
      - name: "transcription_completeness"
        type: text
      - name: "year_after_which"
        type: integer
      - name: "year_before_which"
        type: integer

features:
  - source: places
    joins:
      - "LEFT JOIN names n ON id = n.place_id"
    identifier: "id"
    geometry: "ST_Point(representative_longitude, representative_latitude)"
    names:
      - "title"
      - "n.romanized_form_1"
      - "n.romanized_form_2"
      - "n.romanized_form_3"
      - "n.attested_form"
    data:
      - "title"
      # ... as before ...
```

Note that `id` is `text` in this file and `integer` in `places.csv`. You declare each source on its own terms. The type describes the column in *that* file. Here the column holds a slug and not a number. The pair of columns that the join compares must match: `places.id` and `names.place_id`. Both are integers.

Two conventions keep join clauses short. Give every joined table a **short alias** (`n` here). Refer to its columns through the alias (`n.attested_form`). Write the columns of the *own* source of the block **bare** (`id`, `title`). Do this everywhere in the block, also in the join condition. You never write a prefix for them. Use `LEFT JOIN` and not `JOIN`, unless you want to drop the places that have no match. An inner join here silently discards the 15,301 places that have no row in `names.csv`.

The name count increases from 42,242 to 77,923. Pompeii now has its Latin, Greek, and Italian names:

``` python
print(gazetteer.find("433032").names)
# ['Colonia Cornelia Veneria Pompeianorum', 'Pompei', 'Pompeia', 'Pompeii', 'Πομπηία']
```

> [!WARNING]
> **A one-to-many join multiplies rows. This changes what `data` means.** After this join, Pompeii has five rows and not one. The build collects the names across all the rows. This is what we want. The data values are different. The build takes each one from the *first* row of the group. Among rows that a join multiplied, "first" is arbitrary. Thus, if you read `n.language_tag` into `data`, you store one unpredictable language for each place. Use a one-to-many join to collect **names**. Get **attributes** from the own columns of the place, or aggregate them explicitly, as in the next step.

### Step 6: Clean Up the Names

Names sometimes need work before you can use them. The source data mixes names with editorial notation. A short look through the Pleiades titles shows three patterns:

``` none
Visurgis (river)                     qualifier in parentheses (4,295 titles)
Sigoulones?                          uncertain identification (1,373 titles)
Bisutun/Bagistana/Vastan?/Baptana    alternative readings, slash-separated (2,226 titles)
[Kangavar]/Concobar                  reconstructed form in brackets (152 titles)
```

No text that mentions the Weser calls it "Visurgis (river)". Thus a feature whose only name has a qualifier is not findable. Each `names` entry can be any scalar SQL expression. Use this to correct the problem. Build the expression in steps and not all at once. Remove the notation. Then split the remainder on the slashes. Let `unnest` turn the resulting list into one name for each element:

``` yaml
features:
  - source: places
    # ... as before ...
    names:
      - "title"
      - >-
        unnest(string_split(regexp_replace(title,
        '\s*\([^)]*\)|\?|\[|\]', '', 'g'), '/'))
      - "n.romanized_form_1"
      - "n.romanized_form_2"
      - "n.romanized_form_3"
      - "n.attested_form"
```

For the four titles above, this one entry produces `Visurgis`; `Sigoulones`; `Bisutun`, `Bagistana`, `Vastan`, `Baptana`; and `Kangavar`, `Concobar`. The build also keeps the raw `title` as a name. Thus nothing is lost if the notation is part of the real name. The build drops duplicates and empty results automatically. Thus you can write expressions like this one generously.

> [!TIP]
> `>-` is the folded block scalar of YAML. It joins the following lines into one string. Long expressions are much easier to read this way. Unlike in a quoted string, you do not double the backslashes. Thus you can write regular expressions exactly as SQL sees them.

The name count increases to 79,578. Whether an expression like this is worth writing depends on your data. To find out, sort your name column and read a few hundred values. This takes ten minutes and tells you more than any guess.

### Step 7: Aggregate a Many-to-Many Relation

Place types are the most useful attribute to tell places with the same name apart. In Pleiades, they are in two more files. `places_place_types.csv` maps places to type keys. `place_types.csv` translates those keys into readable terms. A place can have more than one type.

Declare both files first. Nothing can reference a source that does not exist yet:

``` yaml
sources:
  # ... places and names ...

  - name: places_place_types
    url: https://atlantides.org/downloads/pleiades/gis/pleiades_gis_data.zip
    file: places_place_types.csv
    delimiter: ","
    quote: '"' # default
    skip_rows: 1
    crs: EPSG:4326 # default
    attributes:
      - name: "place_id"
        type: integer
      - name: "place_type"
        type: text

  - name: place_types
    url: https://atlantides.org/downloads/pleiades/gis/pleiades_gis_data.zip
    file: place_types.csv
    delimiter: ","
    quote: '"' # default
    skip_rows: 1
    crs: EPSG:4326 # default
    attributes:
      - name: "key"
        type: text
      - name: "term"
        type: text
      - name: "definition"
        type: text
      - name: "same_as"
        type: text
      - name: "uri"
        type: text
```

No `features` block names either of them. You must declare a source that only supports a join or an expression. It does not back features of its own.

The obvious method is to join both files and read the term:

``` yaml
features:
  - source: places
    # Don't do this
    joins:
      - "LEFT JOIN names n ON id = n.place_id"
      - "LEFT JOIN places_place_types x ON id = x.place_id"
      - "LEFT JOIN place_types t ON x.place_type = t.key"
    data:
      - "t.term AS place_type"
```

This builds, but it is wrong in the way that the previous step warned about. Pompeii is both a `settlement` and an `urban area`. The join multiplies its rows, and `data` keeps one of the two types, unpredictably. This also shows a **chained join**. The second clause joins to a table that the first clause brought in. This is the correct pattern when the relation is many-to-*one* (a code and its label). It is not correct here.

We want all the types of a place in one value. The `data` entries are arbitrary scalar expressions. Thus a subquery can aggregate the relation without multiplying rows. The two joins can go away again:

``` yaml
features:
  - source: places
    joins:
      - "LEFT JOIN names n ON id = n.place_id"
    # ... identifier, geometry and names as before ...
    data:
      - "title"
      - >-
        (SELECT string_agg(DISTINCT coalesce(t.term, x.place_type), ', '
        ORDER BY coalesce(t.term, x.place_type))
        FROM places_place_types x
        LEFT JOIN place_types t ON x.place_type = t.key
        WHERE x.place_id = id)
        AS place_types
      - "representative_latitude AS latitude"
      - "representative_longitude AS longitude"
      - "location_precision"
      - "description"
      - "uri"
```

The subquery reads the bridge table for one place (`WHERE x.place_id = id`, where `id` is the own column of the current place). It looks up each key in the vocabulary. It joins the results into one string. It is an ordinary SQL query with its own `FROM` and its own join. Only one reference to `id` connects it to the feature that the build makes.

The `coalesce` is necessary because 1,904 rows of the bridge table reference keys that the vocabulary file does not have. Without it, those places silently lose a type. The `ORDER BY` is also necessary. Without it, the aggregation order is not specified, and a rebuild of the same configuration can produce different strings for places with more than one type. Pompeii now gets `"settlement, urban area"` in a stable way.

### Step 8: Join a Spatial Source

Pleiades has no administrative hierarchy. It has no "in Italy, in Campania" that can help to disambiguate. We can calculate a hierarchy. We have polygons of the Roman provinces. The province of a place is the polygon that contains its point. This is a **spatial join**. It is the main reason to bring in a second dataset.

A **spatial source** is any file from which the build can read geometry: Shapefile, GeoPackage, GeoJSON, and other formats that GDAL supports. It is different from a tabular source because it has no `delimiter`. It declares exactly one attribute of type `geometry`. The attribute is always named `geometry`:

``` yaml
sources:
  # ... the four Pleiades files ...

  - name: provinces
    url: https://urbesetorbis.com/downloads/empire2.geojson
    file: empire2.geojson
    crs: EPSG:3857 # Web Mercator, unlike the rest
    attributes:
      - name: "fid"
        type: integer
      - name: "Title"
        type: text
      - name: "Government"
        type: text
      - name: "StartYear"
        type: integer
      - name: "geometry"
        type: geometry
```

Three differences from a tabular source are important. The two text-format keys are not present. A spatial format has its own field names. Thus there is no header row to skip and nothing to unquote. If you declare `skip_rows` or `quote` on such a source, this is an error and not a no-op. A spatial source also selects its fields **by name**, unlike a delimited file. Thus it can declare a subset of them, in any order. This file also has a `color` field, which we omit. And `crs` finally has an effect, because this file is in Web Mercator and not in the degrees that the gazetteer stores.

This last point needs attention. In most geospatial tools, this is where the work starts. Here it is where the work ends. The declaration `crs: EPSG:3857` is all that you do. The build reprojects the geometries into the coordinate system of the gazetteer when it reads the source. This occurs before any of your expressions see them. Thus, from the point of view of the configuration, every geometry in every source is already in the same system. Now the join:

``` yaml
features:
  - source: places
    joins:
      - "LEFT JOIN names n ON id = n.place_id"
      - >-
        LEFT JOIN provinces p
        ON ST_Within(ST_Point(representative_longitude, representative_latitude),
        p.geometry)
    # ... identifier, geometry and names as before ...
    data:
      # ... title and place_types as before ...
      - "p.Title AS province"
      - "p.Government AS province_government"
      # ... the remaining attributes as before ...
```

A spatial join reads exactly like an attribute join. The condition is a spatial predicate and not an equality. Examples are `ST_Within`, `ST_Intersects`, and `ST_Contains`. Here the predicate asks which province polygon contains the point of the place. You do not handle coordinates in any way. There are degrees on the left and degrees on the right, because the build converted the polygons when it read them. For lines and polygons, reduce one side to a representative point with `ST_Centroid` if the predicate needs it.

> [!NOTE]
> The build does not convert one kind of geometry for you: a geometry that you build yourself from plain number columns, such as `ST_Point(x, y)` on a source whose `crs` is not the `crs` of the gazetteer. The build converts it when it is the `geometry` of a feature, like any other geometry. Inside a join condition, you must write `ST_Transform` around it yourself. This does not occur here, because the coordinates of Pleiades are already in degrees.

Of the 34,678 located places, 26,887 are inside a province. The other places are outside the empire, or they are in the empire at a different date. `provinces` is a many-to-one relation. Thus it is safe to read two of its columns into `data` here.

### Step 9: Consider a Second Kind of Place

Until now, every feature comes from one file. A configuration can have as many `features` blocks as it has sources. Each block projects a different file. Each block has its own identifier scheme, geometry, names, and attributes. In this way, one artifact can hold really different kinds of place. For example, it can hold settlements from one dataset and administrative areas from another.

The provinces are the obvious candidate here, because ancient texts name them very often. A block over that source looks like this:

``` yaml
features:
  - source: places
    # ... the block from step 8 ...

  - source: provinces
    identifier: "'province:' || fid"
    geometry: "geometry"
    names:
      - "Title"
      - "unnest(string_split(Title, ' et '))"
    data:
      - "Title AS title"
      - "'Roman province' AS place_types"
      - "Government AS government"
      - "StartYear AS start_year"
```

These ten lines have four points that a second block must always get right:

- **Identifiers must be unique in the whole gazetteer and not only in one block.** The own `fid` values of the provinces are 1 to 44. They collide with the Pleiades place ids. The expression adds a prefix to them. This gives `province:1` and so on. If two blocks produce the same identifier, the build fails and names the collision. It does not silently merge two places.
- `geometry: "geometry"` takes the polygon directly from the spatial source. It replaces a point that the build makes from coordinates. Nothing else in the block changes because its geometry is a MultiPolygon.
- Some provinces are administrative pairs. `unnest(string_split(Title, ' et '))` makes `Creta` and `Cyrenaica` findable together with `Creta et Cyrenaica`.
- `'Roman province' AS place_types` stores a constant. Blocks can store completely different attributes, and they usually do. But it is good to agree on a few keys, here `title` and `place_types`. Then anything that reads the gazetteer finds them on every feature, whatever its origin.

This gazetteer does not have that block. Check the reason before you add a block of your own. Pleiades already contains the provinces. `Sicilia (Roman province)`, `Dacia (province)`, and the others are places in `places.csv`. They have descriptions, alternative names, and Pleiades ids. If you add the polygons as features, you duplicate each of them under a second identifier. A text that mentions Sicilia then produces two candidates for the same province. They differ only in the way that the province is drawn, as a point or as an area. In this gazetteer, the provinces dataset has its place as the *boundaries* that locate other places. Step 8 uses it for this. It does not have a place as a second set of places.

Add a second block when its source contributes places that the first source does not have. Assume that your boundaries come from a dataset that has no counterpart in your main file. Then the block above is exactly what you write.

### Step 10: Use the Finished Gazetteer

The configuration is now complete. The end of this walkthrough shows it in full. The installation takes well under one minute and 0.6 GB of working disk space. It makes an artifact of about 23 MB with 42,242 features and 79,578 names. The `disk` key of the file declares this measured figure. Thus a build that has too little space says so before it starts and not halfway through:

``` bash
geoparser install pleiades.yaml
```

``` bash
geoparser list
```

Check it from the outside before you trust it. Look up places that you know. Confirm that the names, attributes, and coordinates are what you expect:

``` python
from geoparser import Gazetteer

gazetteer = Gazetteer("pleiades")

for feature in gazetteer.search("Sicilia", method="exact"):
    geometry = feature.geometry.geom_type if feature.geometry else "no geometry"
    print(
        feature.identifier,
        feature.data["title"],
        "|",
        feature.data["place_types"],
        "|",
        geometry,
    )
```

``` text
462492  Sicilia (island)          | island   | Point
981549  Sicilia (Roman province)  | province | Point
```

The gazetteer is now finished, and everything earlier in this guide applies to it. What it takes to resolve against it depends on the resolver. Each resolver uses a gazetteer in its own way. Some resolvers need information about your attributes. For example, the `SentenceTransformerResolver` describes candidates in words. You must give it the keys from which it builds that description. [modules](modules.md) documents what each resolver expects.

### Step 11: Expect to Retune the Modules

A finished gazetteer is not the end of the work. The developers did not select the default recognizer and the default resolver for your data. A gazetteer that is as far from the defaults as this one shows two mismatches immediately:

- **The recognizer can fail to find your placenames.** The developers trained the default spaCy model on contemporary news text. In *"Pliny describes the eruption that buried Pompeii and Herculaneum in Campania"*, it labels `Campania` as a place, but it misses `Pompeii` and `Herculaneum` completely. The contents of the gazetteer do not matter if the recognizer finds nothing to look up. Try a larger spaCy model. Try a model that is trained on your domain. Or give the spans yourself with a manual recognizer.
- **The threshold of the resolver can be tuned for other data.** The developers fine-tuned the default embedding model on GeoNames-style descriptions. Descriptions like `Campania (region) in Italia` are lower on its similarity scale than the default `min_similarity` of 0.6 expects. The correct candidate scores 0.55, and the resolver rejects it. Then the resolver widens its search and selects a worse candidate. If you lower the threshold to 0.45, the resolver resolves `Campania` correctly.

Neither mismatch is a fault in the configuration. The build output does not show either one. Thus resolve a few names for which you know the answer. When a result is wrong, examine the candidates and their scores. [modules](modules.md) describes the parameters. [training](training.md) describes how to fine-tune a resolver against your own gazetteer. This is the real solution. The developers optimized the default models for GeoNames. The recommended way to close the gap is to train on data that is annotated with the features of your gazetteer.

### The Complete Configuration

This is the complete configuration that the walkthrough uses:

[Download `pleiades.yaml`](https://github.com/NoeFlandre/geoparser/blob/main/docs/examples/pleiades.yaml)

The repository keeps this example beside the documentation. Thus you can validate it and reuse it without a copy of a large data artifact.

You have now used every mechanism of the configuration format: tabular and spatial sources, archives and plain files, attribute joins, chained joins, one-to-many joins, spatial joins across coordinate systems, derived names, aggregated attributes, and (at least on paper) a second feature block. The reference below gives the details.

## Configuration Reference

This section lists every key that the configuration format accepts. The walkthrough above is the best way to learn the format. Use this reference when you write your own configuration.

A configuration has two top-level lists. `sources` declares the files to read and what is in them. `features` declares how the rows of a source become places. Sources are temporary. The build stages them and then discards them. Thus the `features` blocks are what shape the gazetteer.

### Top-Level Keys

| Key        | Meaning                                                                                                                                                                                                                                                          |
|------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `name`     | The name of the gazetteer. Use it to install, query, and uninstall the gazetteer. It has letters, digits, underscores, and hyphens. If you install a configuration, it replaces any artifact with the same name.                                                  |
| `crs`      | The coordinate reference system in which the build stores all geometries. The default is `EPSG:4326`. The build reprojects sources in other systems into it at build time.                                                                                       |
| `disk`     | Optional. The free bytes that the gazetteers volume needs. The build checks this before it starts. Do not set it until you measure what a build costs. A guessed value either blocks builds that would work or does not catch builds that will fail.             |
| `sources`  | The list of source declarations. At least one is necessary.                                                                                                                                                                                                      |
| `features` | The list of feature blocks. At least one is necessary.                                                                                                                                                                                                           |

### Sources

A source is one file to read. It is **tabular** if it declares a `delimiter`. Otherwise it is **spatial**.

| Key          | Meaning                                                                                                                                                                                                                           |
|--------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `name`       | Identifies the source in the configuration. It is the name that you use to join the source. It must be a valid SQL identifier: letters, digits, and underscores. It must not start with a digit.                                  |
| `url`        | Where to download the file from. Exactly one of `url` or `path` is necessary. More than one source can name the same URL. The build then downloads it one time for each build. The build discards the downloads when it finishes.  |
| `path`       | A local file or directory in place of a download. Relative paths resolve against the directory of the configuration file. Thus you can move a configuration and its data together.                                                |
| `file`       | The file to read. When `url` or `path` points to a ZIP archive or a directory, the build searches it recursively for a file with this name. Otherwise it must match the name of the file.                                         |
| `sha256`     | Optional. The expected SHA-256 digest of the acquired file, as 64 hexadecimal characters. If it does not match, the build aborts.                                                                                                 |
| `delimiter`  | The field separator of a delimited text file (`","`, `"\t"`, `"|"`). If you set it, the source is tabular.                                                                                                                         |
| `quote`      | The quote character for tabular sources. The default is `"`. Set it to `""` to disable quote handling. Raw tab-separated exports need this. GeoNames files have unbalanced quote characters in ordinary values.                  |
| `skip_rows`  | The number of leading lines of a tabular file to discard. Use `1` for a header row and more for licence preambles. By default, the build skips nothing. A header row that you do not skip becomes a feature.                     |
| `crs`        | The coordinate reference system of the geometry and coordinates of this source. The default is the `crs` of the gazetteer. The build converts geometry when it reads the source. Thus the rest of the configuration uses one system. |
| `attributes` | The columns of the source. Each column has a `name` and a `type`.                                                                                                                                                                 |

`quote` and `skip_rows` describe a delimited text file. The build rejects them on a spatial source. (It cannot reject `delimiter`, because the declaration of `delimiter` makes a source tabular.) The attribute types are `text`, `integer`, `real`, and `geometry`. Two rules are different for the two kinds of source:

- A **tabular** source must declare **every** column, in the order of the file. It must not declare a `geometry` attribute. The declaration is the schema of the file. If the count does not match, the file cannot be parsed. The build does not drop columns.
- A **spatial** source can declare any **subset** of the fields of the file, in any order. It must declare exactly **one** attribute of type `geometry` with the name `geometry`.

More than one source can point to the same `url` with different `file` values. This is how you use an archive with many files. The build downloads it one time.

### Feature Blocks

Each block turns the rows of one source into features. A source backs a maximum of one block. The `source` name of the block is what appears as `feature.source` in the artifact.

| Key          | Meaning                                                                                                                       |
|--------------|-------------------------------------------------------------------------------------------------------------------------------|
| `source`     | The source whose rows this block projects.                                                                                    |
| `joins`      | Optional. A list of raw SQL join clauses that add columns from other sources to those rows.                                   |
| `identifier` | The stable identifier of the feature. It must read only the own source of the block. The build skips rows for which it evaluates to `NULL`. |
| `geometry`   | Optional. The geometry of the feature, as a geometry column or an expression that builds one. It must read only the own source of the block. |
| `names`      | One or more names. Each name is a column or an expression. At least one is necessary.                                         |
| `data`       | Optional. The attributes. Write each one as it appears in a SQL `SELECT` list.                                                |

Write blocks in reading order: source, joins, and then everything that derives from them.

### Values and Expressions

`identifier`, `geometry`, every `names` entry, and every `data` entry is a column reference or a scalar SQL expression. One rule applies to all of them: **a bare name is a column of the own source of the block. A column of a joined source is written `<alias>.<column>`.** The same rule applies inside join conditions. Thus nothing in a block needs a prefix for its own columns.

DuckDB evaluates the expressions. Thus you can use its [scalar function library](https://duckdb.org/docs/stable/sql/functions/overview): string manipulation, `CASE`, arithmetic, regular expressions, `ST_` spatial constructors, and subqueries over any declared source. These patterns occur often:

``` yaml
# Names
- "name"                                              # a column
- "unnest(string_split(alternatenames, ','))"         # one name per value of a multi-value column
- "regexp_replace(name, '\\s*\\(.*\\)', '')"          # strip a parenthesised qualifier
- "n.attested_form"                                   # a column of a joined source

# Geometry
- "geometry"                                          # a spatial source's geometry column
- "ST_Point(longitude, latitude)"                     # built from coordinate columns (longitude first)
- "ST_GeomFromText(geometry_wkt)"                     # parsed from a WKT text column

# Data
- "population"                                        # stored under its own name
- "c.Country AS country_name"                         # a joined column, renamed
- "upper(name) AS name_upper"                         # an expression (alias required)
- "'Roman province' AS place_types"                   # a constant
```

A `data` entry that is a plain column reference can omit the alias. The key is then the name of the column. Anything else has no name of its own. You must give it one. Two entries must not store the same key.

A name expression can return a list. Each element then becomes its own name. This is what `unnest` is for. The build drops names that are `NULL`, empty, or whitespace. It collapses duplicates. Thus you can write name expressions generously.

### Joins

A join is a raw SQL join clause that the build appends to the source of the block. The whole joined table becomes available. There is no separate list of columns to import. Reference what you need in `data`.

``` yaml
joins:
  # Attribute join: match on equal values
  - "LEFT JOIN countryInfo c ON country_code = c.ISO"
  # Match on an expression
  - "LEFT JOIN admin1CodesASCII a1 ON country_code || '.' || admin1_code = a1.code"
  # Chained join: reference a table joined earlier
  - "LEFT JOIN admin2Codes a2 ON a1.code || '.' || admin2_code = a2.code"
  # Spatial join
  - "LEFT JOIN municipalities g ON ST_Within(ST_Centroid(geometry), g.geometry)"
```

The build applies joins in order. A later clause can reference any table that an earlier clause brought in. This is how you express hierarchies with many levels (place, municipality, district, canton). Use `LEFT JOIN`. An inner join drops the rows that have no match. This silently removes places from your gazetteer.

Cardinality causes most of the surprises with joins, because it changes what `data` means. A many-to-one join is safe. A one-to-many join multiplies the rows of a place. The build collects `names` across all of them. But it takes each `data` value from the first row of the group. Among rows that a multiplication produced, this is arbitrary. Collect names with a one-to-many join. Aggregate attributes with a subquery.

Coordinate systems, on the other hand, need no work from you. The build converts the geometry of every source to the `crs` of the gazetteer when it reads the source. Thus a spatial join between sources in different systems needs nothing. There is one exception: a geometry that you construct from plain number columns, such as `ST_Point(x, y)` over a source whose `crs` is not the `crs` of the gazetteer. The build converts it when it is the `geometry` of a feature. Inside a join condition, you must put `ST_Transform` around it yourself.

### Duplicate Identifiers

In one block, rows that have the same identifier are **merged into one feature**. The build collects all their names. It unites their geometries into one geometry that can have many parts. It takes each data value from the first row. This is automatic. In this way, datasets that spread a place over several records (multi-part geometries, one row for each name) become one place.

``` none
p1  North Summit  800     →  one feature "p1", names {North Summit, South Summit},
p1  South Summit  1200       height 800, geometry MultiPoint of both rows
p2  Lone Hill     300      →  one feature "p2"
```

Across blocks, this is an error. Identifiers must be unique in the whole gazetteer. A collision fails the build. The error names the identifier and the blocks that it came from. Add a namespace with an expression (`"'province:' || fid"`) or merge the blocks.

### What the Format Does Not Do

If you know the limits, you do not waste time to look for keys that do not exist:

- **There is no row filter.** A block has no `where`. To exclude rows, make the `identifier` evaluate to `NULL` for them. The build skips rows that have no identifier: `identifier: "CASE WHEN feature_class <> 'X' THEN id END"`. To filter *joined* rows, add the condition to the join: `"LEFT JOIN names n ON id = n.place_id AND n.association_certainty = 'certain'"`.
- **`identifier` and `geometry` must come from the own source of the block.** Only `names` and `data` can read joined columns. The identity and the location of a place are properties of its own record. Joins exist to describe a place. They do not decide which places exist. When the build compiles the feature blocks, it rejects a qualified reference in either key with an explicit message. This occurs after the build prepared the sources. Thus, on a large dataset, the build downloads and stages the files before you see the error.
- **One block for each source.** To project one file into two kinds of feature, declare it two times under different source names.
- **No user code.** You can only use SQL expressions that the build evaluates. If you need real preprocessing, do it before the build. Put the result in a file that you reference with `path`.
- **Names have no order and no label.** The search index has no preferred name and no language of a name. If you need them, store them in `data`.
- **The build ignores unrecognized keys and does not report them.** It silently drops a key that the format does not know. This includes a misspelling of a key that it knows. Thus `skiprows: 1` validates and skips nothing. If a setting seems to have no effect, first check its spelling against the tables above.

### Installing and Iterating

To install a configuration, give the file to the same command, in place of a pre-configured name:

``` bash
geoparser install path/to/my_gazetteer.yaml
```

It then behaves exactly like a pre-configured gazetteer:

``` bash
geoparser list
```

``` bash
geoparser uninstall my_gazetteer
```

The build validates the configuration, gets the files, runs the projections, and writes the artifact. If anything is wrong, the build stops with a message and installs nothing. A successful build atomically replaces any previous artifact with the same name. Thus you can safely iterate on a configuration.

The build discards downloaded files when it finishes. Thus each rebuild fetches them again. While you still change a configuration, download the files one time by hand. Point the sources at them with `path` in place of `url`. Each iteration then costs only the processing time.

> [!NOTE]
> To build a gazetteer with geometries, you need the spatial extension of DuckDB. The build fetches it automatically the first time. To build offline, first run one spatial build while you are connected. The extension is then cached.

### Further Examples

The build of the pre-configured gazetteers is exactly the same. Read their files when your own gazetteer works: [geonames.yaml](https://github.com/NoeFlandre/geoparser/blob/main/geoparser/gazetteer/configs/geonames.yaml) (a large tabular dataset with four lookup joins) and [swissnames3d.yaml](https://github.com/NoeFlandre/geoparser/blob/main/geoparser/gazetteer/configs/swissnames3d.yaml) (six spatial sources, chained spatial joins, and merging of multi-part geometries).
