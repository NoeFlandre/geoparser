# Installation

To set up the Irchel Geoparser, do two steps. First, install the Python package. Then install a gazetteer. The resolver uses the gazetteer to resolve place names. This page describes both steps. You need both before you can parse a text.

The package needs **Python 3.10 or newer** and about **3 GB of disk space**. PyTorch uses most of this space. A gazetteer needs much more space. The quantity depends on the gazetteer that you choose.

## Installing the Package

We recommend that you install into a virtual environment. The library and its dependencies then cannot change anything else on your machine.

=== "macOS / Linux"

    ```bash
    python3 -m venv geoparser-env
    ```

    ```bash
    source geoparser-env/bin/activate
    ```

    ```bash
    pip install geoparser
    ```

=== "Windows"

    ```powershell
    python -m venv geoparser-env
    ```

    ```powershell
    geoparser-env\Scripts\activate
    ```

    ```powershell
    pip install geoparser
    ```

The installation downloads about 3 GB and takes a few minutes. While pip resolves the versions, a quiet pause is normal.

When the environment is active, your prompt starts with `(geoparser-env)`. The environment is active in that terminal only. Activate it again in each new terminal. If you do not, you can get the error `ModuleNotFoundError: No module named 'geoparser'` after a successful installation. To leave the environment, run `deactivate`.

To make sure that the package is installed and available, run this command:

``` bash
geoparser list
```

``` text
No gazetteers installed.
```

This output is correct at this point. The package is installed, but there is no gazetteer yet.

## Installing a Gazetteer

A geoparser answers two questions about a text. Which words are place names? Which places do these names refer to? The Irchel Geoparser answers the second question with a **gazetteer**. A gazetteer is a database of places with their names, coordinates, and attributes. The resolver selects from the entries of the gazetteer. To recognize that "Springfield" is a place name, you need only the text. To decide which Springfield it is, and to give it coordinates, you must select from a set of candidate places.

Therefore a gazetteer is a required part of the setup. Some approaches predict coordinates directly from the text. They do not use a set of candidates. This library does not use this design. Nothing resolves until you install a gazetteer.

Gazetteers are large. They come from third parties and have their own licences. The best gazetteer depends on what you study. For these reasons, the package does not include a gazetteer. You install one yourself. You do this once with one command.

### Choosing One

The library has ready-made configurations for two gazetteers. You can install either of them without a configuration of your own. Install **GeoNames** unless you have a specific reason not to. It is global and it covers all types of places. The default models of the library are tuned for it. Choose **SwissNames3D** if your material is Swiss. It describes that one country in much more detail than GeoNames. You can install both. You choose between them for each resolver. They do not affect each other.

=== "GeoNames"

    **The recommended choice.** A global gazetteer with more than 13 million names. It covers countries, administrative divisions, cities, towns, and neighbourhoods. It also covers natural features, such as mountains and rivers, and points of interest, such as buildings and monuments. The default resolver models of the library are fine-tuned on it. This documentation assumes it everywhere.

    - **Website**: [geonames.org](https://www.geonames.org/)
    - **Licence**: CC BY 4.0
    - **Coverage**: Global, all place types
    - **Disk space**: **30.7 GB of free space is necessary during the installation**. The installed file is about **10.2 GB**.
    - **Installation time**: about **10 to 15 minutes**. The time depends on the hardware and the network.

    ```bash
    geoparser install geonames
    ```

    The coverage is different in different regions. Some parts of the world have much more detail than others.

=== "SwissNames3D"

    **For Swiss material.** The official register of Swiss place names from Swisstopo, the Federal Office of Topography. In Switzerland, it is much more detailed than GeoNames. It has detailed feature classifications and full geometries (points, lines, and polygons). Each feature is linked to its municipality, district, and canton.

    - **Website**: [Swisstopo SwissNames3D](https://www.swisstopo.admin.ch/en/landscape-model-swissnames3d)
    - **Coverage**: Switzerland only
    - **Disk space**: **3.5 GB of free space is necessary during the installation**. The installed file is about **0.7 GB**.
    - **Installation time**: about **1 to 2 minutes**

    ```bash
    geoparser install swissnames3d
    ```

    The attribute names are in German (`NAME`, `OBJEKTART`, `KANTON_NAME`). The default resolver models trained on English text against GeoNames. Therefore, decrease `min_similarity`. If possible, fine-tune the model. Read [training modules](guides/training.md).

If you work on a region, a period, or a domain that neither gazetteer covers, build a gazetteer from your own data. Read [custom gazetteers](guides/custom-gazetteers.md).

### Running the Install

This example uses GeoNames:

``` bash
geoparser install geonames
```

The command downloads the source data and transforms it. It builds one self-contained file. It reports three stages:

``` text
─────────────────────────────────── geonames ────────────────────────────────────

Prepared sources                          ━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:08:42
Compiled features                         ━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:03:11
Built artifact                            ━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:02:35

Summary

Features  13,041,196
Names     19,483,772
```

When the summary prints, the gazetteer is installed and ready.

The build needs much more disk space than the finished file. The command stages the source data before it makes the file smaller. Before the build starts, the command checks the free space. When the build ends, the command deletes the intermediate files. The build also needs about 4 GB of RAM. Close other heavy applications. If a build fails, the command leaves a previously installed gazetteer with the same name as it was. It is safe to run the command again.

If the gazetteer is already installed, `install` tells you and stops. To rebuild it, add `--force`. The option `--keep-downloads` keeps the downloaded source files in the `.downloads` folder of the data directory. A later rebuild then does not download them again. An unknown name makes the command exit with status 2 and list the built-in gazetteers. A failed build prints one line and exits with status 1. Add `--verbose` to see the full traceback.

After you install a gazetteer, `list` shows it with its size on disk:

``` bash
geoparser list
```

``` text
geonames  (9700.7 MB)
```

The size changes when the upstream data changes. Use the size only as an indication. To remove a gazetteer that you no longer need, run this command:

``` bash
geoparser uninstall geonames
```

``` text
Remove gazetteer 'geonames'? [y/N]: y
Removed gazetteer 'geonames'.
```

To skip the confirmation, for example in scripts, add `--yes`.

### Checking the Gazetteer

Before you write pipeline code, make sure that you can query the gazetteer:

``` python
from geoparser import Gazetteer

gazetteer = Gazetteer("geonames")
results = gazetteer.search("Paris", method="exact")

print(f"{len(results)} places named Paris")
for feature in results[:3]:
    print(" ", feature.data.get("name"), "|", feature.data.get("country_name"))
```

``` text
122 places named Paris
  Paris | Spain
  Paris | Spain
  Paris | Armenia
```

The number of results is correct. It is also correct that the French capital is not at the top of the list. Place names are ambiguous. GeoNames has 122 different places with the name Paris. `search()` returns all of them in no specific order. It does not know which place is the most prominent. A resolver selects between candidates like these. It uses the context of the name. The [concepts](concepts.md) page describes this.

The setup is now complete. Parse your first text in the [quickstart](quickstart.md). The other sections on this page are optional.

## Using a GPU

Everything in this documentation works on a CPU. If you have an NVIDIA GPU, recognition and resolution run much faster on it. On Linux and Windows, the PyTorch build that `pip` selects normally has CUDA enabled. To make sure that your GPU is visible, run this command:

``` bash
python -c "import torch; print(torch.cuda.is_available())"
```

If the command prints `False` and you have a compatible card, install PyTorch again. Follow the instructions for your CUDA version on the PyTorch [Get Started](https://pytorch.org/get-started/locally/) page. On Apple Silicon, PyTorch uses the Metal backend. No extra step is necessary.

## Working in Jupyter

Install Jupyter into the same environment. If you do not, the notebook runs with a different Python and does not find the library:

``` bash
pip install jupyter
```

``` bash
python -m ipykernel install --user --name geoparser-env --display-name "Python (geoparser)"
```

``` bash
jupyter lab
```

Then select the "Python (geoparser)" kernel. In the notebook, `import sys; print(sys.executable)` must print a path inside `geoparser-env`.

## Where Data Is Stored

The library keeps gazetteers and project data outside your working directory. It uses the standard location for application data of your operating system:

| Platform    | Location                                       |
|-------------|------------------------------------------------|
| **Windows** | `C:\Users\<Username>\AppData\Local\geoparser\` |
| **macOS**   | `~/Library/Application Support/geoparser/`     |
| **Linux**   | `~/.local/share/geoparser/`                    |

Each gazetteer is one self-contained file under `gazetteers/`. Your projects, documents, and results are separate. They are in `geoparser.db`. Because they are separate, if you install a gazetteer again, your projects do not change. If you delete a project, your gazetteers do not change. To back up a gazetteer or to move it to another machine, copy its file.

To store application data in a different place, set `GEOPARSER_DATA_DIR` to the base directory. Do this before you install gazetteers or start the application:

```bash
export GEOPARSER_DATA_DIR=/path/to/geoparser-data
python -m geoparser install geonames
```

The library stores the gazetteer artifacts in `gazetteers/`. It stores the project database in `geoparser.db`. It stores the annotator database in `annotator/annotator.db`. All of them are in that directory. The library reads the variable when a database engine is first used. Set it before you start the process. `GEOPARSER_GAZETTEERS_DIR` can still override the gazetteer directory only.

## Upgrading

``` bash
pip install --upgrade geoparser
```

> [!WARNING]
> The database format is not stable between releases. Your project database can be from an older version. In this case, the library refuses to open it and tells you. There is no automatic migration yet. You must delete `geoparser.db`. This deletes the stored projects and results. Export all data that you want to keep before you delete it. Read [working with results](guides/results.md).

To remove everything, delete the environment folder. To remove the gazetteers and projects also, delete the data directory above. The library installs nothing in other places.
