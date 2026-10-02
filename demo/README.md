# Irchel Geoparser Demo

This notebook extracts place names from *Around the World in Eighty Days* by Jules Verne and shows them on a map. The image is built locally from this checkout and its locked project dependencies. The image does not contain a GeoNames gazetteer. The gazetteer and the Hugging Face cache use the same persistent volume as the runtime image.

## Build and Run

From the repository root, set a Jupyter token and build the demo image:

```bash
export JUPYTER_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe())')"
docker compose --profile demo build demo
```

Install GeoNames into the shared volume. Then start the notebook:

```bash
docker compose --profile install run --rm install
docker compose --profile demo up demo
```

Open [http://localhost:8888/lab/tree/demo.ipynb](http://localhost:8888/lab/tree/demo.ipynb). Enter the value of `JUPYTER_TOKEN` when the page asks for it. The token is required. If it is missing, the container exits with an error. Set `HF_TOKEN` in the shell only if a Hugging Face resource that you use needs authentication.

The notebook uses the `en_core_web_trf` spaCy model that is installed in the image. This transformer pipeline needs the `spacy-curated-transformers` plugin. The image installs the plugin from the lockfile. The plugin has no Python 3.14 release yet. Outside the image, on that interpreter, use a non-transformer model for the requested language, for example `en_core_web_lg` for English.

The install command above downloads GeoNames into the named volume one time. The image does not contain the gazetteer. `docker compose down` stops the services and keeps the named data volume.

Refer to the project [README](../README.md) for the Python workflow, the CLI workflow, the citation metadata, the data paths, and the runtime container.
