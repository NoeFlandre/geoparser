# Irchel Geoparser Demo

This notebook extracts and maps place names mentioned in Jules Verne's *Around the World in Eighty Days*. The image is built locally from this checkout and its locked project dependencies. It does not contain a GeoNames gazetteer; the gazetteer and Hugging Face cache use the same persistent volume as the runtime image.

## Build and Run

From the repository root, set a Jupyter token and build the demo image:

```bash
export JUPYTER_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe())')"
docker compose --profile demo build demo
```

Install GeoNames into the shared volume, then start the notebook:

```bash
docker compose --profile install run --rm install
docker compose --profile demo up demo
```

Open [http://localhost:8888/lab/tree/demo.ipynb](http://localhost:8888/lab/tree/demo.ipynb) and enter the value of `JUPYTER_TOKEN` when prompted. The token is required; the container exits with an error if it is missing. Set `HF_TOKEN` in the shell only if a Hugging Face resource you use requires authentication.

The notebook uses the `en_core_web_trf` spaCy model installed in the image. That transformer pipeline needs the `spacy-curated-transformers` plugin, which the image installs from the lockfile. The plugin has no Python 3.14 release yet; outside the image on that interpreter, use a non-transformer model for the requested language, for example `en_core_web_lg` for English. The install command above downloads GeoNames into the named volume once; the gazetteer is not baked into the image. `docker compose down` stops the services and keeps the named data volume.

See the project [README](../README.md) for the Python and CLI workflows, citation metadata, data paths, and runtime container.
