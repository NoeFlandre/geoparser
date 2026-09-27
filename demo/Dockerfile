FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.16 /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    GEOPARSER_DATA_DIR=/data/geoparser \
    HF_HOME=/data/hf \
    HF_HUB_CACHE=/data/hf/hub \
    HF_DATASETS_CACHE=/data/hf/datasets \
    PYTHONUNBUFFERED=1

# Install the checked-out project and demo tools from the committed lockfile.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --locked --no-dev --no-install-project --group demo

COPY geoparser ./geoparser
# en_core_web_trf needs the spacy-curated-transformers>=0.3.1,<1 plugin, which
# the locked project dependencies install on this Python 3.12 image.
RUN uv sync --locked --no-dev --group demo \
    && uv run --no-sync python -m spacy download en_core_web_trf

WORKDIR /workspace
COPY demo/demo.ipynb ./demo.ipynb
RUN groupadd --gid 1000 geoparser \
    && useradd --uid 1000 --gid 1000 --create-home --shell /usr/sbin/nologin geoparser \
    && mkdir -p /data/geoparser /data/hf /workspace \
    && chown -R 1000:1000 /data /workspace

VOLUME ["/data"]
EXPOSE 8888
USER 1000:1000

CMD ["sh", "-c", "if [ -z \"${JUPYTER_TOKEN:-}\" ]; then echo 'JUPYTER_TOKEN is required' >&2; exit 2; fi; exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --ServerApp.token=\"${JUPYTER_TOKEN}\" --ServerApp.root_dir=/workspace"]
