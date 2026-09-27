# Reproducible runtime image for the CLI and library.
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

# Install locked runtime dependencies before copying source so dependency
# layers remain reusable when application code changes.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --locked --no-dev --no-install-project

COPY geoparser ./geoparser
# The annotator lists installed spaCy models when a session is created, and
# the runtime venv has no pip for `spacy download`, so install the small
# English model (the one the test group locks) with uv.
RUN uv sync --locked --no-dev \
    && uv pip install --python /opt/venv/bin/python --no-deps \
        "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0.tar.gz"

RUN groupadd --gid 1000 geoparser \
    && useradd --uid 1000 --gid 1000 --create-home --shell /usr/sbin/nologin geoparser \
    && mkdir -p /data/geoparser /data/hf \
    && chown -R 1000:1000 /data

VOLUME ["/data"]
EXPOSE 8000
USER 1000:1000

ENTRYPOINT ["python", "-m", "geoparser"]
CMD ["--help"]
