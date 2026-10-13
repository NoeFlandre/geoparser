#!/usr/bin/env bash
# Prepare the same GeoNames database bytes on each resumable benchmark job.

set -euo pipefail

: "${GEOPARSER_GAZETTEERS_DIR:?Set the node-local gazetteer directory}"
: "${GEOPARSER_GAZETTEER_ARTIFACT_DIR:?Set the durable artifact directory}"

installed="$GEOPARSER_GAZETTEERS_DIR/geonames.db"
artifact="$GEOPARSER_GAZETTEER_ARTIFACT_DIR/geonames.db"
mkdir -p "$GEOPARSER_GAZETTEERS_DIR" "$GEOPARSER_GAZETTEER_ARTIFACT_DIR"

if [[ ! -s "$artifact" ]]; then
    if [[ ! -s "$installed" ]]; then
        python -m geoparser install geonames
    fi
    if [[ ! -s "$installed" ]]; then
        echo "GeoNames install did not create $installed" >&2
        exit 1
    fi
    staged="$GEOPARSER_GAZETTEER_ARTIFACT_DIR/.geonames.db.$$.tmp"
    cp "$installed" "$staged"
    mv "$staged" "$artifact"
fi

if [[ ! -s "$artifact" ]]; then
    echo "Persistent GeoNames artifact is empty: $artifact" >&2
    exit 1
fi
cp "$artifact" "$installed"
size="$(wc -c < "$artifact" | tr -d '[:space:]')"
echo "GeoNames artifact: $artifact (${size} bytes)"
