# ADR 0002: CPU-only PyTorch in Linux containers

## Status

Accepted — 2026-09-11

## Context

The default Linux PyTorch distribution pulls CUDA runtime libraries. The container has no GPU contract. In the first smoke build, these libraries filled the small local Docker disk. The build could not commit the image.

## Decision

Resolve `torch` from the explicit PyTorch CPU index on Linux. Keep the platform-native PyPI resolution on macOS and Windows. The lockfile stays the single source of truth for all platforms. The container continues to use `uv sync --locked`.

## Tradeoffs

CPU containers cannot use CUDA acceleration. This is correct for the current runtime image. The image gives the CLI and the library. It does not promise GPU scheduling. A GPU deployment can use a separate environment and index policy. The default image does not become larger.

## Consequences

Linux CI and the Docker smoke tests use a much smaller and deterministic runtime. The PyTorch index configuration is now part of the dependency contract. Review it again if GPU-backed container execution becomes a supported deployment target.
