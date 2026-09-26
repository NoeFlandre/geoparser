"""Acceptance checks for the runtime and demo container documentation."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def _links(page: Path) -> list[Path]:
    """Return local Markdown link targets from a public README."""
    text = page.read_text(encoding="utf-8")
    targets = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text)
    paths = []
    for target in targets:
        target = target.strip().split(maxsplit=1)[0].strip("<>")
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        paths.append(page.parent / unquote(parsed.path))
    return paths


@pytest.mark.unit
class TestRuntimeImage:
    def test_runs_as_uid_1000_and_persists_data_and_huggingface_cache(self):
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

        assert re.search(r"(?m)^USER\s+1000(?::1000)?\s*$", dockerfile)
        assert re.search(r"(?m)^VOLUME\s+(?:\[\s*)?\"?/data", dockerfile)
        assert re.search(r"(?m)^EXPOSE\s+8000\s*$", dockerfile)
        assert "GEOPARSER_DATA_DIR=/data/geoparser" in dockerfile
        assert "HF_HOME=/data/hf" in dockerfile
        assert "HF_HUB_CACHE=/data/hf/hub" in dockerfile
        assert "HF_DATASETS_CACHE=/data/hf/datasets" in dockerfile
        assert re.search(r"(?m)^ENTRYPOINT\s+", dockerfile)
        assert re.search(r"(?m)^CMD\s+", dockerfile)

    def test_compose_install_and_annotator_share_persistent_data(self):
        compose = yaml.safe_load((ROOT / "compose.yaml").read_text(encoding="utf-8"))
        services = compose["services"]
        install = services["install"]
        annotator = services["annotator"]

        assert install["volumes"] == annotator["volumes"]
        assert install["volumes"] == ["geoparser-data:/data"]
        assert "geoparser-data" in compose["volumes"]
        assert "8000:8000" in annotator["ports"]
        assert annotator["command"] == [
            "annotator",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--no-browser",
        ]
        for service in (install, annotator):
            assert service["environment"]["HF_TOKEN"] == "${HF_TOKEN:-}"


@pytest.mark.unit
class TestDemoImage:
    def test_builds_locked_checkout_without_baking_geonames(self):
        dockerfile = (ROOT / "demo/Dockerfile").read_text(encoding="utf-8")

        assert "COPY pyproject.toml uv.lock" in dockerfile
        assert "uv sync --locked" in dockerfile
        assert "COPY geoparser ./geoparser" in dockerfile
        assert "install geonames" not in dockerfile
        assert re.search(r"(?m)^USER\s+1000(?::1000)?\s*$", dockerfile)
        assert 'VOLUME ["/data"]' in dockerfile
        assert "GEOPARSER_DATA_DIR=/data/geoparser" in dockerfile
        assert "HF_HOME=/data/hf" in dockerfile
        assert "JUPYTER_TOKEN" in dockerfile
        assert "en_core_web_trf" in dockerfile
        assert "en_core_web_sm" not in dockerfile
        assert "xx_sent_ud_sm" not in dockerfile
        assert "build-essential" not in dockerfile
        assert "curl" not in dockerfile

    def test_compose_demo_reuses_data_volume_and_requires_a_token(self):
        compose = yaml.safe_load((ROOT / "compose.yaml").read_text(encoding="utf-8"))
        demo = compose["services"]["demo"]

        assert demo["volumes"] == ["geoparser-data:/data"]
        assert "8888:8888" in demo["ports"]
        assert demo["environment"]["JUPYTER_TOKEN"] == "${JUPYTER_TOKEN:-}"


@pytest.mark.unit
class TestDockerSmokeWorkflow:
    def test_ci_builds_both_images_and_checks_the_required_token(self):
        workflow = yaml.safe_load(
            (ROOT / ".github/workflows/docker.yml").read_text(encoding="utf-8")
        )
        steps = workflow["jobs"]["docker-smoke"]["steps"]
        commands = "\n".join(step.get("run", "") for step in steps)

        assert "docker build --file Dockerfile" in commands
        assert "docker build --file demo/Dockerfile" in commands
        assert "JUPYTER_TOKEN is required" in commands


@pytest.mark.unit
class TestReadmeLinks:
    def test_readme_documents_the_requested_sections(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        headings = {
            heading.strip().casefold()
            for heading in re.findall(r"(?m)^##\s+(.+?)\s*$", readme)
        }

        assert {"citation", "cli", "data paths", "outputs", "docker"} <= headings

    @pytest.mark.parametrize("readme", [ROOT / "README.md", ROOT / "demo/README.md"])
    def test_local_links_resolve(self, readme: Path):
        missing = [target for target in _links(readme) if not target.exists()]

        assert not missing, f"{readme.relative_to(ROOT)} has broken links: {missing}"
