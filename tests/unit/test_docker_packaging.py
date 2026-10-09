"""Acceptance checks for the runtime and demo container documentation."""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest
import yaml

from tests.conftest import PROJECT_ROOT

ROOT = PROJECT_ROOT


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
    def test_runtime_image_runs_as_unprivileged_user(self):
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        assert re.search(r"(?m)^USER\s+1000(?::1000)?\s*$", dockerfile)

    def test_runtime_image_persists_geoparser_data(self):
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        assert re.search(r"(?m)^VOLUME\s+(?:\[\s*)?\"?/data", dockerfile)
        assert "GEOPARSER_DATA_DIR=/data/geoparser" in dockerfile

    def test_runtime_image_persists_model_cache_locations(self):
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        assert "HF_HOME=/data/hf" in dockerfile
        assert "HF_HUB_CACHE=/data/hf/hub" in dockerfile
        assert "HF_DATASETS_CACHE=/data/hf/datasets" in dockerfile

    def test_runtime_image_exposes_the_annotator_command(self):
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        assert re.search(r"(?m)^EXPOSE\s+8000\s*$", dockerfile)
        assert re.search(r"(?m)^ENTRYPOINT\s+", dockerfile)
        assert re.search(r"(?m)^CMD\s+", dockerfile)

    @pytest.mark.parametrize(
        ("package", "source"),
        [
            # The annotator's session form lists installed models.
            ("en_core_web_sm", "en-core-web-sm"),
            # The default parse resolver loads the sentence splitter.
            ("xx_sent_ud_sm", "xx-sent-ud-sm"),
        ],
    )
    def test_installs_the_locked_spacy_models_it_needs(self, package, source):
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        url = re.search(rf'{source} = \{{ url = "([^"]+)" \}}', pyproject)

        assert url is not None
        assert f"{package} @ {url.group(1)}" in dockerfile

    def test_demo_installs_the_locked_sentence_splitter(self):
        """The notebook's resolver loads xx_sent_ud_sm when it is built."""
        dockerfile = (ROOT / "demo" / "Dockerfile").read_text(encoding="utf-8")
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        url = re.search(r'xx-sent-ud-sm = \{ url = "([^"]+)" \}', pyproject)

        assert url is not None
        assert f"xx_sent_ud_sm @ {url.group(1)}" in dockerfile

    def test_compose_install_and_annotator_share_the_named_volume(self):
        compose = yaml.safe_load((ROOT / "compose.yaml").read_text(encoding="utf-8"))
        services = compose["services"]
        install = services["install"]
        annotator = services["annotator"]

        assert install["volumes"] == annotator["volumes"]
        assert install["volumes"] == ["geoparser-data:/data"]
        assert "geoparser-data" in compose["volumes"]

    def test_compose_annotator_binds_loopback_and_uses_expected_arguments(self):
        compose = yaml.safe_load((ROOT / "compose.yaml").read_text(encoding="utf-8"))
        annotator = compose["services"]["annotator"]
        assert annotator["ports"] == ["127.0.0.1:8000:8000"]
        assert annotator["command"] == [
            "annotator",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--no-browser",
        ]

    def test_compose_install_and_annotator_accept_the_huggingface_token(self):
        compose = yaml.safe_load((ROOT / "compose.yaml").read_text(encoding="utf-8"))
        services = compose["services"]
        install = services["install"]
        annotator = services["annotator"]
        for service in (install, annotator):
            assert service["environment"]["HF_TOKEN"] == "${HF_TOKEN:-}"


@pytest.mark.unit
class TestDemoImage:
    def test_demo_image_builds_a_locked_checkout_without_geonames(self):
        dockerfile = (ROOT / "demo/Dockerfile").read_text(encoding="utf-8")
        assert "COPY pyproject.toml uv.lock" in dockerfile
        assert "uv sync --locked" in dockerfile
        assert "COPY geoparser ./geoparser" in dockerfile
        assert "install geonames" not in dockerfile

    @pytest.mark.parametrize(
        "required_pattern",
        (
            r"(?m)^USER\s+1000(?::1000)?\s*$",
            r'VOLUME \["/data"\]',
            r"GEOPARSER_DATA_DIR=/data/geoparser",
            r"HF_HOME=/data/hf",
            r"JUPYTER_TOKEN",
        ),
    )
    def test_demo_image_runs_unprivileged_with_a_persistent_data_volume(
        self, required_pattern
    ):
        dockerfile = (ROOT / "demo/Dockerfile").read_text(encoding="utf-8")
        assert re.search(required_pattern, dockerfile)

    def test_demo_image_keeps_the_transformer_and_omits_build_tools(self):
        dockerfile = (ROOT / "demo/Dockerfile").read_text(encoding="utf-8")
        assert "en_core_web_trf" in dockerfile
        assert "en_core_web_sm" not in dockerfile
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
    def test_cli_annotator_example_stays_on_loopback(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        command = next(
            line.strip()
            for line in readme.splitlines()
            if line.strip().startswith("geoparser annotator")
        )
        arguments = shlex.split(command)

        if "--host" in arguments:
            assert arguments[arguments.index("--host") + 1] == "127.0.0.1"

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

    def test_docs_do_not_point_at_the_upstream_demo_image(self):
        pages = [
            ROOT / "README.md",
            *ROOT.glob("demo/*.md"),
            *ROOT.glob("docs/**/*.md"),
        ]
        stale = [
            page.relative_to(ROOT)
            for page in pages
            if "dguzh/geoparser-demo" in page.read_text(encoding="utf-8")
        ]

        assert not stale, f"pages reference the upstream demo image: {stale}"
