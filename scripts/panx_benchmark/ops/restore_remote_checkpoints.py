from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path, PurePosixPath

from huggingface_hub import HfApi, hf_hub_download


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Restore one PAN-X snapshot from the HF results dataset."
    )
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--remote-root",
        default="runs/panx/2026-10-02-cd0c750-transformers-5-18-cpu/checkpoints",
    )
    parser.add_argument("--repo-id", default="NoeFlandre/geoparser-benchmark-results")
    parser.add_argument("--revision", help="HF dataset commit; default is current main")
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("/workspace/hf-cache/hub")
    )
    arguments = parser.parse_args()

    api = HfApi()
    revision = (
        arguments.revision
        or api.repo_info(repo_id=arguments.repo_id, repo_type="dataset").sha
    )
    prefix = f"{arguments.remote_root.rstrip('/')}/{arguments.snapshot_id}/"
    files = sorted(
        path
        for path in api.list_repo_files(
            repo_id=arguments.repo_id,
            repo_type="dataset",
            revision=revision,
        )
        if path.startswith(prefix) and not path.endswith("/")
    )
    if not files:
        parser.error(f"No remote checkpoint files found under {prefix!r}")

    remote_root = PurePosixPath(arguments.remote_root)
    for remote_path in files:
        relative = PurePosixPath(remote_path).relative_to(remote_root)
        destination = arguments.checkpoint_dir.joinpath(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        downloaded = Path(
            hf_hub_download(
                repo_id=arguments.repo_id,
                filename=remote_path,
                repo_type="dataset",
                revision=revision,
                cache_dir=str(arguments.cache_dir),
            )
        )
        temporary = destination.with_suffix(destination.suffix + ".restore-tmp")
        shutil.copyfile(downloaded, temporary)
        temporary.replace(destination)
        print(f"restored {remote_path} sha256={sha256(destination)}")
    print(f"restored_files={len(files)} revision={revision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
