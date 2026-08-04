"""Capture GPU, Docker image, and package versions for a baseline run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--vllm-version", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output", default="results/runtime/runtime_manifest.json")
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> str:
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    git_status = run(["git", "status", "--porcelain"])
    if git_status and not args.allow_dirty:
        raise RuntimeError("refusing to capture an official run from a dirty Git worktree")

    image_details = json.loads(run(["docker", "image", "inspect", args.image]))[0]
    container_details = json.loads(
        run(["docker", "container", "inspect", "medtrace-vllm"])
    )[0]
    container_command = container_details["Config"].get("Cmd") or []
    if "--revision" not in container_command:
        raise RuntimeError("running container command does not pin --revision")
    revision_index = container_command.index("--revision") + 1
    if revision_index >= len(container_command) or container_command[revision_index] != args.model_revision:
        raise RuntimeError("running container uses an unexpected model revision")
    package_probe = (
        "import json, torch, transformers, vllm; "
        "print(json.dumps({'torch': torch.__version__, "
        "'transformers': transformers.__version__, 'vllm': vllm.__version__}))"
    )
    package_versions = json.loads(
        run(
            [
                "docker",
                "exec",
                "medtrace-vllm",
                "python",
                "-c",
                package_probe,
            ]
        )
    )
    if package_versions["vllm"] != args.vllm_version:
        raise RuntimeError(
            f"image reports vLLM {package_versions['vllm']}, "
            f"expected {args.vllm_version}"
        )

    gpu_csv = run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": run(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(git_status),
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "requested_image": args.image,
        "image": {
            "id": image_details.get("Id"),
            "repo_digests": image_details.get("RepoDigests"),
            "architecture": image_details.get("Architecture"),
            "os": image_details.get("Os"),
            "size_bytes": image_details.get("Size"),
        },
        "container": {
            "id": container_details.get("Id"),
            "image": container_details["Config"].get("Image"),
            "command": container_command,
            "started_at": container_details["State"].get("StartedAt"),
        },
        "packages": package_versions,
        "gpus_csv": gpu_csv.splitlines(),
        "docker": json.loads(run(["docker", "version", "--format", "{{json .}}"])),
        "host": platform.platform(),
    }
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    write_json(output_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Runtime manifest saved to: {output_path}")


if __name__ == "__main__":
    main()
