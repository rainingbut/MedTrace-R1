"""Capture a reproducible Docker or native vLLM baseline runtime."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("docker", "native"), default="docker")
    parser.add_argument("--image")
    parser.add_argument("--pid-file", default="results/runtime/vllm-server.pid")
    parser.add_argument("--requirements-file", default="requirements.txt")
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_server_command(
    command: list[str], model_id: str, model_revision: str
) -> None:
    if "--revision" not in command:
        raise RuntimeError("running server command does not pin --revision")
    revision_index = command.index("--revision") + 1
    if revision_index >= len(command) or command[revision_index] != model_revision:
        raise RuntimeError("running server uses an unexpected model revision")

    model_is_pinned = False
    if "--model" in command:
        model_index = command.index("--model") + 1
        model_is_pinned = model_index < len(command) and command[model_index] == model_id
    if "serve" in command:
        serve_index = command.index("serve") + 1
        model_is_pinned = model_is_pinned or (
            serve_index < len(command) and command[serve_index] == model_id
        )
    if not model_is_pinned:
        raise RuntimeError("running server uses an unexpected model id")


def package_versions(command_prefix: list[str]) -> dict[str, str]:
    probe = (
        "import json, torch, transformers, vllm; "
        "print(json.dumps({'torch': torch.__version__, "
        "'transformers': transformers.__version__, 'vllm': vllm.__version__}))"
    )
    return json.loads(run([*command_prefix, "-c", probe]))


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def docker_runtime(args: argparse.Namespace) -> dict[str, Any]:
    if not args.image:
        raise ValueError("--image is required for the Docker backend")
    image_details = json.loads(run(["docker", "image", "inspect", args.image]))[0]
    container_details = json.loads(
        run(["docker", "container", "inspect", "medtrace-vllm"])
    )[0]
    command = container_details["Config"].get("Cmd") or []
    validate_server_command(command, args.model_id, args.model_revision)
    versions = package_versions(
        ["docker", "exec", "medtrace-vllm", "python"]
    )
    return {
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
            "command": command,
            "started_at": container_details["State"].get("StartedAt"),
        },
        "process": None,
        "packages": versions,
        "requirements": None,
        "pip_freeze": None,
        "docker": json.loads(run(["docker", "version", "--format", "{{json .}}"])),
    }


def native_runtime(args: argparse.Namespace) -> dict[str, Any]:
    pid_path = resolve_repo_path(args.pid_file)
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError) as exc:
        raise RuntimeError(f"invalid native vLLM PID file: {pid_path}") from exc
    command_text = run(["ps", "-ww", "-p", str(pid), "-o", "args="])
    if not command_text:
        raise RuntimeError(f"native vLLM process {pid} is not running")
    command = shlex.split(command_text)
    validate_server_command(command, args.model_id, args.model_revision)

    requirements_path = resolve_repo_path(args.requirements_file)
    versions = package_versions([sys.executable])
    return {
        "requested_image": None,
        "image": None,
        "container": None,
        "process": {"pid": pid, "command": command},
        "packages": versions,
        "requirements": {
            "file": str(requirements_path.relative_to(REPO_ROOT)),
            "sha256": sha256_file(requirements_path),
        },
        "pip_freeze": run([sys.executable, "-m", "pip", "freeze", "--all"]).splitlines(),
        "docker": None,
    }


def main() -> None:
    args = parse_args()
    git_status = run(["git", "status", "--porcelain"])
    if git_status and not args.allow_dirty:
        raise RuntimeError("refusing to capture an official run from a dirty Git worktree")

    runtime = docker_runtime(args) if args.backend == "docker" else native_runtime(args)
    if runtime["packages"]["vllm"] != args.vllm_version:
        raise RuntimeError(
            f"runtime reports vLLM {runtime['packages']['vllm']}, "
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
        "schema_version": 2,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_backend": args.backend,
        "git_commit": run(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(git_status),
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "requested_image": runtime["requested_image"],
        "image": runtime["image"],
        "container": runtime["container"],
        "process": runtime["process"],
        "packages": runtime["packages"],
        "requirements": runtime["requirements"],
        "pip_freeze": runtime["pip_freeze"],
        "gpus_csv": gpu_csv.splitlines(),
        "docker": runtime["docker"],
        "host": platform.platform(),
        "python_executable": sys.executable,
        "autodl_container_uuid": os.environ.get("AutoDLContainerUUID"),
    }
    output_path = resolve_repo_path(args.output)
    write_json(output_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Runtime manifest saved to: {output_path}")


if __name__ == "__main__":
    main()
