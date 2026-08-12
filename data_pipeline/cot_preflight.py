"""Zero-generation preflight for the real 40-question CoT pilot."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from typing import Any

from data_pipeline.cot_api import get_models, require_api_key


REPO_ROOT = Path(__file__).resolve().parents[1]


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def check_real_environment(
    config: dict[str, Any], *, remote_model_checks: bool = True
) -> dict[str, Any]:
    checks: dict[str, Any] = {"api_keys": {}, "models": {}, "gpu": {}}
    teacher = config["teacher"]
    validator = config["validator"]
    screener = config["screener"]

    teacher_key = require_api_key(str(teacher["api_key_env"]))
    validator_key = require_api_key(str(validator["api_key_env"]))
    local_key = require_api_key(str(screener["api_key_env"]), allow_empty=True)
    checks["api_keys"] = {
        teacher["api_key_env"]: True,
        validator["api_key_env"]: True,
        screener["api_key_env"]: bool(os.environ.get(str(screener["api_key_env"]))),
    }

    try:
        gpu_csv = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("nvidia-smi preflight failed") from exc
    first_gpu = gpu_csv.splitlines()[0]
    parts = [part.strip() for part in first_gpu.split(",")]
    total_mib = int(parts[-2])
    checks["gpu"] = {"name": parts[0], "memory_total_mib": total_mib}
    if total_mib < 20_000:
        raise RuntimeError(
            f"local screener requires the approved ~24 GB GPU; found {total_mib} MiB"
        )

    runtime_manifest_path = _repo_path(str(screener["runtime_manifest"]))
    if not runtime_manifest_path.is_file():
        raise RuntimeError(
            f"local screener runtime manifest is missing: {runtime_manifest_path}"
        )
    with runtime_manifest_path.open("r", encoding="utf-8") as handle:
        runtime_manifest = json.load(handle)
    if runtime_manifest.get("git_dirty") is not False:
        raise RuntimeError("screener runtime manifest was captured from a dirty worktree")
    try:
        current_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("unable to resolve the pilot Git commit") from exc
    if runtime_manifest.get("git_commit") != current_commit:
        raise RuntimeError(
            "screener runtime manifest Git commit differs from the pilot checkout"
        )
    expected_runtime = {
        "model_id": screener["model_id"],
        "model_revision": screener["model_revision"],
    }
    for field, expected in expected_runtime.items():
        if runtime_manifest.get(field) != expected:
            raise RuntimeError(
                f"screener runtime {field} mismatch: "
                f"{runtime_manifest.get(field)!r} != {expected!r}"
            )
    actual_vllm = (runtime_manifest.get("packages") or {}).get("vllm")
    if actual_vllm != str(screener["expected_vllm_version"]):
        raise RuntimeError(
            f"screener vLLM mismatch: {actual_vllm!r} != "
            f"{screener['expected_vllm_version']!r}"
        )
    checks["screener_runtime"] = {
        "manifest": runtime_manifest_path.relative_to(REPO_ROOT).as_posix(),
        "model_id": runtime_manifest["model_id"],
        "model_revision": runtime_manifest["model_revision"],
        "vllm": actual_vllm,
        "git_commit": current_commit,
    }

    if remote_model_checks:
        model_specs = (
            ("teacher", teacher, teacher_key),
            ("validator", validator, validator_key),
            ("screener", screener, local_key),
        )
        for role, spec, key in model_specs:
            available = get_models(
                str(spec["base_url"]), key, float(spec["timeout_seconds"])
            )
            expected = str(spec["model_id"])
            if expected not in available:
                raise RuntimeError(
                    f"{role} model is not available from configured endpoint: {expected}"
                )
            checks["models"][role] = expected
    checks["status"] = "passed"
    return checks


def redact_preflight(checks: dict[str, Any]) -> str:
    """Serialize only booleans and public model/runtime metadata."""

    return json.dumps(checks, ensure_ascii=False, indent=2, sort_keys=True)
