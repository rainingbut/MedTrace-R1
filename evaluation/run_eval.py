"""Run a reproducible multiple-choice evaluation via an OpenAI-compatible API."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any
from urllib import request as urllib_request

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised by users without eval extras
    raise SystemExit(
        "PyYAML is required. Install it with: pip install -r requirements/eval.txt"
    ) from exc

try:
    from evaluation.answer_extractor import extract_answer
    from evaluation.metrics import score_records
except ModuleNotFoundError:  # Support: python evaluation/run_eval.py
    from answer_extractor import extract_answer
    from metrics import score_records


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Evaluation YAML file")
    parser.add_argument("--input-file", help="Override config input_file")
    parser.add_argument("--output-root", help="Override config output_root")
    parser.add_argument("--run-dir", help="Resume an existing run directory")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N records")
    parser.add_argument(
        "--limit-per-benchmark",
        type=int,
        help="Evaluate the first N records from each benchmark",
    )
    parser.add_argument(
        "--runtime-manifest",
        help="Override config runtime_manifest with a captured GPU/container manifest",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use mock_output or a fixed dummy answer; do not call a model",
    )
    return parser.parse_args()


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("evaluation config must be a YAML mapping")
    return config


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            required = {"id", "benchmark", "split", "question", "choices", "answer"}
            missing = required - set(record)
            if missing:
                raise ValueError(f"{path}:{line_number} missing fields {sorted(missing)}")
            if record["answer"] not in record["choices"]:
                raise ValueError(f"{path}:{line_number} answer is not a valid choice")
            records.append(record)
    if not records:
        raise ValueError(f"no records found in {path}")
    return records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _format_prompt(record: dict[str, Any], prompt_style: str) -> str:
    choices = "\n".join(
        f"{label}. {text}" for label, text in record["choices"].items()
    )
    if prompt_style == "reasoning":
        instruction = (
            "Analyze the medical multiple-choice question carefully. Explain your reasoning, "
            "then end with exactly one line in the form `Final Answer: X`, where X is an "
            "available option letter. Do not write anything after that line."
        )
    elif prompt_style == "direct":
        instruction = (
            "Answer the medical multiple-choice question. End with exactly one line in the "
            "form `Final Answer: X`, where X is an available option letter."
        )
    else:
        raise ValueError(f"unsupported prompt_style: {prompt_style}")
    return f"{instruction}\n\nQuestion:\n{record['question']}\n\nOptions:\n{choices}"


def _post_chat_completion(
    config: dict[str, Any], prompt: str, request_seed: int
) -> tuple[str, int | None, str | None]:
    api_key_env = str(config.get("api_key_env", "MEDTRACE_API_KEY"))
    api_key = os.environ.get(api_key_env, "EMPTY")
    base_url = str(config["base_url"]).rstrip("/")
    url = f"{base_url}/chat/completions"
    payload = {
        "model": config["model"],
        "messages": [
            {"role": "system", "content": config["system_prompt"]},
            {"role": "user", "content": prompt},
        ],
        "temperature": config["temperature"],
        "top_p": config["top_p"],
        "max_tokens": config["max_new_tokens"],
        "seed": request_seed,
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib_request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib_request.urlopen(request, timeout=float(config["timeout_seconds"])) as response:
        response_body = json.loads(response.read().decode("utf-8"))
    choice = response_body["choices"][0]
    content = choice["message"]["content"]
    usage = response_body.get("usage") or {}
    return content, usage.get("completion_tokens"), choice.get("finish_reason")


def _get_server_info(config: dict[str, Any]) -> dict[str, Any]:
    api_key_env = str(config.get("api_key_env", "MEDTRACE_API_KEY"))
    api_key = os.environ.get(api_key_env, "EMPTY")
    base_url = str(config["base_url"]).rstrip("/")
    server_root = base_url[:-3] if base_url.endswith("/v1") else base_url
    headers = {"Authorization": f"Bearer {api_key}"}

    def get_json(url: str) -> dict[str, Any]:
        request = urllib_request.Request(url, headers=headers)
        with urllib_request.urlopen(
            request, timeout=float(config["timeout_seconds"])
        ) as response:
            return json.loads(response.read().decode("utf-8"))

    return {
        "models": get_json(f"{base_url}/models"),
        "version": get_json(f"{server_root}/version"),
    }


def _evaluate_one(
    record: dict[str, Any], config: dict[str, Any], dry_run: bool
) -> dict[str, Any]:
    prompt = _format_prompt(record, str(config["prompt_style"]))
    started = time.perf_counter()
    raw_response = ""
    completion_tokens: int | None = None
    finish_reason: str | None = "mock" if dry_run else None
    failure: str | None = None

    if dry_run:
        raw_response = str(record.get("mock_output", "Final Answer: A"))
    else:
        attempts = int(config["max_retries"]) + 1
        for attempt in range(attempts):
            try:
                raw_response, completion_tokens, finish_reason = _post_chat_completion(
                    config, prompt, int(config["seed"])
                )
                failure = None
                break
            except (KeyError, ValueError, OSError, json.JSONDecodeError) as exc:
                failure = f"{type(exc).__name__}: {exc}"
                if attempt + 1 < attempts:
                    time.sleep(min(2**attempt, 8))

    extraction = extract_answer(raw_response, record["choices"].keys())
    return {
        "id": record["id"],
        "benchmark": record["benchmark"],
        "split": record["split"],
        "answer": record["answer"],
        "prompt": prompt,
        "raw_response": raw_response,
        "extracted_answer": extraction.answer,
        "parse_status": extraction.parse_status,
        "format_valid": extraction.format_valid,
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
        "latency_seconds": round(time.perf_counter() - started, 6),
        "error": failure,
    }


def _load_existing_predictions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return _load_prediction_jsonl(path)


def _load_prediction_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
    return records


def _validate_config(config: dict[str, Any], dry_run: bool) -> None:
    required = {
        "model",
        "model_revision",
        "expected_vllm_version",
        "expected_vllm_image",
        "base_url",
        "input_file",
        "output_root",
        "prompt_style",
        "system_prompt",
        "temperature",
        "top_p",
        "max_new_tokens",
        "seed",
        "concurrency",
        "timeout_seconds",
        "max_retries",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"evaluation config is missing fields: {sorted(missing)}")
    if float(config["temperature"]) != 0.0:
        raise ValueError("baseline evaluation requires temperature: 0")
    if int(config["concurrency"]) < 1:
        raise ValueError("concurrency must be positive")
    if not dry_run and str(config["model_revision"]).startswith("REPLACE_WITH_"):
        raise ValueError("pin model_revision to an exact commit SHA before a real run")


def _select_records(
    records: list[dict[str, Any]],
    limit: int | None,
    limit_per_benchmark: int | None,
) -> list[dict[str, Any]]:
    if limit is not None and limit_per_benchmark is not None:
        raise ValueError("use only one of --limit and --limit-per-benchmark")
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        return records[:limit]
    if limit_per_benchmark is None:
        return records
    if limit_per_benchmark < 1:
        raise ValueError("--limit-per-benchmark must be positive")

    counts: dict[str, int] = {}
    selected: list[dict[str, Any]] = []
    for record in records:
        benchmark = str(record["benchmark"])
        count = counts.get(benchmark, 0)
        if count < limit_per_benchmark:
            selected.append(record)
            counts[benchmark] = count + 1
    return selected


def main() -> None:
    args = parse_args()
    config_path = _resolve_repo_path(args.config).resolve()
    config = _load_yaml(config_path)
    if args.input_file:
        config["input_file"] = args.input_file
    if args.output_root:
        config["output_root"] = args.output_root
    if args.runtime_manifest:
        config["runtime_manifest"] = args.runtime_manifest
    _validate_config(config, args.dry_run)

    input_path = _resolve_repo_path(str(config["input_file"])).resolve()
    records = _load_jsonl(input_path)
    records = _select_records(records, args.limit, args.limit_per_benchmark)

    runtime_manifest: dict[str, Any] | None = None
    runtime_manifest_path: Path | None = None
    if config.get("runtime_manifest"):
        runtime_manifest_path = _resolve_repo_path(
            str(config["runtime_manifest"])
        ).resolve()
        if runtime_manifest_path.exists():
            with runtime_manifest_path.open("r", encoding="utf-8") as handle:
                runtime_manifest = json.load(handle)
            if not args.dry_run:
                expected_runtime = {
                    "model_id": config["model"],
                    "model_revision": config["model_revision"],
                    "requested_image": config["expected_vllm_image"],
                }
                for field, expected in expected_runtime.items():
                    if runtime_manifest.get(field) != expected:
                        raise ValueError(
                            f"runtime manifest {field} does not match config: "
                            f"{runtime_manifest.get(field)!r} != {expected!r}"
                        )
                actual_vllm = (runtime_manifest.get("packages") or {}).get("vllm")
                if actual_vllm != str(config["expected_vllm_version"]):
                    raise ValueError(
                        "runtime manifest vLLM version does not match config: "
                        f"{actual_vllm!r} != {config['expected_vllm_version']!r}"
                    )
        elif not args.dry_run:
            raise ValueError(f"runtime manifest does not exist: {runtime_manifest_path}")
    elif not args.dry_run:
        raise ValueError("a runtime_manifest is required for a real baseline run")

    if args.run_dir:
        run_dir = _resolve_repo_path(args.run_dir).resolve()
        if not run_dir.is_dir():
            raise ValueError(f"resume directory does not exist: {run_dir}")
    else:
        output_root = _resolve_repo_path(str(config["output_root"]))
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe_model = str(config["model"]).replace("/", "_").replace("\\", "_")
        suffix = "dryrun" if args.dry_run else str(config["prompt_style"])
        run_dir = output_root / f"{timestamp}_{safe_model}_{suffix}_seed{config['seed']}"
        run_dir.mkdir(parents=True, exist_ok=False)

    predictions_path = run_dir / "predictions.jsonl"
    existing = _load_existing_predictions(predictions_path)
    existing_ids = {str(record["id"]) for record in existing}
    if len(existing_ids) != len(existing):
        raise ValueError("resume file contains duplicate prediction ids")

    requested_ids = {str(record["id"]) for record in records}
    unexpected_ids = existing_ids - requested_ids
    if unexpected_ids:
        raise ValueError(
            "resume file contains ids outside the requested input: "
            f"{sorted(unexpected_ids)[:3]}"
        )

    config_sha256 = _sha256(config_path)
    input_sha256 = _sha256(input_path)
    runtime_manifest_sha256 = (
        _sha256(runtime_manifest_path)
        if runtime_manifest_path is not None and runtime_manifest_path.exists()
        else None
    )
    metadata_path = run_dir / "metadata.json"
    previous_metadata: dict[str, Any] = {}
    if args.run_dir and metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            previous_metadata = json.load(handle)
        if previous_metadata.get("config_sha256") != config_sha256:
            raise ValueError("cannot resume: evaluation config hash has changed")
        if previous_metadata.get("input_sha256") != input_sha256:
            raise ValueError("cannot resume: input data hash has changed")
        if bool(previous_metadata.get("dry_run")) != args.dry_run:
            raise ValueError("cannot resume: dry-run mode has changed")
        if previous_metadata.get("runtime_manifest_sha256") != runtime_manifest_sha256:
            raise ValueError("cannot resume: runtime manifest hash has changed")

    pending = [record for record in records if str(record["id"]) not in existing_ids]
    git_status = _git_value("status", "--porcelain")
    server_info = None if args.dry_run else _get_server_info(config)
    run_started = time.perf_counter()
    original_started_at = previous_metadata.get(
        "started_at_utc", datetime.now(timezone.utc).isoformat()
    )
    previous_elapsed_seconds = float(previous_metadata.get("elapsed_seconds", 0.0))
    metadata = {
        "schema_version": 1,
        "status": "running",
        "started_at_utc": original_started_at,
        "dry_run": args.dry_run,
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "resolved_config": config,
        "input_file": str(input_path),
        "input_sha256": input_sha256,
        "runtime_manifest_file": (
            str(runtime_manifest_path) if runtime_manifest_path is not None else None
        ),
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "runtime_manifest": runtime_manifest,
        "server_info": server_info,
        "requested_records": len(records),
        "already_completed": len(existing),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_dirty": bool(git_status),
        "python": sys.version,
        "platform": platform.platform(),
    }
    _write_json(metadata_path, metadata)

    completed = list(existing)
    with predictions_path.open("a", encoding="utf-8") as prediction_file:
        with ThreadPoolExecutor(max_workers=int(config["concurrency"])) as executor:
            futures = {
                executor.submit(_evaluate_one, record, config, args.dry_run)
                for record in pending
            }
            for future in as_completed(futures):
                prediction = future.result()
                prediction_file.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                prediction_file.flush()
                completed.append(prediction)
                print(
                    f"[{len(completed)}/{len(records)}] {prediction['id']} "
                    f"answer={prediction['extracted_answer']} error={bool(prediction['error'])}"
                )

    metrics = score_records(completed)
    _write_json(run_dir / "metrics.json", metrics)
    metadata.update(
        {
            "status": "complete",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "completed_records": len(completed),
            "elapsed_seconds": round(
                previous_elapsed_seconds + time.perf_counter() - run_started, 6
            ),
        }
    )
    _write_json(metadata_path, metadata)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
