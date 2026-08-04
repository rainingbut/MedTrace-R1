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


def main() -> None:
    args = parse_args()
    config_path = _resolve_repo_path(args.config).resolve()
    config = _load_yaml(config_path)
    if args.input_file:
        config["input_file"] = args.input_file
    if args.output_root:
        config["output_root"] = args.output_root
    _validate_config(config, args.dry_run)

    input_path = _resolve_repo_path(str(config["input_file"])).resolve()
    records = _load_jsonl(input_path)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        records = records[: args.limit]

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
    metadata_path = run_dir / "metadata.json"
    if args.run_dir and metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            previous_metadata = json.load(handle)
        if previous_metadata.get("config_sha256") != config_sha256:
            raise ValueError("cannot resume: evaluation config hash has changed")
        if previous_metadata.get("input_sha256") != input_sha256:
            raise ValueError("cannot resume: input data hash has changed")
        if bool(previous_metadata.get("dry_run")) != args.dry_run:
            raise ValueError("cannot resume: dry-run mode has changed")

    pending = [record for record in records if str(record["id"]) not in existing_ids]
    git_status = _git_value("status", "--porcelain")
    metadata = {
        "schema_version": 1,
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": args.dry_run,
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "resolved_config": config,
        "input_file": str(input_path),
        "input_sha256": input_sha256,
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
        }
    )
    _write_json(metadata_path, metadata)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
