"""Preview or execute an isolated six-event validator v2 recovery canary."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import yaml

from data_pipeline.cot_api import (
    get_models,
    require_api_key,
    validate_validator_result,
    validator_response_format,
)
from data_pipeline.cot_budget import BudgetLedger
from data_pipeline.cot_diagnostics import error_detail
from data_pipeline.cot_prompts import (
    VALIDATOR_SYSTEM_PROMPT,
    build_validator_recovery_prompt,
)
from data_pipeline.cot_recovery_config import validate_recovery_config
from data_pipeline.run_cot_pilot_real import (
    _append_jsonl,
    _call_with_budget,
    _load_completed,
    _load_jsonl,
    _write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "metadata.json",
    "teacher_events.jsonl",
    "screener_events.jsonl",
    "validator_events.jsonl",
    "canonical_trajectories.jsonl",
    "sft_verified.jsonl",
    "process_train.jsonl",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/cot/validator_recovery_v2.yaml"
    )
    parser.add_argument(
        "--execute-canary-6",
        action="store_true",
        help="authorize exactly the frozen six-event paid recovery canary",
    )
    return parser.parse_args()


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hashes(run_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in SOURCE_FILES:
        path = run_dir / name
        if not path.is_file():
            raise RuntimeError(f"recovery source file is missing: {name}")
        hashes[name] = _sha256(path)
    return hashes


def _event_key(event: dict[str, Any]) -> tuple[str, int]:
    return str(event["record_id"]), int(event["candidate_index"])


def _failure_details(event: dict[str, Any]) -> list[str]:
    return [error_detail(str(error)) for error in event.get("errors") or []]


def select_canary(
    teachers: dict[tuple[Any, ...], dict[str, Any]],
    validators: dict[tuple[Any, ...], dict[str, Any]],
    quotas: dict[str, int],
) -> list[tuple[str, int]]:
    """Select deterministically across first-error details within each benchmark."""

    by_benchmark_detail: dict[str, dict[str, deque[tuple[str, int]]]] = defaultdict(
        lambda: defaultdict(deque)
    )
    for raw_key, event in sorted(validators.items()):
        key = (str(raw_key[0]), int(raw_key[1]))
        if event.get("status") == "complete":
            continue
        teacher = teachers.get(raw_key)
        if teacher is None:
            raise RuntimeError("failed validator event has no matching teacher event")
        benchmark = str((teacher.get("record") or {}).get("benchmark") or "unknown")
        details = _failure_details(event)
        first_detail = details[0] if details else "unknown"
        by_benchmark_detail[benchmark][first_detail].append(key)

    selected: list[tuple[str, int]] = []
    for benchmark, quota in quotas.items():
        buckets = by_benchmark_detail.get(benchmark) or {}
        detail_order = sorted(buckets)
        if sum(len(bucket) for bucket in buckets.values()) < quota:
            raise RuntimeError(f"not enough failed {benchmark} events for canary")
        while quota:
            made_progress = False
            for detail in detail_order:
                bucket = buckets[detail]
                if bucket and quota:
                    selected.append(bucket.popleft())
                    quota -= 1
                    made_progress = True
            if not made_progress:
                raise RuntimeError(f"unable to fill {benchmark} canary quota")
    return selected


def _preview(
    selected: list[tuple[str, int]],
    teachers: dict[tuple[Any, ...], dict[str, Any]],
    validators: dict[tuple[Any, ...], dict[str, Any]],
) -> dict[str, Any]:
    by_benchmark: Counter[str] = Counter()
    by_first_error: Counter[str] = Counter()
    for key in selected:
        by_benchmark[str(teachers[key]["record"]["benchmark"])] += 1
        details = _failure_details(validators[key])
        by_first_error[details[0] if details else "unknown"] += 1
    return {
        "schema_version": "medtrace.validator-recovery-preview.v2",
        "contains_private_text_or_ids": False,
        "selected_events": len(selected),
        "by_benchmark": dict(sorted(by_benchmark.items())),
        "by_first_error_detail": dict(sorted(by_first_error.items())),
    }


def _git_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip())
    return {"git_commit": commit, "git_dirty": dirty}


def _prepare_private_manifest(
    path: Path,
    *,
    config_sha256: str,
    source_hashes: dict[str, str],
    selected: list[tuple[str, int]],
) -> None:
    expected = {
        "schema_version": "medtrace.validator-recovery-private-manifest.v2",
        "config_sha256": config_sha256,
        "source_artifact_sha256": source_hashes,
        "selected_keys": [list(key) for key in selected],
    }
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            actual = json.load(handle)
        if actual != expected:
            raise RuntimeError("cannot resume: private recovery manifest changed")
    else:
        _write_json(path, expected)


def main() -> None:
    args = parse_args()
    config_path = _repo_path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    validate_recovery_config(config)

    run_dir = _repo_path(str(config["source_run_dir"]))
    source_hashes_before = _source_hashes(run_dir)
    with (run_dir / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    identity = config["source_identity"]
    actual_identity = {
        "config_sha256": metadata.get("config_sha256"),
        "questions_sha256": metadata.get("questions_sha256"),
        "generation_git_commit": (
            (metadata.get("preflight") or {}).get("screener_runtime") or {}
        ).get("git_commit"),
    }
    for field, expected in identity.items():
        if field == "expected_failed_validator_events":
            continue
        if actual_identity.get(field) != expected:
            raise RuntimeError(f"recovery source identity mismatch: {field}")

    teachers = _load_completed(
        run_dir / "teacher_events.jsonl", ("record_id", "candidate_index")
    )
    screeners = _load_completed(
        run_dir / "screener_events.jsonl", ("record_id", "candidate_index")
    )
    validators = _load_completed(
        run_dir / "validator_events.jsonl", ("record_id", "candidate_index")
    )
    failed = sum(event.get("status") != "complete" for event in validators.values())
    if failed != int(identity["expected_failed_validator_events"]):
        raise RuntimeError("recovery source failed-validator count changed")
    quotas = {key: int(value) for key, value in config["canary"]["by_benchmark"].items()}
    selected = select_canary(teachers, validators, quotas)
    preview = _preview(selected, teachers, validators)
    print(json.dumps(preview, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    if not args.execute_canary_6:
        print("Preview only; no API request was made.", flush=True)
        return

    runtime_config = dict(config)
    runtime_config["execution_enabled"] = True
    validate_recovery_config(runtime_config, execute=True)
    git_state = _git_state()
    if git_state["git_dirty"]:
        raise RuntimeError("recovery canary requires a clean Git worktree")
    api_key = require_api_key(config["validator"]["api_key_env"])
    available = get_models(
        str(config["validator"]["base_url"]), api_key,
        float(config["validator"]["timeout_seconds"]),
    )
    if config["validator"]["model_id"] not in available:
        raise RuntimeError("frozen validator model is unavailable")

    output_dir = run_dir / str(config["output_subdir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    _prepare_private_manifest(
        output_dir / "private_manifest.json",
        config_sha256=_sha256(config_path),
        source_hashes=source_hashes_before,
        selected=selected,
    )
    event_path = output_dir / "canary_events.jsonl"
    completed = _load_completed(event_path, ("record_id", "candidate_index"))
    if any(key not in selected for key in completed):
        raise RuntimeError("recovery log contains a key outside the frozen canary")
    pricing = config["budget"]
    ledger = BudgetLedger(
        hard_cap_cny=float(pricing["api_hard_cap_cny_equivalent"]),
        stop_fraction=float(pricing["stop_before_limit_fraction"]),
        spent_cny=sum(
            float(event.get("usage", {}).get("cost_cny", 0))
            for event in completed.values()
        ),
    )
    for key in selected:
        if key in completed:
            continue
        teacher = teachers[key]
        if key not in screeners:
            raise RuntimeError("canary event has no matching screener event")
        record = teacher["record"]
        rule = teacher["rule_check"]
        old_validator = validators[key]
        details = _failure_details(old_validator)
        prompt = build_validator_recovery_prompt(
            record["question"], record["choices"], record["answer"],
            rule["steps"], rule["predicted_answer"], details,
        )
        call = _call_with_budget(
            role="validator",
            config=config["validator"],
            system_prompt=VALIDATOR_SYSTEM_PROMPT,
            user_prompt=prompt,
            ledger=ledger,
            pricing=pricing,
            api_key=api_key,
            response_format_json=True,
            response_format=validator_response_format(len(rule["steps"])),
            validate=lambda value, count=len(rule["steps"]): validate_validator_result(
                value, count
            ),
        )
        event = {
            "schema_version": "medtrace.validator-recovery-event.v2",
            "record_id": key[0],
            "candidate_index": key[1],
            "source_validator_status": old_validator["status"],
            "source_error_details": details,
            "status": call["status"],
            "result": call["parsed"],
            "request_id": call["request_id"],
            "finish_reason": call["finish_reason"],
            "usage": call["usage"],
            "attempt_diagnostics": call["attempt_diagnostics"],
            "response_content_sha256": (
                hashlib.sha256(call["content"].encode("utf-8")).hexdigest()
                if call["content"] else None
            ),
            "reasoning_content_sha256": call["reasoning_content_sha256"],
        }
        _append_jsonl(event_path, event)
        completed[key] = event
        print(
            f"recovery canary {len(completed)}/{len(selected)} "
            f"status={call['status']}", flush=True,
        )

    source_hashes_after = _source_hashes(run_dir)
    if source_hashes_after != source_hashes_before:
        raise RuntimeError("an original pilot artifact changed during recovery")
    status_counts = Counter(event["status"] for event in completed.values())
    recovery_metadata = {
        "schema_version": "medtrace.validator-recovery-canary.v2",
        "status": "complete" if len(completed) == len(selected) else "partial",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "contains_private_text_or_ids": False,
        "source_identity": actual_identity,
        "source_artifacts_unchanged": True,
        "runtime": git_state,
        "selected_events": len(selected),
        "completed_events": len(completed),
        "status_counts": dict(sorted(status_counts.items())),
        "spent_cny_equivalent": round(ledger.spent_cny, 8),
    }
    _write_json(output_dir / "canary_metadata.json", recovery_metadata)
    print(json.dumps(recovery_metadata, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
