"""Preview, preflight, or execute the approved three-case HTTP 429 recovery."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import yaml

from data_pipeline.cot_api import (
    get_models,
    require_api_key,
    validate_validator_result,
    validator_response_format,
)
from data_pipeline.cot_budget import BudgetExceeded, BudgetLedger
from data_pipeline.cot_prompts import (
    VALIDATOR_SYSTEM_PROMPT,
    build_validator_strict_prompt,
)
from data_pipeline.prm_negative_canary_config import (
    validate_prm_negative_canary_config,
)
from data_pipeline.prm_negative_recovery_config import (
    validate_prm_negative_recovery_config,
)
from data_pipeline.prm_negative_recovery_state import (
    load_source_state,
    recovery_attempts_by_candidate,
    sha256_file,
    source_canary_hashes,
)
from data_pipeline.run_cot_pilot_real import (
    _append_jsonl,
    _call_with_budget,
    _load_jsonl,
    _write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/cot/prm_negative_validator_recovery_v1.yaml",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--preflight-only", action="store_true")
    group.add_argument("--execute-recovery-3", action="store_true")
    return parser.parse_args()


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


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


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("PRM negative recovery config must be an object")
    validate_prm_negative_recovery_config(config)
    source_config_path = _repo_path(str(config["source_canary_config"]))
    source_config = yaml.safe_load(source_config_path.read_text(encoding="utf-8"))
    if not isinstance(source_config, dict):
        raise ValueError("source PRM canary config must be an object")
    validate_prm_negative_canary_config(source_config)
    if source_config["source_run_dir"] != config["source_run_dir"]:
        raise ValueError("recovery and source run directories differ")
    if source_config["output_subdir"] != config["source_canary_subdir"]:
        raise ValueError("recovery and source canary directories differ")
    if source_config["validator"] != config["validator"]:
        raise ValueError("recovery validator differs from source canary")
    return config


def _records_by_hash(run_dir: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for event in _load_jsonl(run_dir / "teacher_events.jsonl"):
        record = event["record"]
        records.setdefault(str(record["content_sha256"]), record)
    return records


def _prepare_manifest(
    path: Path,
    *,
    config_sha256: str,
    source_hashes: dict[str, str],
    selected_ids: list[str],
) -> None:
    value = {
        "schema_version": "medtrace.prm-negative-validator-recovery-manifest.v1",
        "config_sha256": config_sha256,
        "source_canary_sha256": source_hashes,
        "selected_candidate_ids": selected_ids,
    }
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise RuntimeError("cannot resume: PRM recovery manifest changed")
    else:
        _write_json(path, value)


def _failure_categories(call: dict[str, Any]) -> set[str]:
    return {
        str(item.get("error_category"))
        for item in call.get("attempt_diagnostics") or []
        if item.get("error_category")
    }


def _sleep_before_request(seconds: int, reason: str) -> None:
    print(f"PRM recovery throttle: waiting {seconds}s ({reason})", flush=True)
    time.sleep(seconds)


def _validate_existing_attempts(
    grouped: dict[str, list[dict[str, Any]]],
    source_state: dict[str, Any],
    max_attempts: int,
) -> None:
    selected = set(source_state["selected_ids"])
    if any(candidate_id not in selected for candidate_id in grouped):
        raise RuntimeError("recovery log contains a candidate outside selection")
    for candidate_id, attempts in grouped.items():
        if len(attempts) > max_attempts:
            raise RuntimeError("recovery log exceeds the per-candidate attempt cap")
        numbers = [int(event["recovery_attempt"]) for event in attempts]
        if numbers != list(range(1, len(numbers) + 1)):
            raise RuntimeError("recovery attempt numbers are not contiguous")
        candidate = source_state["candidate_by_id"][candidate_id]
        complete_seen = False
        for event in attempts:
            if complete_seen:
                raise RuntimeError("recovery log continued after a complete result")
            if event.get("status") == "complete":
                validate_validator_result(
                    event.get("result"), len(candidate["trajectory"]["steps"])
                )
                complete_seen = True


def main() -> None:
    args = parse_args()
    config_path = _repo_path(args.config)
    config = _load_config(config_path)
    source_state = load_source_state(config, REPO_ROOT)
    preview = {
        "schema_version": "medtrace.prm-negative-validator-recovery-preview.v1",
        "contains_private_text_or_ids": False,
        "selected_http_429_events": len(source_state["selected_ids"]),
        "max_attempts_per_candidate": config["throttle"][
            "max_total_attempts_per_candidate"
        ],
        "maximum_new_requests": (
            len(source_state["selected_ids"])
            * int(config["throttle"]["max_total_attempts_per_candidate"])
        ),
        "api_hard_cap_cny_equivalent": config["budget"][
            "api_hard_cap_cny_equivalent"
        ],
        "budget_stop_limit_cny": (
            float(config["budget"]["api_hard_cap_cny_equivalent"])
            * float(config["budget"]["stop_before_limit_fraction"])
        ),
    }
    print(json.dumps(preview, indent=2, sort_keys=True), flush=True)
    if not args.preflight_only and not args.execute_recovery_3:
        print("Preview only; no API request was made.", flush=True)
        return

    git_state = _git_state()
    if git_state["git_dirty"]:
        raise RuntimeError("PRM recovery requires a clean Git worktree")
    api_key = require_api_key(str(config["validator"]["api_key_env"]))
    models = get_models(
        str(config["validator"]["base_url"]), api_key,
        float(config["validator"]["timeout_seconds"]),
    )
    if config["validator"]["model_id"] not in models:
        raise RuntimeError("frozen recovery validator model is unavailable")
    preflight = {
        "status": "passed",
        "git_commit": git_state["git_commit"],
        "selected_events": len(source_state["selected_ids"]),
        "validator_model": config["validator"]["model_id"],
    }
    print(json.dumps(preflight, indent=2, sort_keys=True), flush=True)
    if args.preflight_only:
        return

    runtime_config = dict(config)
    runtime_config["execution_enabled"] = True
    validate_prm_negative_recovery_config(runtime_config, execute=True)
    output_dir = REPO_ROOT / str(config["source_run_dir"]) / str(
        config["output_subdir"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    source_hashes_before = source_state["source_canary_hashes"]
    _prepare_manifest(
        output_dir / "private_manifest.json",
        config_sha256=sha256_file(config_path),
        source_hashes=source_hashes_before,
        selected_ids=source_state["selected_ids"],
    )
    attempt_path = output_dir / "recovery_attempts.jsonl"
    existing_attempts = _load_jsonl(attempt_path)
    grouped = recovery_attempts_by_candidate(existing_attempts)
    max_attempts = int(config["throttle"]["max_total_attempts_per_candidate"])
    _validate_existing_attempts(grouped, source_state, max_attempts)
    pricing = config["budget"]
    ledger = BudgetLedger(
        hard_cap_cny=float(pricing["api_hard_cap_cny_equivalent"]),
        stop_fraction=float(pricing["stop_before_limit_fraction"]),
        spent_cny=sum(
            float(event.get("usage", {}).get("cost_cny", 0))
            for event in existing_attempts
        ),
    )
    records = _records_by_hash(source_state["run_dir"])
    request_made_this_run = False
    previous_candidate_had_request = False
    budget_stopped = False
    for candidate_id in source_state["selected_ids"]:
        candidate = source_state["candidate_by_id"][candidate_id]
        attempts = grouped.setdefault(candidate_id, [])
        if any(event.get("status") == "complete" for event in attempts):
            previous_candidate_had_request = False
            continue
        while len(attempts) < max_attempts:
            if attempts:
                categories = {
                    str(value)
                    for value in attempts[-1].get("error_categories") or []
                }
                if "http_429" not in categories:
                    break
                delay = int(config["throttle"]["http_429_retry_delay_seconds"])
                reason = "HTTP 429 retry"
            elif not request_made_this_run:
                delay = int(config["throttle"]["initial_delay_seconds"])
                reason = "initial rate-limit cooldown"
            elif previous_candidate_had_request:
                delay = int(config["throttle"]["inter_candidate_delay_seconds"])
                reason = "inter-candidate spacing"
            else:
                delay = int(config["throttle"]["inter_candidate_delay_seconds"])
                reason = "resumed candidate spacing"
            _sleep_before_request(delay, reason)
            record = records[candidate["source"]["content_sha256"]]
            trajectory = candidate["trajectory"]
            steps = trajectory["steps"]
            prompt = build_validator_strict_prompt(
                record["question"], record["choices"], record["answer"],
                steps, trajectory["predicted_answer"],
            )
            try:
                call = _call_with_budget(
                    role="validator", config=config["validator"],
                    system_prompt=VALIDATOR_SYSTEM_PROMPT, user_prompt=prompt,
                    ledger=ledger, pricing=pricing, api_key=api_key,
                    response_format_json=True,
                    response_format=validator_response_format(len(steps)),
                    validate=lambda value, count=len(steps): (
                        validate_validator_result(value, count)
                    ),
                )
            except BudgetExceeded as exc:
                budget_stopped = True
                print(f"PRM recovery stopped by budget gate: {exc}", flush=True)
                break
            event = {
                "schema_version": "medtrace.prm-negative-validator-recovery-attempt.v1",
                "candidate_id": candidate_id,
                "recovery_attempt": len(attempts) + 1,
                "origin": candidate["origin"],
                "benchmark": candidate["source"]["dataset"],
                "source_status": source_state["source_event_by_id"][candidate_id][
                    "status"
                ],
                "status": call["status"],
                "result": call["parsed"],
                "request_id": call["request_id"],
                "finish_reason": call["finish_reason"],
                "usage": call["usage"],
                "error_categories": sorted(_failure_categories(call)),
                "attempt_diagnostics": call["attempt_diagnostics"],
                "response_content_sha256": (
                    hashlib.sha256(call["content"].encode("utf-8")).hexdigest()
                    if call["content"] else None
                ),
                "reasoning_content_sha256": call["reasoning_content_sha256"],
            }
            _append_jsonl(attempt_path, event)
            attempts.append(event)
            request_made_this_run = True
            previous_candidate_had_request = True
            print(
                f"PRM recovery candidate attempt={event['recovery_attempt']}/"
                f"{max_attempts} status={call['status']} "
                f"cost={ledger.spent_cny:.6f} CNY",
                flush=True,
            )
            if call["status"] == "complete":
                break
            if "http_429" not in event["error_categories"]:
                break
        if budget_stopped:
            break

    attempts_flat = [event for values in grouped.values() for event in values]
    terminal = 0
    recovered = 0
    for candidate_id in source_state["selected_ids"]:
        values = grouped.get(candidate_id, [])
        if any(event.get("status") == "complete" for event in values):
            terminal += 1
            recovered += 1
        elif len(values) >= max_attempts or (
            values and "http_429" not in set(values[-1].get("error_categories") or [])
        ):
            terminal += 1
    if source_canary_hashes(source_state["canary_dir"]) != source_hashes_before:
        raise RuntimeError("source PRM canary changed during validator recovery")
    metadata = {
        "schema_version": "medtrace.prm-negative-validator-recovery-run.v1",
        "status": (
            "complete" if terminal == len(source_state["selected_ids"]) else "partial"
        ),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "contains_private_text_or_ids": False,
        "source_canary_unchanged": True,
        "runtime": preflight,
        "selected_events": len(source_state["selected_ids"]),
        "request_attempts": len(attempts_flat),
        "terminal_events": terminal,
        "recovered_complete": recovered,
        "spent_cny_equivalent": round(ledger.spent_cny, 8),
        "budget_stop_limit_cny": ledger.stop_limit_cny,
        "budget_stopped": budget_stopped,
    }
    _write_json(output_dir / "metadata.json", metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    if budget_stopped:
        raise RuntimeError("PRM recovery stopped at the approved budget line")


if __name__ == "__main__":
    main()
