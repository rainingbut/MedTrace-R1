"""Run the full paid 40-question CoT pilot with resumable phase logs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable, Iterable

import yaml

from data_pipeline.cot_api import (
    ChatResult,
    parse_json_object,
    post_chat_completion,
    require_api_key,
    validate_screener_result,
    validate_validator_result,
)
from data_pipeline.cot_budget import BudgetLedger, request_cost_cny
from data_pipeline.cot_config import validate_pilot_config
from data_pipeline.cot_preflight import check_real_environment, redact_preflight
from data_pipeline.cot_prompts import (
    SCREENER_SYSTEM_PROMPT,
    TEACHER_SYSTEM_PROMPT,
    VALIDATOR_SYSTEM_PROMPT,
    build_screener_prompt,
    build_teacher_prompt,
    build_validator_prompt,
)
from data_pipeline.cot_rules import check_teacher_response


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cot/pilot_v1.yaml")
    parser.add_argument("--pilot-manifest", default=None)
    parser.add_argument("--run-dir", default="results/cot/pilot_v1_real")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--execute-40",
        action="store_true",
        help="explicitly authorize the frozen 40-question paid pilot",
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


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
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


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _estimated_input_tokens(system_prompt: str, user_prompt: str) -> int:
    return max(1, (len(system_prompt) + len(user_prompt) + 3) // 4)


def _cost_for_result(
    role: str,
    result: ChatResult,
    estimated_input: int,
    estimated_output: int,
    pricing: dict[str, Any],
) -> tuple[int, int, float, bool]:
    input_tokens = result.input_tokens or estimated_input
    output_tokens = result.output_tokens or estimated_output
    estimated = result.input_tokens is None or result.output_tokens is None
    return (
        input_tokens,
        output_tokens,
        request_cost_cny(role, input_tokens, output_tokens, pricing),
        estimated,
    )


def _call_with_budget(
    *,
    role: str,
    config: dict[str, Any],
    system_prompt: str,
    user_prompt: str,
    ledger: BudgetLedger,
    pricing: dict[str, Any],
    api_key: str,
    response_format_json: bool,
    validate: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    estimated_input = _estimated_input_tokens(system_prompt, user_prompt)
    estimated_output = int(config["max_output_tokens"])
    estimated_cost = request_cost_cny(
        role, estimated_input, estimated_output, pricing
    )
    attempts = int(config.get("max_retries", 0)) + 1
    errors: list[str] = []
    charged_for_errors = 0.0
    for attempt in range(attempts):
        ledger.assert_can_spend(estimated_cost)
        attempt_cost = estimated_cost
        try:
            extras: dict[str, Any] = {}
            if role == "teacher":
                extras["enable_thinking"] = bool(config["enable_thinking"])
            elif role == "validator":
                extras["thinking"] = {"type": "enabled"}
                extras["reasoning_effort"] = "high"
            result = post_chat_completion(
                base_url=str(config["base_url"]),
                api_key=api_key,
                model=str(config["model_id"]),
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=float(config["temperature"]),
                top_p=float(config["top_p"]) if "top_p" in config else None,
                max_tokens=int(config["max_output_tokens"]),
                timeout_seconds=float(config["timeout_seconds"]),
                response_format_json=response_format_json,
                extra_body=extras,
            )
            input_tokens, output_tokens, cost, usage_estimated = _cost_for_result(
                role,
                result,
                estimated_input,
                max(1, len(result.content) // 4),
                pricing,
            )
            attempt_cost = cost
            parsed = parse_json_object(result.content) if response_format_json else None
            if parsed is not None and validate is not None:
                parsed = validate(parsed)
            ledger.record(cost)
            return {
                "status": "complete",
                "content": result.content,
                "reasoning_content_sha256": (
                    hashlib.sha256(result.reasoning_content.encode("utf-8")).hexdigest()
                    if result.reasoning_content
                    else None
                ),
                "request_id": result.request_id,
                "finish_reason": result.finish_reason,
                "parsed": parsed,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost_cny": round(cost + charged_for_errors, 8),
                    "estimated": usage_estimated or bool(charged_for_errors),
                    "failed_attempt_reserve_cny": round(charged_for_errors, 8),
                },
                "attempts": attempt + 1,
                "errors": errors,
            }
        except Exception as exc:  # every failed attempt is preserved and budgeted
            charged_for_errors += attempt_cost
            ledger.record(attempt_cost)
            errors.append(f"{type(exc).__name__}: {exc}")
            if attempt + 1 < attempts:
                time.sleep(min(2**attempt, 8))
    return {
        "status": "api_or_parse_error",
        "content": "",
        "reasoning_content_sha256": None,
        "request_id": None,
        "finish_reason": None,
        "parsed": None,
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_cny": round(charged_for_errors, 8),
            "estimated": True,
            "failed_attempt_reserve_cny": round(charged_for_errors, 8),
        },
        "attempts": attempts,
        "errors": errors,
    }


def _load_completed(path: Path, key_fields: tuple[str, ...]) -> dict[tuple[Any, ...], dict[str, Any]]:
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    for event in _load_jsonl(path):
        key = tuple(event[field] for field in key_fields)
        if key in result:
            raise ValueError(f"duplicate resume key in {path}: {key}")
        result[key] = event
    return result


def _canonical_from_events(
    teacher_event: dict[str, Any],
    screener_event: dict[str, Any],
    validator_event: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any] | None:
    if teacher_event["status"] != "complete" or not teacher_event["rule_check"]["passed"]:
        return None
    if validator_event["status"] != "complete":
        return None
    record = teacher_event["record"]
    rule = teacher_event["rule_check"]
    screening_result = screener_event["result"]
    validation_result = validator_event["result"]
    raw_digest = hashlib.sha256(teacher_event["raw_response"].encode("utf-8")).hexdigest()
    trajectory_digest = hashlib.sha256(
        f"{record['id']}:{teacher_event['candidate_index']}:{raw_digest}".encode("utf-8")
    ).hexdigest()
    screening = {
        "status": screener_event["status"],
        "model_id": config["screener"]["model_id"],
        "model_revision": config["screener"]["model_revision"],
        "prompt_version": config["screener"]["prompt_version"],
        "verdict": screening_result["verdict"],
        "suspected_first_error_step": screening_result["suspected_first_error_step"],
        "error_codes": screening_result["error_codes"],
    }
    verification = {
        "status": "complete",
        "provider": config["validator"]["provider"],
        "model_id": config["validator"]["model_id"],
        "provider_version": config["validator"]["provider_version"],
        "prompt_version": config["validator"]["prompt_version"],
        **validation_result,
        "request_id": validator_event["request_id"],
        "usage": validator_event["usage"],
    }
    return {
        "schema_version": "medtrace.cot.v1",
        "trajectory_id": f"sha256:{trajectory_digest}",
        "source": {
            "dataset": record["benchmark"],
            "config": "default",
            "split": record["split"],
            "source_revision": record["source_revision"],
            "source_file_sha256": record["source_file_sha256"],
            "source_id": record["source_id"],
            "source_index": record["source_index"],
            "content_sha256": record["content_sha256"],
        },
        "problem": {
            "question": record["question"],
            "choices": record["choices"],
            "gold_answer": record["answer"],
        },
        "generation": {
            "run_mode": "real",
            "provider": config["teacher"]["provider"],
            "model_id": config["teacher"]["model_id"],
            "provider_version": config["teacher"]["model_id"],
            "model_revision": None,
            "prompt_version": config["teacher"]["prompt_version"],
            "candidate_index": teacher_event["candidate_index"],
            "decoding": {
                key: config["teacher"][key]
                for key in ("temperature", "top_p", "max_output_tokens")
            },
            "request_id": teacher_event["request_id"],
            "usage": teacher_event["usage"],
        },
        "trajectory": {
            "steps": [
                {"index": index, "text": text}
                for index, text in enumerate(rule["steps"])
            ],
            "predicted_answer": rule["predicted_answer"],
            "raw_response_sha256": raw_digest,
        },
        "rule_check": {"passed": True, "failure_codes": []},
        "screening": screening,
        "verification": verification,
        "disposition": (
            "sft_accept" if validation_result["trajectory_label"] == 1 else "prm_only"
        ),
    }


def main() -> None:
    args = parse_args()
    config_path = _repo_path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    runtime_config = dict(config)
    runtime_config["execution_enabled"] = True
    validate_pilot_config(runtime_config, real_run=True)

    pilot_manifest_path = _repo_path(
        args.pilot_manifest or config["outputs"]["pilot_input_manifest"]
    )
    with pilot_manifest_path.open("r", encoding="utf-8") as handle:
        pilot_manifest = json.load(handle)
    if pilot_manifest.get("config_sha256") != _sha256(config_path):
        raise ValueError(
            "pilot manifest was selected with a different config; "
            "rerun data_pipeline.select_cot_pilot"
        )
    questions_path = _repo_path(pilot_manifest["questions_file"])
    if _sha256(questions_path) != pilot_manifest["questions_sha256"]:
        raise ValueError("pilot questions hash differs from manifest")
    records = _load_jsonl(questions_path)
    if len(records) != 40 or any(record["split"] != "train" for record in records):
        raise ValueError("real pilot requires exactly 40 train questions")

    preflight = check_real_environment(config)
    print(redact_preflight(preflight), flush=True)
    if args.preflight_only:
        return
    if not args.execute_40:
        raise RuntimeError("paid run requires the explicit --execute-40 flag")

    run_dir = _repo_path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_sha256 = _sha256(config_path)
    questions_sha256 = _sha256(questions_path)
    metadata_path = run_dir / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            previous = json.load(handle)
        if previous.get("config_sha256") != config_sha256:
            raise ValueError("cannot resume: config hash changed")
        if previous.get("questions_sha256") != questions_sha256:
            raise ValueError("cannot resume: questions hash changed")

    teacher_path = run_dir / "teacher_events.jsonl"
    screener_path = run_dir / "screener_events.jsonl"
    validator_path = run_dir / "validator_events.jsonl"
    teachers = _load_completed(teacher_path, ("record_id", "candidate_index"))
    screeners = _load_completed(screener_path, ("record_id", "candidate_index"))
    validators = _load_completed(validator_path, ("record_id", "candidate_index"))

    pricing = config["budget"]
    spent = sum(
        float(event.get("usage", {}).get("cost_cny", 0))
        for events in (teachers, screeners, validators)
        for event in events.values()
    )
    ledger = BudgetLedger(
        hard_cap_cny=float(pricing["api_hard_cap_cny_equivalent"]),
        stop_fraction=float(pricing["stop_before_limit_fraction"]),
        spent_cny=spent,
    )
    teacher_key = require_api_key(config["teacher"]["api_key_env"])
    validator_key = require_api_key(config["validator"]["api_key_env"])
    local_key = require_api_key(config["screener"]["api_key_env"], allow_empty=True)

    candidates = int(config["sampling"]["candidates_per_question"])
    rule_config = config["rule_filter"]
    total = len(records) * candidates
    for record in records:
        for candidate_index in range(candidates):
            key = (record["id"], candidate_index)
            if key in teachers:
                continue
            prompt = build_teacher_prompt(record["question"], record["choices"])
            call = _call_with_budget(
                role="teacher",
                config=config["teacher"],
                system_prompt=TEACHER_SYSTEM_PROMPT,
                user_prompt=prompt,
                ledger=ledger,
                pricing=pricing,
                api_key=teacher_key,
                response_format_json=False,
            )
            if call["status"] == "complete":
                checked = check_teacher_response(
                    call["content"],
                    record["choices"],
                    record["answer"],
                    min_steps=int(rule_config["min_steps"]),
                    max_steps=int(rule_config["max_steps"]),
                    max_trajectory_characters=int(rule_config["max_trajectory_characters"]),
                    max_step_characters=int(rule_config["max_step_characters"]),
                    finish_reason=call["finish_reason"],
                ).to_dict()
            else:
                checked = {
                    "passed": False,
                    "steps": [],
                    "predicted_answer": None,
                    "failure_codes": ["api_error"],
                }
            event = {
                "schema_version": "medtrace.teacher-event.v1",
                "record_id": record["id"],
                "candidate_index": candidate_index,
                "record": record,
                "status": call["status"],
                "raw_response": call["content"],
                "request_id": call["request_id"],
                "finish_reason": call["finish_reason"],
                "usage": call["usage"],
                "attempts": call["attempts"],
                "errors": call["errors"],
                "reasoning_content_sha256": call["reasoning_content_sha256"],
                "rule_check": checked,
            }
            _append_jsonl(teacher_path, event)
            teachers[key] = event
            print(f"teacher {len(teachers)}/{total} cost={ledger.spent_cny:.4f} CNY", flush=True)

    eligible_teachers = {
        key: event for key, event in teachers.items() if event["rule_check"]["passed"]
    }
    for key, teacher_event in eligible_teachers.items():
        if key in screeners:
            continue
        record = teacher_event["record"]
        rule = teacher_event["rule_check"]
        prompt = build_screener_prompt(
            record["question"], record["choices"], record["answer"],
            rule["steps"], rule["predicted_answer"],
        )
        call = _call_with_budget(
            role="screener",
            config=config["screener"],
            system_prompt=SCREENER_SYSTEM_PROMPT,
            user_prompt=prompt,
            ledger=ledger,
            pricing=pricing,
            api_key=local_key,
            response_format_json=True,
            validate=lambda value, count=len(rule["steps"]): validate_screener_result(value, count),
        )
        if call["status"] == "complete":
            result = call["parsed"]
            status = "complete"
        else:
            result = {
                "verdict": "review",
                "suspected_first_error_step": None,
                "error_codes": ["judge_parse_error"],
                "concise_reason": "local screener failed; escalated to strong verifier",
            }
            status = "parse_error"
        event = {
            "schema_version": "medtrace.screener-event.v1",
            "record_id": key[0], "candidate_index": key[1], "status": status,
            "result": result, "request_id": call["request_id"], "usage": call["usage"],
            "attempts": call["attempts"], "errors": call["errors"],
        }
        _append_jsonl(screener_path, event)
        screeners[key] = event
        print(f"screener {len(screeners)}/{len(eligible_teachers)}", flush=True)

    validation_keys = {
        key for key, event in screeners.items() if event["result"]["verdict"] in {"pass", "review"}
    }
    for key in sorted(validation_keys):
        if key in validators:
            continue
        teacher_event = teachers[key]
        record = teacher_event["record"]
        rule = teacher_event["rule_check"]
        prompt = build_validator_prompt(
            record["question"], record["choices"], record["answer"],
            rule["steps"], rule["predicted_answer"],
        )
        call = _call_with_budget(
            role="validator",
            config=config["validator"],
            system_prompt=VALIDATOR_SYSTEM_PROMPT,
            user_prompt=prompt,
            ledger=ledger,
            pricing=pricing,
            api_key=validator_key,
            response_format_json=True,
            validate=lambda value, count=len(rule["steps"]): validate_validator_result(value, count),
        )
        event = {
            "schema_version": "medtrace.validator-event.v1",
            "record_id": key[0], "candidate_index": key[1], "status": call["status"],
            "result": call["parsed"], "request_id": call["request_id"], "usage": call["usage"],
            "attempts": call["attempts"], "errors": call["errors"],
            "reasoning_content_sha256": call["reasoning_content_sha256"],
        }
        _append_jsonl(validator_path, event)
        validators[key] = event
        print(f"validator {len(validators)}/{len(validation_keys)} cost={ledger.spent_cny:.4f} CNY", flush=True)

    canonical = [
        value
        for key in sorted(validators)
        if (value := _canonical_from_events(teachers[key], screeners[key], validators[key], config)) is not None
    ]
    _write_jsonl(run_dir / "canonical_trajectories.jsonl", canonical)
    sft = [item for item in canonical if item["disposition"] == "sft_accept"]
    _write_jsonl(run_dir / "sft_verified.jsonl", sft)
    prm: list[dict[str, Any]] = []
    for item in canonical:
        for step in item["verification"]["steps"]:
            prm.append({
                "trajectory_id": item["trajectory_id"], "step_index": step["index"],
                "prefix": [value["text"] for value in item["trajectory"]["steps"][:step["index"] + 1]],
                "label": step["prefix_label"], "error_codes": step["error_codes"],
                "source": item["source"],
            })
    _write_jsonl(run_dir / "process_train.jsonl", prm)
    metadata = {
        "schema_version": "medtrace.cot-real-pilot.v1", "status": "complete",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(), "dry_run": False,
        "config_sha256": config_sha256, "questions_sha256": questions_sha256,
        "questions": len(records), "teacher_events": len(teachers),
        "rule_passed": len(eligible_teachers), "screener_events": len(screeners),
        "validator_events": len(validators), "canonical_trajectories": len(canonical),
        "sft_records": len(sft), "prm_records": len(prm),
        "spent_cny_equivalent": round(ledger.spent_cny, 8),
        "budget_stop_limit_cny": ledger.stop_limit_cny,
        "preflight": preflight,
    }
    _write_json(metadata_path, metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
