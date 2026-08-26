"""Prepare and execute the frozen 24-case PRM negative-sample canary."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Callable

import yaml

from data_pipeline.build_strict_pilot_view import _sha256, _source_hashes
from data_pipeline.cot_api import (
    get_models,
    require_api_key,
    validate_validator_result,
    validator_response_format,
)
from data_pipeline.cot_budget import BudgetLedger
from data_pipeline.cot_prompts import (
    PRM_MUTATOR_SYSTEM_PROMPT,
    PRM_STUDENT_SYSTEM_PROMPT,
    VALIDATOR_SYSTEM_PROMPT,
    build_controlled_mutation_prompt,
    build_prm_student_prompt,
    build_validator_strict_prompt,
)
from data_pipeline.cot_rules import check_teacher_response
from data_pipeline.prm_negative_canary_config import (
    validate_prm_negative_canary_config,
)
from data_pipeline.prm_negative_policy import structurally_usable_wrong_answer
from data_pipeline.run_cot_pilot_real import (
    _append_jsonl,
    _call_with_budget,
    _load_completed,
    _load_jsonl,
    _write_json,
    _write_jsonl,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cot/prm_negative_canary_v1.yaml")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--preflight-only", action="store_true")
    group.add_argument(
        "--execute-canary-24",
        action="store_true",
        help="execute the approved local generation and 24 paid validator calls",
    )
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


def _event_key(event: dict[str, Any]) -> tuple[str, int]:
    return str(event["record_id"]), int(event["candidate_index"])


def _rank(seed: int, namespace: str, *values: object) -> str:
    payload = ":".join([str(seed), namespace, *(str(value) for value in values)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _benchmark(record: dict[str, Any]) -> str:
    value = str(record.get("benchmark") or "")
    if value not in {"medqa", "medmcqa"}:
        raise ValueError("candidate record has an unexpected benchmark")
    return value


def _record_identity(record: dict[str, Any]) -> str:
    return str(record["content_sha256"])


def _ordered_records(
    records: list[dict[str, Any]], seed: int, namespace: str
) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for record in records:
        unique.setdefault(_record_identity(record), record)
    return sorted(
        unique.values(),
        key=lambda record: _rank(seed, namespace, _record_identity(record)),
    )


def select_existing_natural(
    teachers: list[dict[str, Any]], seed: int, quota_by_benchmark: dict[str, int]
) -> list[dict[str, Any]]:
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in teachers:
        if structurally_usable_wrong_answer(event):
            pools[_benchmark(event["record"])].append(event)
    selected: list[dict[str, Any]] = []
    for benchmark, quota in quota_by_benchmark.items():
        ordered = sorted(
            pools.get(benchmark, []),
            key=lambda event: _rank(
                seed, "existing", _record_identity(event["record"]),
                event["candidate_index"],
            ),
        )
        distinct: list[dict[str, Any]] = []
        repeated: list[dict[str, Any]] = []
        seen: set[str] = set()
        for event in ordered:
            identity = _record_identity(event["record"])
            target = distinct if identity not in seen else repeated
            target.append(event)
            seen.add(identity)
        chosen = [*distinct, *repeated][: int(quota)]
        if len(chosen) != int(quota):
            raise RuntimeError(f"not enough natural candidates for {benchmark}")
        selected.extend(chosen)
    return selected


def _candidate_id(
    *,
    origin: str,
    source_hash: str,
    steps: list[str],
    predicted_answer: str,
    parent_trajectory_id: str | None,
    intended_error_step: int | None,
) -> str:
    payload = {
        "origin": origin,
        "source_hash": source_hash,
        "steps": steps,
        "predicted_answer": predicted_answer,
        "parent_trajectory_id": parent_trajectory_id,
        "intended_error_step": intended_error_step,
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


def _source_from_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": _benchmark(record),
        "source_id": str(record["source_id"]),
        "content_sha256": str(record["content_sha256"]),
        "split": str(record["split"]),
    }


def build_candidate(
    *,
    record: dict[str, Any],
    origin: str,
    steps: list[str],
    predicted_answer: str,
    generator_role: str,
    model_id: str,
    model_revision: str | None,
    prompt_version: str,
    run_git_commit: str,
    parent_trajectory_id: str | None = None,
    intended_error_step: int | None = None,
) -> dict[str, Any]:
    source = _source_from_record(record)
    candidate = {
        "schema_version": "medtrace.prm-negative-candidate.v1",
        "candidate_id": _candidate_id(
            origin=origin,
            source_hash=source["content_sha256"],
            steps=steps,
            predicted_answer=predicted_answer,
            parent_trajectory_id=parent_trajectory_id,
            intended_error_step=intended_error_step,
        ),
        "source": source,
        "origin": origin,
        "parent_trajectory_id": parent_trajectory_id,
        "trajectory": {"steps": steps, "predicted_answer": predicted_answer},
        "provenance": {
            "generator_role": generator_role,
            "model_id": model_id,
            "model_revision": model_revision,
            "prompt_version": prompt_version,
            "source_run_git_commit": run_git_commit,
        },
        "intended_error_step": intended_error_step,
        "disposition": "requires_independent_validation",
    }
    validate_candidate(candidate)
    return candidate


def validate_candidate(candidate: dict[str, Any]) -> None:
    required = {
        "schema_version", "candidate_id", "source", "origin",
        "parent_trajectory_id", "trajectory", "provenance",
        "intended_error_step", "disposition",
    }
    if set(candidate) != required:
        raise ValueError("candidate keys differ from contract")
    if candidate["schema_version"] != "medtrace.prm-negative-candidate.v1":
        raise ValueError("candidate schema_version changed")
    if not str(candidate["candidate_id"]).startswith("sha256:"):
        raise ValueError("candidate id is not content-addressed")
    source = candidate["source"]
    if source.get("dataset") not in {"medqa", "medmcqa"} or source.get("split") != "train":
        raise ValueError("candidate source is outside the train boundary")
    trajectory = candidate["trajectory"]
    steps = trajectory.get("steps")
    if not isinstance(steps, list) or not 3 <= len(steps) <= 8:
        raise ValueError("candidate step count is outside [3, 8]")
    if not all(isinstance(step, str) and step.strip() for step in steps):
        raise ValueError("candidate contains an invalid step")
    origin = candidate["origin"]
    if origin == "controlled_single_error":
        target = candidate["intended_error_step"]
        if type(target) is not int or not 0 <= target < len(steps):
            raise ValueError("controlled candidate target is invalid")
        if candidate["parent_trajectory_id"] is None:
            raise ValueError("controlled candidate has no parent")
    elif candidate["intended_error_step"] is not None:
        raise ValueError("non-controlled candidate has an intended error label")
    if candidate["disposition"] != "requires_independent_validation":
        raise ValueError("candidate was assigned a label before validation")


def _rule_is_structurally_usable(rule: dict[str, Any], record: dict[str, Any]) -> bool:
    failures = set(str(value) for value in rule.get("failure_codes") or [])
    steps = rule.get("steps")
    predicted = rule.get("predicted_answer")
    return (
        failures.issubset({"gold_answer_mismatch"})
        and isinstance(steps, list)
        and 3 <= len(steps) <= 8
        and all(isinstance(step, str) and step.strip() for step in steps)
        and isinstance(predicted, str)
        and predicted in (record.get("choices") or {})
    )


def _record_from_canonical(record: dict[str, Any]) -> dict[str, Any]:
    source = record["source"]
    problem = record["problem"]
    return {
        "benchmark": source["dataset"],
        "source_id": source["source_id"],
        "content_sha256": source["content_sha256"],
        "split": source["split"],
        "question": problem["question"],
        "choices": problem["choices"],
        "answer": problem["gold_answer"],
    }


def _target_index(position: str, step_count: int) -> int:
    if position == "early":
        return 0
    if position == "middle":
        return (step_count - 1) // 2
    if position == "late":
        return step_count - 1
    raise ValueError(f"unknown controlled target position: {position}")


def _local_generation_call(
    config: dict[str, Any],
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    *,
    deadline_epoch: float,
    response_format_json: bool = False,
    validate: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    remaining = deadline_epoch - time.time()
    attempts = int(config["local_model"].get("max_retries", 0)) + 1
    retry_sleep_reserve = max(0, attempts - 1)
    if remaining <= retry_sleep_reserve + attempts:
        raise RuntimeError("local GPU generation window has reached its one-hour cap")
    local_config = dict(config["local_model"])
    local_config["timeout_seconds"] = min(
        float(local_config["timeout_seconds"]),
        (remaining - retry_sleep_reserve) / attempts,
    )
    return _call_with_budget(
        role="screener",
        config=local_config,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        ledger=BudgetLedger(hard_cap_cny=1, stop_fraction=1),
        pricing=config["budget"],
        api_key=api_key,
        response_format_json=response_format_json,
        validate=validate,
    )


def _validate_replacement(value: dict[str, Any]) -> dict[str, Any]:
    if set(value) != {"replacement_step"}:
        raise ValueError("controlled mutation JSON keys differ")
    replacement = value["replacement_step"]
    if (
        not isinstance(replacement, str)
        or not replacement.strip()
        or len(replacement) > 3000
        or "<step>" in replacement
        or "<answer>" in replacement
    ):
        raise ValueError("controlled mutation replacement is invalid")
    value["replacement_step"] = replacement.strip()
    return value


def _preflight(config: dict[str, Any]) -> dict[str, Any]:
    state = _git_state()
    if state["git_dirty"]:
        raise RuntimeError("PRM negative canary requires a clean Git worktree")
    local = config["local_model"]
    validator = config["validator"]
    local_key = require_api_key(str(local["api_key_env"]), allow_empty=True)
    validator_key = require_api_key(str(validator["api_key_env"]))
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    ).stdout.strip().splitlines()[0]
    parts = [part.strip() for part in gpu.split(",")]
    if int(parts[-1]) < 20_000:
        raise RuntimeError("PRM canary requires the approved ~24 GB GPU")
    manifest_path = _repo_path(str(local["runtime_manifest"]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "git_commit": state["git_commit"],
        "git_dirty": False,
        "model_id": local["model_id"],
        "model_revision": local["model_revision"],
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise RuntimeError(f"local runtime manifest mismatch: {field}")
    if (manifest.get("packages") or {}).get("vllm") != local["expected_vllm_version"]:
        raise RuntimeError("local runtime vLLM version mismatch")
    model_checks = (
        (local, local_key),
        (validator, validator_key),
    )
    for spec, key in model_checks:
        available = get_models(
            str(spec["base_url"]), key, float(spec["timeout_seconds"])
        )
        if spec["model_id"] not in available:
            raise RuntimeError(f"configured model is unavailable: {spec['model_id']}")
    return {
        "status": "passed",
        "git_commit": state["git_commit"],
        "gpu_name": parts[0],
        "local_model": local["model_id"],
        "validator_model": validator["model_id"],
        "api_keys_present": {
            local["api_key_env"]: bool(os.environ.get(str(local["api_key_env"]))),
            validator["api_key_env"]: True,
        },
    }


def _prepare_manifest(
    path: Path,
    *,
    config_sha256: str,
    source_hashes: dict[str, str],
    git_commit: str,
) -> dict[str, Any]:
    frozen = {
        "schema_version": "medtrace.prm-negative-canary-private-manifest.v1",
        "config_sha256": config_sha256,
        "source_artifact_sha256": source_hashes,
        "runtime_git_commit": git_commit,
    }
    if path.exists():
        value = json.loads(path.read_text(encoding="utf-8"))
        if any(value.get(key) != expected for key, expected in frozen.items()):
            raise RuntimeError("cannot resume: PRM canary private manifest changed")
        if not isinstance(value.get("local_generation_started_at_epoch"), (int, float)):
            raise RuntimeError("cannot resume: local GPU generation window is missing")
    else:
        value = {
            **frozen,
            "local_generation_started_at_epoch": time.time(),
        }
        _write_json(path, value)
    return value


def _event_map(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(key[0]): event
        for key, event in _load_completed(path, ("attempt_id",)).items()
    }


def _existing_candidates(
    events: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    commit = str(config["source_identity"]["generation_git_commit"])
    return [
        build_candidate(
            record=event["record"],
            origin="existing_teacher_answer_mismatch",
            steps=list(event["rule_check"]["steps"]),
            predicted_answer=str(event["rule_check"]["predicted_answer"]),
            generator_role="existing_teacher",
            model_id="qwen3-max-2026-01-23",
            model_revision=None,
            prompt_version="teacher_v1",
            run_git_commit=commit,
        )
        for event in events
    ]


def _generate_students(
    *,
    config: dict[str, Any],
    records: list[dict[str, Any]],
    output_dir: Path,
    local_key: str,
    git_commit: str,
    excluded_questions: set[str],
    deadline_epoch: float,
) -> list[dict[str, Any]]:
    path = output_dir / "student_generation_events.jsonl"
    events = _event_map(path)
    quota = config["selection"]["each_origin_by_benchmark"]
    seed = int(config["selection"]["seed"])
    cap = int(config["local_model"]["max_attempted_questions_per_origin_and_benchmark"])
    candidates: list[dict[str, Any]] = []
    for benchmark, target in quota.items():
        accepted = [
            event["candidate"] for event in events.values()
            if event.get("benchmark") == benchmark and event.get("candidate") is not None
        ]
        attempted = sum(event.get("benchmark") == benchmark for event in events.values())
        pool = [record for record in records if _benchmark(record) == benchmark]
        pool = _ordered_records(pool, seed, f"student:{benchmark}")
        pool.sort(key=lambda record: _record_identity(record) in excluded_questions)
        for record in pool:
            if len(accepted) >= int(target) or attempted >= cap:
                break
            attempt_id = f"student:{_record_identity(record)}"
            if attempt_id in events:
                continue
            prompt = build_prm_student_prompt(record["question"], record["choices"])
            call = _local_generation_call(
                config, PRM_STUDENT_SYSTEM_PROMPT, prompt, local_key,
                deadline_epoch=deadline_epoch,
            )
            rule: dict[str, Any] | None = None
            candidate: dict[str, Any] | None = None
            status = call["status"]
            if status == "complete":
                rule = check_teacher_response(
                    call["content"], record["choices"], record["answer"]
                ).to_dict()
                if _rule_is_structurally_usable(rule, record):
                    candidate = build_candidate(
                        record=record,
                        origin="local_student",
                        steps=list(rule["steps"]),
                        predicted_answer=str(rule["predicted_answer"]),
                        generator_role="local_student",
                        model_id=config["local_model"]["model_id"],
                        model_revision=config["local_model"]["model_revision"],
                        prompt_version=config["student"]["prompt_version"],
                        run_git_commit=git_commit,
                    )
                    status = "accepted"
                else:
                    status = "rule_reject"
            event = {
                "schema_version": "medtrace.prm-student-generation-event.v1",
                "attempt_id": attempt_id,
                "benchmark": benchmark,
                "status": status,
                "record": record,
                "raw_response": call["content"],
                "rule_check": rule,
                "candidate": candidate,
                "request_id": call["request_id"],
                "attempt_diagnostics": call["attempt_diagnostics"],
            }
            _append_jsonl(path, event)
            events[attempt_id] = event
            attempted += 1
            if candidate is not None:
                accepted.append(candidate)
        if len(accepted) != int(target):
            raise RuntimeError(
                f"local student produced {len(accepted)}/{target} candidates for {benchmark}"
            )
        candidates.extend(accepted)
    return candidates


def _generate_mutations(
    *,
    config: dict[str, Any],
    canonical: list[dict[str, Any]],
    output_dir: Path,
    local_key: str,
    git_commit: str,
    excluded_questions: set[str],
    deadline_epoch: float,
) -> list[dict[str, Any]]:
    path = output_dir / "controlled_mutation_events.jsonl"
    events = _event_map(path)
    quota = config["selection"]["each_origin_by_benchmark"]
    seed = int(config["selection"]["seed"])
    cap = int(config["local_model"]["max_attempted_questions_per_origin_and_benchmark"])
    cycle = list(config["controlled_mutator"]["target_position_cycle"])
    candidates: list[dict[str, Any]] = []
    for benchmark, target in quota.items():
        accepted = [
            event["candidate"] for event in events.values()
            if event.get("benchmark") == benchmark and event.get("candidate") is not None
        ]
        attempted = sum(event.get("benchmark") == benchmark for event in events.values())
        pool = [
            record for record in canonical
            if record["source"]["dataset"] == benchmark
            and record["verification"]["trajectory_label"] == 1
        ]
        pool.sort(
            key=lambda record: _rank(seed, f"mutator:{benchmark}", record["trajectory_id"])
        )
        pool.sort(
            key=lambda record: record["source"]["content_sha256"] in excluded_questions
        )
        for parent in pool:
            if len(accepted) >= int(target) or attempted >= cap:
                break
            attempt_id = f"mutator:{parent['trajectory_id']}"
            if attempt_id in events:
                continue
            original_steps = [value["text"] for value in parent["trajectory"]["steps"]]
            position = cycle[len(accepted) % len(cycle)]
            target_index = _target_index(position, len(original_steps))
            record = _record_from_canonical(parent)
            prompt = build_controlled_mutation_prompt(
                record["question"], record["choices"], original_steps,
                parent["trajectory"]["predicted_answer"], target_index,
            )
            call = _local_generation_call(
                config, PRM_MUTATOR_SYSTEM_PROMPT, prompt, local_key,
                deadline_epoch=deadline_epoch,
                response_format_json=True,
                validate=_validate_replacement,
            )
            rule: dict[str, Any] | None = None
            candidate: dict[str, Any] | None = None
            status = call["status"]
            if status == "complete":
                replacement = call["parsed"]["replacement_step"]
                steps = list(original_steps)
                steps[target_index] = replacement
                synthetic = "".join(f"<step>{step}</step>" for step in steps)
                synthetic += (
                    f"<answer>{parent['trajectory']['predicted_answer']}</answer>"
                )
                rule = check_teacher_response(
                    synthetic, record["choices"], record["answer"]
                ).to_dict()
                controlled = (
                    rule["passed"]
                    and steps[target_index] != original_steps[target_index]
                    and rule["predicted_answer"]
                    == parent["trajectory"]["predicted_answer"]
                )
                if controlled:
                    candidate = build_candidate(
                        record=record,
                        origin="controlled_single_error",
                        steps=steps,
                        predicted_answer=str(rule["predicted_answer"]),
                        generator_role="controlled_mutator",
                        model_id=config["local_model"]["model_id"],
                        model_revision=config["local_model"]["model_revision"],
                        prompt_version=config["controlled_mutator"]["prompt_version"],
                        run_git_commit=git_commit,
                        parent_trajectory_id=parent["trajectory_id"],
                        intended_error_step=target_index,
                    )
                    status = "accepted"
                else:
                    status = "control_reject"
            event = {
                "schema_version": "medtrace.prm-controlled-mutation-event.v1",
                "attempt_id": attempt_id,
                "benchmark": benchmark,
                "status": status,
                "record": record,
                "parent_trajectory_id": parent["trajectory_id"],
                "target_step": target_index,
                "raw_response": call["content"],
                "rule_check": rule,
                "candidate": candidate,
                "request_id": call["request_id"],
                "attempt_diagnostics": call["attempt_diagnostics"],
            }
            _append_jsonl(path, event)
            events[attempt_id] = event
            attempted += 1
            if candidate is not None:
                accepted.append(candidate)
        if len(accepted) != int(target):
            raise RuntimeError(
                f"controlled mutator produced {len(accepted)}/{target} candidates for {benchmark}"
            )
        candidates.extend(accepted)
    return candidates


def _candidate_counts(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    by_origin = Counter(candidate["origin"] for candidate in candidates)
    by_benchmark = Counter(candidate["source"]["dataset"] for candidate in candidates)
    by_origin_benchmark = Counter(
        f"{candidate['origin']}:{candidate['source']['dataset']}"
        for candidate in candidates
    )
    return {
        "total": len(candidates),
        "by_origin": dict(sorted(by_origin.items())),
        "by_benchmark": dict(sorted(by_benchmark.items())),
        "by_origin_benchmark": dict(sorted(by_origin_benchmark.items())),
    }


def _validate_candidate_mix(candidates: list[dict[str, Any]], config: dict[str, Any]) -> None:
    if len({candidate["candidate_id"] for candidate in candidates}) != len(candidates):
        raise RuntimeError("PRM canary candidate IDs are not unique")
    counts = _candidate_counts(candidates)
    if counts["total"] != int(config["selection"]["total"]):
        raise RuntimeError("PRM canary candidate total changed")
    if counts["by_origin"] != config["selection"]["by_origin"]:
        raise RuntimeError("PRM canary origin mix changed")
    if counts["by_benchmark"] != {"medmcqa": 12, "medqa": 12}:
        raise RuntimeError("PRM canary benchmark mix changed")
    for origin in config["selection"]["by_origin"]:
        for benchmark, quota in config["selection"]["each_origin_by_benchmark"].items():
            if counts["by_origin_benchmark"].get(f"{origin}:{benchmark}") != int(quota):
                raise RuntimeError("PRM canary origin/benchmark mix changed")


def main() -> None:
    args = parse_args()
    config_path = _repo_path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("PRM negative canary config must be an object")
    validate_prm_negative_canary_config(config)
    run_dir = _repo_path(str(config["source_run_dir"]))
    hashes_before = _source_hashes(run_dir)
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    identity = config["source_identity"]
    actual_identity = {
        "config_sha256": metadata.get("config_sha256"),
        "questions_sha256": metadata.get("questions_sha256"),
        "generation_git_commit": (
            (metadata.get("preflight") or {}).get("screener_runtime") or {}
        ).get("git_commit"),
        "questions": metadata.get("questions"),
        "teacher_events": metadata.get("teacher_events"),
        "canonical_trajectories": metadata.get("canonical_trajectories"),
        "prm_records": metadata.get("prm_records"),
    }
    if actual_identity != identity:
        raise RuntimeError("PRM canary source identity mismatch")
    teachers_list = _load_jsonl(run_dir / "teacher_events.jsonl")
    quota = {
        key: int(value)
        for key, value in config["selection"]["each_origin_by_benchmark"].items()
    }
    natural_events = select_existing_natural(
        teachers_list, int(config["selection"]["seed"]), quota
    )
    preview = {
        "schema_version": "medtrace.prm-negative-canary-preview.v1",
        "contains_private_text_or_ids": False,
        "existing_natural_selected": len(natural_events),
        "target_total": int(config["selection"]["total"]),
        "target_by_origin": config["selection"]["by_origin"],
        "target_by_benchmark": {"medmcqa": 12, "medqa": 12},
        "api_hard_cap_cny_equivalent": config["budget"]["api_hard_cap_cny_equivalent"],
    }
    print(json.dumps(preview, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    if not args.preflight_only and not args.execute_canary_24:
        print("Preview only; no model request was made.", flush=True)
        return

    preflight = _preflight(config)
    print(json.dumps(preflight, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    strict_dir = run_dir / str(config["strict_source_subdir"])
    strict_manifest_path = strict_dir / "manifest.json"
    if not strict_manifest_path.is_file():
        raise RuntimeError("strict pilot view is missing; run build_strict_pilot_view first")
    strict_manifest = json.loads(strict_manifest_path.read_text(encoding="utf-8"))
    if strict_manifest.get("counts") != config["strict_source_expected"]:
        raise RuntimeError("strict pilot view counts changed")
    if strict_manifest.get("source_artifact_sha256") != hashes_before:
        raise RuntimeError("strict pilot view was built from different source artifacts")
    if strict_manifest.get("config_sha256") != _sha256(config_path):
        raise RuntimeError("strict pilot view was built with a different canary config")
    if args.preflight_only:
        return

    runtime_config = dict(config)
    runtime_config["execution_enabled"] = True
    validate_prm_negative_canary_config(runtime_config, execute=True)
    output_dir = run_dir / str(config["output_subdir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    private_manifest = _prepare_manifest(
        output_dir / "private_manifest.json",
        config_sha256=_sha256(config_path),
        source_hashes=hashes_before,
        git_commit=preflight["git_commit"],
    )
    local_deadline_epoch = float(
        private_manifest["local_generation_started_at_epoch"]
    ) + 3600 * float(config["budget"]["local_gpu_hours_cap"])
    local_key = require_api_key(
        str(config["local_model"]["api_key_env"]), allow_empty=True
    )
    natural = _existing_candidates(natural_events, config)
    all_records = _ordered_records(
        [event["record"] for event in teachers_list],
        int(config["selection"]["seed"]), "all-records",
    )
    used_questions = {candidate["source"]["content_sha256"] for candidate in natural}
    students = _generate_students(
        config=config, records=all_records, output_dir=output_dir,
        local_key=local_key, git_commit=preflight["git_commit"],
        excluded_questions=used_questions,
        deadline_epoch=local_deadline_epoch,
    )
    used_questions.update(
        candidate["source"]["content_sha256"] for candidate in students
    )
    strict_canonical = _load_jsonl(strict_dir / "canonical_trajectories.jsonl")
    controlled = _generate_mutations(
        config=config, canonical=strict_canonical, output_dir=output_dir,
        local_key=local_key, git_commit=preflight["git_commit"],
        excluded_questions=used_questions,
        deadline_epoch=local_deadline_epoch,
    )
    candidates = sorted(
        [*natural, *students, *controlled],
        key=lambda candidate: (
            candidate["origin"], candidate["source"]["dataset"],
            candidate["candidate_id"],
        ),
    )
    _validate_candidate_mix(candidates, config)
    candidate_path = output_dir / "candidates.jsonl"
    if candidate_path.exists():
        if _load_jsonl(candidate_path) != candidates:
            raise RuntimeError("cannot resume: PRM canary candidates changed")
    else:
        _write_jsonl(candidate_path, candidates)

    records_by_hash = {
        _record_identity(record): record for record in all_records
    }
    validator_path = output_dir / "validator_events.jsonl"
    completed = _load_completed(validator_path, ("candidate_id",))
    if any(str(key[0]) not in {item["candidate_id"] for item in candidates} for key in completed):
        raise RuntimeError("validator log contains an unknown candidate")
    pricing = config["budget"]
    ledger = BudgetLedger(
        hard_cap_cny=float(pricing["api_hard_cap_cny_equivalent"]),
        stop_fraction=float(pricing["stop_before_limit_fraction"]),
        spent_cny=sum(
            float(event.get("usage", {}).get("cost_cny", 0))
            for event in completed.values()
        ),
    )
    validator_key = require_api_key(str(config["validator"]["api_key_env"]))
    for candidate in candidates:
        key = (candidate["candidate_id"],)
        if key in completed:
            continue
        record = records_by_hash[candidate["source"]["content_sha256"]]
        trajectory = candidate["trajectory"]
        steps = trajectory["steps"]
        prompt = build_validator_strict_prompt(
            record["question"], record["choices"], record["answer"],
            steps, trajectory["predicted_answer"],
        )
        call = _call_with_budget(
            role="validator", config=config["validator"],
            system_prompt=VALIDATOR_SYSTEM_PROMPT, user_prompt=prompt,
            ledger=ledger, pricing=pricing, api_key=validator_key,
            response_format_json=True,
            response_format=validator_response_format(len(steps)),
            validate=lambda value, count=len(steps): validate_validator_result(
                value, count
            ),
        )
        event = {
            "schema_version": "medtrace.prm-negative-validator-event.v1",
            "candidate_id": candidate["candidate_id"],
            "origin": candidate["origin"],
            "benchmark": candidate["source"]["dataset"],
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
        _append_jsonl(validator_path, event)
        completed[key] = event
        print(
            f"PRM canary validator {len(completed)}/{len(candidates)} "
            f"status={call['status']} cost={ledger.spent_cny:.4f} CNY",
            flush=True,
        )
    if _source_hashes(run_dir) != hashes_before:
        raise RuntimeError("an immutable pilot artifact changed during PRM canary")
    metadata_out = {
        "schema_version": "medtrace.prm-negative-canary-run.v1",
        "status": "complete" if len(completed) == len(candidates) else "partial",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "contains_private_text_or_ids": False,
        "source_artifacts_unchanged": True,
        "runtime": preflight,
        "candidate_counts": _candidate_counts(candidates),
        "validator_events": len(completed),
        "spent_cny_equivalent": round(ledger.spent_cny, 8),
        "budget_stop_limit_cny": ledger.stop_limit_cny,
    }
    _write_json(output_dir / "metadata.json", metadata_out)
    print(json.dumps(metadata_out, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
