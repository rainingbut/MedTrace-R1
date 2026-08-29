"""Plan Step 2 scale-up from private real logs without model or GPU calls."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any, Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "medtrace.cot.step2-scaleup-plan.v1"
ORIGIN_CONTROLLED = "controlled_single_error"
ORIGIN_STUDENT = "local_student"
ORIGIN_MISMATCH = "existing_teacher_answer_mismatch"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/cot/step2_scaleup_plan_v1.yaml"
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--write-approved-report", action="store_true")
    return parser.parse_args()


def _repo_path(value: str, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"required private aggregate/log is missing: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected one JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        handle = path.open("r", encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"required private aggregate/log is missing: {path}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            records.append(value)
    return records


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _require_exact_keys(value: dict[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} keys differ")


def validate_config(config: dict[str, Any], *, write: bool = False) -> None:
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported Step 2 scale-up plan schema_version")
    enabled = config.get("write_enabled")
    if type(enabled) is not bool:
        raise ValueError("write_enabled must be boolean")
    if write and not enabled:
        raise ValueError("writing requires an enabled runtime copy")
    if not write and enabled:
        raise ValueError("committed scale-up config must remain write-disabled")
    if config.get("approval_status") != "approved_offline_planning_2026-08-29":
        raise ValueError("offline planning approval is missing")

    expected_files = {
        "pilot_audit": "quality_audit.json",
        "teacher_events": "teacher_events.jsonl",
        "screener_events": "screener_events.jsonl",
        "validator_events": "validator_events.jsonl",
        "strict_sft": "prm_negative_enrichment_v1/strict_source_v1/sft_verified.jsonl",
        "canary_audit": "prm_negative_enrichment_v1/canary_v1/quality_audit.json",
        "canary_candidates": "prm_negative_enrichment_v1/canary_v1/candidates.jsonl",
        "canary_validator_events": "prm_negative_enrichment_v1/canary_v1/validator_events.jsonl",
        "student_generation_events": "prm_negative_enrichment_v1/canary_v1/student_generation_events.jsonl",
        "controlled_generation_events": "prm_negative_enrichment_v1/canary_v1/controlled_mutation_events.jsonl",
        "recovery_audit": "prm_negative_enrichment_v1/canary_v1/validator_recovery_v1/quality_audit.json",
        "recovery_attempts": "prm_negative_enrichment_v1/canary_v1/validator_recovery_v1/recovery_attempts.jsonl",
        "adjudication_audit": "prm_negative_enrichment_v1/canary_v1/human_review_adjudication_audit_v1.json",
        "materialization_audit": "prm_negative_enrichment_v1/prm_negative_materialization_v1/audit.json",
    }
    source_files = config.get("source_files")
    if not isinstance(source_files, dict):
        raise ValueError("source_files must be an object")
    if source_files != expected_files:
        raise ValueError("source_files paths changed")
    if config.get("source_run_dir") != "results/cot/pilot_v1_real":
        raise ValueError("unexpected source_run_dir")
    for value in source_files.values():
        path = Path(str(value))
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("source files must stay under source_run_dir")

    identity = config.get("frozen_identity") or {}
    if identity.get("pilot_questions") != 40:
        raise ValueError("pilot question identity changed")
    if identity.get("candidates_per_question") != 4:
        raise ValueError("teacher candidate count changed")
    if identity.get("questions_per_benchmark") != {"medqa": 20, "medmcqa": 20}:
        raise ValueError("pilot benchmark balance changed")
    if identity.get("strict_sft_trajectories") != 107:
        raise ValueError("strict SFT source identity changed")
    if identity.get("strict_prm_trajectories") != 108:
        raise ValueError("strict PRM source identity changed")
    if identity.get("strict_positive_prefix_records") != 717:
        raise ValueError("strict positive-prefix count changed")
    if identity.get("strict_negative_prefix_records") != 4:
        raise ValueError("strict negative-prefix count changed")
    if identity.get("current_negative_trajectories") != 12:
        raise ValueError("current negative trajectory count changed")
    if identity.get("current_positive_prefix_records") != 749:
        raise ValueError("current positive-prefix count changed")
    if identity.get("current_negative_prefix_records") != 47:
        raise ValueError("current negative-prefix count changed")
    expected_origins = {
        ORIGIN_MISMATCH: 8, ORIGIN_STUDENT: 8, ORIGIN_CONTROLLED: 8,
    }
    if identity.get("canary_candidates_per_origin") != expected_origins:
        raise ValueError("negative canary origin denominators changed")

    scenarios = config.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("at least one scale-up scenario is required")
    names: set[str] = set()
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            raise ValueError("scenario must be an object")
        _require_exact_keys(
            scenario,
            {"name", "new_questions", "target_total_negative_trajectories"},
            "scenario",
        )
        name = str(scenario["name"])
        if name in names:
            raise ValueError("scenario names must be unique")
        names.add(name)
        questions = int(scenario["new_questions"])
        negatives = int(scenario["target_total_negative_trajectories"])
        if questions <= 0 or questions % 2:
            raise ValueError("new_questions must be positive and benchmark-balanced")
        if negatives < int(identity["current_negative_trajectories"]):
            raise ValueError("negative target cannot be below the current count")
    if config.get("recommended_scenario") not in names:
        raise ValueError("recommended_scenario is unknown")
    if scenarios != [
        {"name": "checkpoint", "new_questions": 1000, "target_total_negative_trajectories": 200},
        {"name": "minimum", "new_questions": 2500, "target_total_negative_trajectories": 300},
        {"name": "recommended", "new_questions": 2500, "target_total_negative_trajectories": 500},
        {"name": "robust", "new_questions": 5000, "target_total_negative_trajectories": 1000},
    ] or config.get("recommended_scenario") != "recommended":
        raise ValueError("frozen scenario matrix changed")

    negative = config.get("negative_planning") or {}
    weights = negative.get("guaranteed_origins")
    if not isinstance(weights, dict) or set(weights) != {
        ORIGIN_CONTROLLED, ORIGIN_STUDENT,
    }:
        raise ValueError("exactly two scalable negative origins are required")
    if any(float(weight) <= 0 for weight in weights.values()) or not math.isclose(
        sum(float(weight) for weight in weights.values()), 1.0, abs_tol=1e-9
    ):
        raise ValueError("negative origin weights must be positive and sum to one")
    if negative.get("opportunistic_origins") != [ORIGIN_MISMATCH]:
        raise ValueError("answer mismatch must remain opportunistic")
    if float(negative.get("yield_lower_bound_z", 0)) <= 0:
        raise ValueError("yield lower-bound z must be positive")
    if negative.get("require_distinct_trajectories") is not True:
        raise ValueError("negative trajectories must remain distinct")
    if negative.get("balance_benchmarks") is not True:
        raise ValueError("negative candidates must balance benchmarks")
    ratio_target = float(negative.get("prm_positive_to_negative_ratio_target", 0))
    ratio_bounds = negative.get("prm_positive_to_negative_ratio_bounds")
    if ratio_bounds != [3.0, 8.0] or not 3.0 <= ratio_target <= 8.0:
        raise ValueError("PRM positive/negative ratio policy changed")

    if config.get("dataset_split_policy") != {
        "grouping_unit": "source_question",
        "forbid_group_cross_split": True,
        "evaluation_benchmark_records_forbidden": True,
        "current_12_negatives_default_to_audit_holdout": True,
        "sft_trajectory_split": {"train": 0.95, "validation": 0.05},
        "prm_negative_trajectory_split": {
            "train": 0.80, "validation": 0.10, "test": 0.10,
        },
    }:
        raise ValueError("dataset split policy changed")

    budget = config.get("budget_policy") or {}
    estimate = float(budget.get("estimate_safety_factor", 0))
    stop = float(budget.get("stop_line_factor", 0))
    hard = float(budget.get("hard_cap_factor", 0))
    if not (1 <= estimate < stop < hard):
        raise ValueError("budget factors must satisfy 1 <= estimate < stop < hard")
    if float(budget.get("observed_local_gpu_hourly_rate_cny", -1)) < 0:
        raise ValueError("observed GPU hourly rate cannot be negative")
    if budget.get("require_rate_reconfirmation_before_paid_run") is not True:
        raise ValueError("GPU rate must be reconfirmed before a paid run")
    if budget != {
        "estimate_safety_factor": 1.15,
        "stop_line_factor": 1.25,
        "hard_cap_factor": 1.40,
        "round_cny_up_to": 1,
        "observed_local_gpu_hourly_rate_cny": 2.5,
        "observed_rate_provenance": "docs/FULL_BASELINE_HANDOFF.md",
        "require_rate_reconfirmation_before_paid_run": True,
        "pilot_local_gpu_hours_cap": 1,
        "negative_canary_local_gpu_hours_cap": 1,
    }:
        raise ValueError("budget policy changed")

    proposal = config.get("paid_canary_proposal") or {}
    if proposal.get("questions_per_benchmark") != {"medqa": 50, "medmcqa": 50}:
        raise ValueError("paid canary must balance 100 questions")
    if proposal.get("new_questions") != 100:
        raise ValueError("paid canary question count changed")
    if proposal.get("negative_candidates") != {
        ORIGIN_CONTROLLED: 32, ORIGIN_STUDENT: 8,
    }:
        raise ValueError("paid canary negative mix changed")
    if proposal.get("quality_gates") != {
        "teacher_rule_pass_rate_minimum": 0.75,
        "strict_sft_per_question_minimum": 2.0,
        "validator_contract_success_rate_minimum": 0.98,
        "controlled_exact_first_error_rate_minimum": 0.80,
        "human_trajectory_label_accuracy_minimum": 0.90,
        "human_exact_first_error_accuracy_minimum": 0.80,
        "require_both_benchmarks": True,
        "require_all_cost_ledgers_complete": True,
    }:
        raise ValueError("paid canary quality gates changed")

    if config.get("outputs") != {
        "aggregate_json": "reports/step2_scaleup_budget_v1.json",
        "aggregate_markdown": "reports/step2_scaleup_budget_v1.md",
    }:
        raise ValueError("aggregate output paths changed")

    authorization = config.get("authorization")
    expected_authorization = {
        "offline_only": True,
        "model_or_api_calls_authorized": False,
        "gpu_inference_authorized": False,
        "data_generation_authorized": False,
        "training_authorized": False,
        "git_push_authorized": False,
        "paid_canary_authorized": False,
        "full_scale_generation_authorized": False,
    }
    if authorization != expected_authorization:
        raise ValueError("offline authorization boundary changed")


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Step 2 scale-up config must be an object")
    validate_config(value)
    return value


def wilson_lower_bound(successes: int, total: int, z: float) -> float:
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("invalid binomial observation")
    probability = successes / total
    denominator = 1 + z * z / total
    centre = probability + z * z / (2 * total)
    margin = z * math.sqrt(
        probability * (1 - probability) / total + z * z / (4 * total * total)
    )
    return max(0.0, (centre - margin) / denominator)


def _usage_total(events: Iterable[dict[str, Any]], field: str) -> float:
    total = 0.0
    for event in events:
        usage = event.get("usage") or {}
        value = usage.get(field, 0)
        if value is None:
            value = 0
        number = float(value)
        if number < 0:
            raise ValueError(f"negative usage field: {field}")
        total += number
    return total


def _physical_attempts(events: Iterable[dict[str, Any]]) -> int:
    total = 0
    for event in events:
        attempts = int(event.get("attempts", 1) or 1)
        if attempts <= 0:
            raise ValueError("physical request attempts must be positive")
        total += attempts
    return total


def _accepted_generation(events: Iterable[dict[str, Any]]) -> int:
    return sum(event.get("candidate") is not None for event in events)


def _allocate(total: int, weights: dict[str, float]) -> dict[str, int]:
    raw = {key: total * value for key, value in weights.items()}
    result = {key: math.floor(value) for key, value in raw.items()}
    remaining = total - sum(result.values())
    order = sorted(weights, key=lambda key: (raw[key] - result[key], key), reverse=True)
    for key in order[:remaining]:
        result[key] += 1
    return dict(sorted(result.items()))


def _round_up(value: float, unit: int) -> float:
    return float(math.ceil(value / unit) * unit)


def _safe_ratio(numerator: float, denominator: float, name: str) -> float:
    if denominator <= 0:
        raise ValueError(f"zero denominator for {name}")
    return numerator / denominator


def collect_observations(
    config: dict[str, Any], *, repo_root: Path = REPO_ROOT
) -> dict[str, Any]:
    run_dir = _repo_path(str(config["source_run_dir"]), repo_root)
    paths = {
        key: run_dir / str(value) for key, value in config["source_files"].items()
    }
    pilot = _load_json(paths["pilot_audit"])
    canary = _load_json(paths["canary_audit"])
    recovery = _load_json(paths["recovery_audit"])
    adjudication = _load_json(paths["adjudication_audit"])
    materialization = _load_json(paths["materialization_audit"])
    teachers = _load_jsonl(paths["teacher_events"])
    screeners = _load_jsonl(paths["screener_events"])
    validators = _load_jsonl(paths["validator_events"])
    strict_sft = _load_jsonl(paths["strict_sft"])
    canary_candidates = _load_jsonl(paths["canary_candidates"])
    canary_validators = _load_jsonl(paths["canary_validator_events"])
    recovery_attempts = _load_jsonl(paths["recovery_attempts"])
    student_events = _load_jsonl(paths["student_generation_events"])
    controlled_events = _load_jsonl(paths["controlled_generation_events"])

    identity = config["frozen_identity"]
    materialized = materialization.get("materialized_candidates") or {}
    materialized_origins = materialized.get("origins") or {}
    expected_canary_origins = identity["canary_candidates_per_origin"]
    candidate_origins = Counter(str(value.get("origin")) for value in canary_candidates)
    teacher_benchmarks = Counter(
        str((event.get("record") or {}).get("benchmark")) for event in teachers
    )
    strict_sft_benchmarks = Counter(
        str((record.get("source") or {}).get("dataset")) for record in strict_sft
    )
    gates = {
        "pilot_invariants_passed": pilot.get("invariants_passed") is True,
        "pilot_contains_no_private_text": pilot.get("contains_private_text") is False,
        "pilot_count_40": (pilot.get("counts") or {}).get("questions")
        == identity["pilot_questions"],
        "teacher_count_160": len(teachers)
        == identity["pilot_questions"] * identity["candidates_per_question"],
        "teacher_sources_train_only": all(
            (event.get("record") or {}).get("split") == "train" for event in teachers
        ),
        "teacher_benchmarks_balanced": teacher_benchmarks == {"medqa": 80, "medmcqa": 80},
        "screener_count_matches_audit": len(screeners)
        == (pilot.get("counts") or {}).get("screener_events"),
        "validator_count_matches_audit": len(validators)
        == (pilot.get("counts") or {}).get("validator_events"),
        "strict_sft_count_107": len(strict_sft) == identity["strict_sft_trajectories"],
        "strict_sft_train_only": all(
            (record.get("source") or {}).get("split") == "train" for record in strict_sft
        ),
        "strict_sft_benchmarks_known": set(strict_sft_benchmarks) == {"medqa", "medmcqa"},
        "canary_integrity_passed": (canary.get("integrity") or {}).get("passed") is True,
        "canary_candidates_24": len(canary_candidates) == 24,
        "canary_origin_mix_frozen": dict(candidate_origins) == expected_canary_origins,
        "recovery_integrity_passed": (recovery.get("integrity") or {}).get("passed") is True,
        "recovery_machine_quality_passed": (recovery.get("machine_quality") or {}).get("passed") is True,
        "adjudication_quality_passed": (adjudication.get("quality_gate") or {}).get("passed") is True,
        "materialization_quality_passed": (materialization.get("quality_gate") or {}).get("passed") is True,
        "materialized_negative_trajectories_11": materialized.get("negative_trajectories") == 11,
        "enriched_negative_trajectories_12": (materialization.get("enriched_derivative") or {}).get("negative_trajectories")
        == identity["current_negative_trajectories"],
        "enriched_positive_prefixes_749": (materialization.get("enriched_derivative") or {}).get("labels", {}).get("1")
        == identity["current_positive_prefix_records"],
        "enriched_negative_prefixes_47": (materialization.get("enriched_derivative") or {}).get("labels", {}).get("0")
        == identity["current_negative_prefix_records"],
        "materialized_origins_expected": materialized_origins
        == {ORIGIN_CONTROLLED: 7, ORIGIN_MISMATCH: 2, ORIGIN_STUDENT: 2},
    }
    failed = sorted(key for key, passed in gates.items() if not passed)
    if failed:
        raise RuntimeError(f"real-log preflight failed aggregate gates: {failed}")

    pilot_cost = _usage_total([*teachers, *screeners, *validators], "cost_cny")
    audited_pilot_cost = float((pilot.get("cost") or {}).get("total_cny", -1))
    if not math.isclose(pilot_cost, audited_pilot_cost, abs_tol=1e-7):
        raise RuntimeError("pilot raw-event cost differs from aggregate audit")
    canary_cost = _usage_total(canary_validators, "cost_cny")
    recovery_cost = _usage_total(recovery_attempts, "cost_cny")
    audited_negative_cost = float((recovery.get("cost") or {}).get("combined_cny_equivalent", -1))
    if not math.isclose(canary_cost + recovery_cost, audited_negative_cost, abs_tol=1e-7):
        raise RuntimeError("negative raw-event cost differs from recovery audit")
    if _usage_total(teachers, "input_tokens") <= 0 or _usage_total(teachers, "output_tokens") <= 0:
        raise RuntimeError("teacher token usage is missing")
    if _usage_total(validators, "input_tokens") <= 0 or _usage_total(validators, "output_tokens") <= 0:
        raise RuntimeError("pilot validator token usage is missing")
    if _usage_total(canary_validators, "input_tokens") <= 0 or _usage_total(canary_validators, "output_tokens") <= 0:
        raise RuntimeError("negative validator token usage is missing")

    observed_by_origin: dict[str, Any] = {}
    z = float(config["negative_planning"]["yield_lower_bound_z"])
    generation_by_origin = {
        ORIGIN_CONTROLLED: controlled_events,
        ORIGIN_STUDENT: student_events,
    }
    for origin in (ORIGIN_CONTROLLED, ORIGIN_STUDENT, ORIGIN_MISMATCH):
        denominator = int(expected_canary_origins[origin])
        accepted = int(materialized_origins.get(origin, 0))
        row: dict[str, Any] = {
            "validated_candidates": denominator,
            "human_approved_negatives": accepted,
            "point_yield": accepted / denominator,
            "wilson_lower_yield": wilson_lower_bound(accepted, denominator, z),
        }
        if origin in generation_by_origin:
            events = generation_by_origin[origin]
            generated = _accepted_generation(events)
            if generated != denominator:
                raise RuntimeError(f"{origin} local generation count differs from canary")
            row["local_generation_attempts"] = len(events)
            row["local_attempts_per_validated_candidate"] = _safe_ratio(
                len(events), generated, f"{origin} local generation yield"
            )
        observed_by_origin[origin] = row

    controlled = (recovery.get("canonical") or {}).get("controlled") or {}
    observations = {
        "source_gates": {"passed": True, "checks": gates, "failed_checks": []},
        "pilot": {
            "questions": identity["pilot_questions"],
            "teacher_candidate_events": len(teachers),
            "teacher_physical_requests": _physical_attempts(teachers),
            "teacher_rule_passed": int((pilot["counts"])["rule_passed"]),
            "screener_events": len(screeners),
            "validator_events": len(validators),
            "validator_physical_requests": _physical_attempts(validators),
            "canonical_trajectories": int(pilot["counts"]["canonical_trajectories"]),
            "strict_sft_trajectories": len(strict_sft),
            "teacher_input_tokens": int(_usage_total(teachers, "input_tokens")),
            "teacher_output_tokens": int(_usage_total(teachers, "output_tokens")),
            "validator_input_tokens": int(_usage_total(validators, "input_tokens")),
            "validator_output_tokens": int(_usage_total(validators, "output_tokens")),
            "teacher_cost_cny": round(_usage_total(teachers, "cost_cny"), 8),
            "validator_cost_cny": round(_usage_total(validators, "cost_cny"), 8),
            "total_api_cost_cny": round(pilot_cost, 8),
        },
        "negative_canary": {
            "validated_candidates": len(canary_candidates),
            "initial_validator_requests": len(canary_validators),
            "recovery_validator_requests": len(recovery_attempts),
            "canonical_strict_contract_valid": int(recovery["canonical"]["strict_contract_valid"]),
            "canonical_strict_process_negatives": int(recovery["canonical"]["strict_process_negatives"]),
            "human_approved_negatives": int(materialized["negative_trajectories"]),
            "by_origin": observed_by_origin,
            "controlled_exact_first_error": int(controlled["exact_intended_first_error"]),
            "controlled_strict_negatives": int(controlled["strict_negatives"]),
            "controlled_exact_match_rate": float(controlled["exact_match_rate"]),
            "initial_input_tokens": int(_usage_total(canary_validators, "input_tokens")),
            "initial_output_tokens": int(_usage_total(canary_validators, "output_tokens")),
            "recovery_input_tokens": int(_usage_total(recovery_attempts, "input_tokens")),
            "recovery_output_tokens": int(_usage_total(recovery_attempts, "output_tokens")),
            "initial_cost_cny": round(canary_cost, 8),
            "recovery_cost_cny": round(recovery_cost, 8),
            "combined_cost_cny": round(canary_cost + recovery_cost, 8),
            "materialized_positive_prefixes": int(materialized["labels"]["1"]),
            "materialized_negative_prefixes": int(materialized["labels"]["0"]),
        },
    }
    return observations


def _plan_normal_lane(questions: int, observed: dict[str, Any]) -> dict[str, Any]:
    ratio = questions / float(observed["questions"])
    teacher_events = questions * 4
    rule_passed = teacher_events * _safe_ratio(
        observed["teacher_rule_passed"], observed["teacher_candidate_events"], "rule pass"
    )
    validator_events = rule_passed * _safe_ratio(
        observed["validator_events"], observed["teacher_rule_passed"], "validator routing"
    )
    return {
        "new_questions": questions,
        "questions_per_benchmark": {"medqa": questions // 2, "medmcqa": questions // 2},
        "teacher_candidate_events": teacher_events,
        "teacher_physical_requests": math.ceil(
            teacher_events * _safe_ratio(
                observed["teacher_physical_requests"], observed["teacher_candidate_events"], "teacher attempts"
            )
        ),
        "expected_rule_passed": round(rule_passed, 2),
        "local_screener_requests": math.ceil(rule_passed),
        "validator_events": math.ceil(validator_events),
        "validator_physical_requests": math.ceil(
            validator_events * _safe_ratio(
                observed["validator_physical_requests"], observed["validator_events"], "validator attempts"
            )
        ),
        "expected_strict_sft_trajectories": round(
            questions * _safe_ratio(
                observed["strict_sft_trajectories"], observed["questions"], "strict SFT yield"
            ), 2
        ),
        "expected_teacher_cost_cny": round(observed["teacher_cost_cny"] * ratio, 8),
        "expected_validator_cost_cny": round(observed["validator_cost_cny"] * ratio, 8),
        "expected_api_cost_cny": round(observed["total_api_cost_cny"] * ratio, 8),
        "expected_teacher_input_tokens": math.ceil(observed["teacher_input_tokens"] * ratio),
        "expected_teacher_output_tokens": math.ceil(observed["teacher_output_tokens"] * ratio),
        "expected_validator_input_tokens": math.ceil(observed["validator_input_tokens"] * ratio),
        "expected_validator_output_tokens": math.ceil(observed["validator_output_tokens"] * ratio),
    }


def _plan_negative_lane(
    target_total: int, config: dict[str, Any], observed: dict[str, Any]
) -> dict[str, Any]:
    current = int(config["frozen_identity"]["current_negative_trajectories"])
    required = target_total - current
    weights = {
        key: float(value)
        for key, value in config["negative_planning"]["guaranteed_origins"].items()
    }
    accepted_quota = _allocate(required, weights)
    candidate_quota: dict[str, int] = {}
    local_attempts: dict[str, int] = {}
    for origin, accepted in accepted_quota.items():
        row = observed["by_origin"][origin]
        lower = float(row["wilson_lower_yield"])
        if lower <= 0:
            raise RuntimeError(f"no conservative yield for {origin}")
        candidate_quota[origin] = math.ceil(accepted / lower)
        local_attempts[origin] = math.ceil(
            candidate_quota[origin] * float(row["local_attempts_per_validated_candidate"])
        )
    base_requests = sum(candidate_quota.values())
    recovery_rate = _safe_ratio(
        observed["recovery_validator_requests"], observed["initial_validator_requests"], "negative recovery"
    )
    recovery_requests = math.ceil(base_requests * recovery_rate)
    initial_ratio = base_requests / observed["initial_validator_requests"]
    recovery_ratio = recovery_requests / observed["recovery_validator_requests"]
    expected_cost = (
        observed["initial_cost_cny"] * initial_ratio
        + observed["recovery_cost_cny"] * recovery_ratio
    )
    return {
        "target_total_negative_trajectories": target_total,
        "current_negative_trajectories": current,
        "required_new_distinct_negative_trajectories": required,
        "guaranteed_accepted_quota_by_origin": accepted_quota,
        "candidate_validation_quota_by_origin": candidate_quota,
        "opportunistic_answer_mismatch_not_counted": True,
        "local_generation_attempts_by_origin": local_attempts,
        "initial_validator_requests": base_requests,
        "recovery_request_reserve": recovery_requests,
        "total_validator_request_budget": base_requests + recovery_requests,
        "expected_validator_input_tokens": math.ceil(
            observed["initial_input_tokens"] * initial_ratio
            + observed["recovery_input_tokens"] * recovery_ratio
        ),
        "expected_validator_output_tokens": math.ceil(
            observed["initial_output_tokens"] * initial_ratio
            + observed["recovery_output_tokens"] * recovery_ratio
        ),
        "expected_api_cost_cny": round(expected_cost, 8),
    }


def _apply_budget(
    normal: dict[str, Any], negative: dict[str, Any], config: dict[str, Any],
    observed: dict[str, Any],
) -> dict[str, Any]:
    policy = config["budget_policy"]
    raw_api = float(normal["expected_api_cost_cny"]) + float(negative["expected_api_cost_cny"])
    unit = int(policy["round_cny_up_to"])
    screener_upper_hours = (
        normal["local_screener_requests"]
        / observed["pilot"]["screener_events"]
        * float(policy["pilot_local_gpu_hours_cap"])
    )
    negative_local_calls = sum(negative["local_generation_attempts_by_origin"].values())
    observed_local_calls = sum(
        observed["negative_canary"]["by_origin"][origin]["local_generation_attempts"]
        for origin in (ORIGIN_CONTROLLED, ORIGIN_STUDENT)
    )
    negative_upper_hours = (
        negative_local_calls / observed_local_calls
        * float(policy["negative_canary_local_gpu_hours_cap"])
    )
    gpu_hours = screener_upper_hours + negative_upper_hours
    gpu_rate = float(policy["observed_local_gpu_hourly_rate_cny"])
    dashscope_raw = float(normal["expected_teacher_cost_cny"])
    openrouter_raw = float(normal["expected_validator_cost_cny"]) + float(
        negative["expected_api_cost_cny"]
    )

    def provider_budget(raw: float) -> dict[str, float]:
        return {
            "raw_expected_cny": round(raw, 8),
            "estimated_cny_with_safety": _round_up(
                raw * float(policy["estimate_safety_factor"]), unit
            ),
            "program_stop_line_cny": _round_up(
                raw * float(policy["stop_line_factor"]), unit
            ),
            "absolute_hard_cap_cny": _round_up(
                raw * float(policy["hard_cap_factor"]), unit
            ),
        }

    return {
        "raw_expected_api_cny": round(raw_api, 8),
        "estimated_api_cny_with_safety": _round_up(
            raw_api * float(policy["estimate_safety_factor"]), unit
        ),
        "program_stop_line_cny": _round_up(raw_api * float(policy["stop_line_factor"]), unit),
        "absolute_hard_cap_cny": _round_up(raw_api * float(policy["hard_cap_factor"]), unit),
        "provider_budget": {
            "dashscope_teacher": provider_budget(dashscope_raw),
            "openrouter_validator": provider_budget(openrouter_raw),
        },
        "provider_physical_requests": {
            "dashscope_teacher": normal["teacher_physical_requests"],
            "openrouter_validator": normal["validator_physical_requests"]
            + negative["total_validator_request_budget"],
        },
        "conservative_local_gpu_hours": round(gpu_hours, 2),
        "conservative_local_gpu_cost_cny_at_observed_rate": round(gpu_hours * gpu_rate, 2),
        "observed_gpu_hourly_rate_cny": gpu_rate,
        "gpu_rate_must_be_reconfirmed": True,
        "gpu_hour_method": "linear scaling from the two approved one-hour caps; upper-bound planning, not measured wall time",
    }


def build_plan(config: dict[str, Any], observations: dict[str, Any]) -> dict[str, Any]:
    scenarios: list[dict[str, Any]] = []
    for value in config["scenarios"]:
        normal = _plan_normal_lane(int(value["new_questions"]), observations["pilot"])
        negative = _plan_negative_lane(
            int(value["target_total_negative_trajectories"]),
            config,
            observations["negative_canary"],
        )
        current_positive = int(config["frozen_identity"]["current_positive_prefix_records"])
        current_negative = int(config["frozen_identity"]["current_negative_prefix_records"])
        required_negative_trajectories = negative["required_new_distinct_negative_trajectories"]
        materialized_positive_per_trajectory = (
            observations["negative_canary"]["materialized_positive_prefixes"]
            / observations["negative_canary"]["human_approved_negatives"]
        )
        materialized_negative_per_trajectory = (
            observations["negative_canary"]["materialized_negative_prefixes"]
            / observations["negative_canary"]["human_approved_negatives"]
        )
        projected_negative_prefixes = current_negative + (
            required_negative_trajectories * materialized_negative_per_trajectory
        )
        positive_floor = current_positive + (
            required_negative_trajectories * materialized_positive_per_trajectory
        )
        ratio_target = float(
            config["negative_planning"]["prm_positive_to_negative_ratio_target"]
        )
        target_positive_prefixes = projected_negative_prefixes * ratio_target
        positive_pool_per_sft = (
            config["frozen_identity"]["strict_positive_prefix_records"]
            / config["frozen_identity"]["strict_prm_trajectories"]
        )
        new_positive_pool = (
            normal["expected_strict_sft_trajectories"] * positive_pool_per_sft
        )
        selected_new_positive = min(
            new_positive_pool,
            float(math.ceil(max(0.0, target_positive_prefixes - positive_floor))),
        )
        final_positive = positive_floor + selected_new_positive
        prm_balance = {
            "policy": "retain all Verified-CoT for SFT; stratify only PRM positive prefixes",
            "positive_to_negative_ratio_target": ratio_target,
            "projected_negative_prefixes": math.ceil(projected_negative_prefixes),
            "positive_prefix_floor_from_existing_and_negative_trajectories": math.ceil(positive_floor),
            "available_new_positive_prefix_pool": math.floor(new_positive_pool),
            "selected_new_positive_prefix_quota": int(selected_new_positive),
            "approximate_selected_new_positive_trajectories": math.floor(
                selected_new_positive / positive_pool_per_sft
            ),
            "projected_final_positive_prefixes": math.floor(final_positive),
            "projected_positive_to_negative_ratio": round(
                final_positive / projected_negative_prefixes, 4
            ),
        }
        split_policy = config["dataset_split_policy"]
        sft_total_rounded = round(normal["expected_strict_sft_trajectories"])
        dataset_split_plan = {
            "grouping_unit": split_policy["grouping_unit"],
            "forbid_group_cross_split": True,
            "sft_expected_trajectory_quota": _allocate(
                sft_total_rounded,
                {
                    key: float(weight)
                    for key, weight in split_policy["sft_trajectory_split"].items()
                },
            ),
            "prm_negative_trajectory_quota": _allocate(
                int(value["target_total_negative_trajectories"]),
                {
                    key: float(weight)
                    for key, weight in split_policy["prm_negative_trajectory_split"].items()
                },
            ),
            "current_12_negatives_default_to_audit_holdout": True,
            "evaluation_benchmark_records_forbidden": True,
        }
        scenarios.append(
            {
                "name": value["name"],
                "normal_verified_cot_sft": normal,
                "prm_negative": negative,
                "prm_class_balance": prm_balance,
                "dataset_split_plan": dataset_split_plan,
                "budget": _apply_budget(normal, negative, config, observations),
            }
        )
    canary_normal = _plan_normal_lane(
        int(config["paid_canary_proposal"]["new_questions"]), observations["pilot"]
    )
    canary_mix = config["paid_canary_proposal"]["negative_candidates"]
    canary_negative_requests = sum(int(value) for value in canary_mix.values())
    negative_observed = observations["negative_canary"]
    canary_recovery = math.ceil(
        canary_negative_requests
        * _safe_ratio(
            negative_observed["recovery_validator_requests"],
            negative_observed["initial_validator_requests"],
            "canary recovery",
        )
    )
    initial_ratio = canary_negative_requests / negative_observed["initial_validator_requests"]
    recovery_ratio = canary_recovery / negative_observed["recovery_validator_requests"]
    canary_negative_cost = (
        negative_observed["initial_cost_cny"] * initial_ratio
        + negative_observed["recovery_cost_cny"] * recovery_ratio
    )
    canary_local_attempts = {
        origin: math.ceil(
            int(count)
            * float(negative_observed["by_origin"][origin]["local_attempts_per_validated_candidate"])
        )
        for origin, count in canary_mix.items()
    }
    canary_negative_plan = {
        "expected_api_cost_cny": round(canary_negative_cost, 8),
        "total_validator_request_budget": canary_negative_requests + canary_recovery,
        "local_generation_attempts_by_origin": canary_local_attempts,
    }
    canary_budget = _apply_budget(
        canary_normal, canary_negative_plan, config, observations
    )
    recommended = next(
        scenario for scenario in scenarios if scenario["name"] == config["recommended_scenario"]
    )
    return {
        "schema_version": "medtrace.step2-scaleup-budget-report.v1",
        "contains_private_text_or_ids": False,
        "zero_call_preflight": {
            "status": "passed",
            "source_gates": observations["source_gates"],
            "network_or_model_calls_made": 0,
            "gpu_inference_calls_made": 0,
            "training_records_written": 0,
        },
        "observed": observations,
        "scenarios": scenarios,
        "recommendation": {
            "scenario": recommended["name"],
            "new_questions": recommended["normal_verified_cot_sft"]["new_questions"],
            "questions_per_benchmark": recommended["normal_verified_cot_sft"]["questions_per_benchmark"],
            "teacher_candidates_per_question": config["frozen_identity"]["candidates_per_question"],
            "target_total_negative_trajectories": recommended["prm_negative"]["target_total_negative_trajectories"],
            "candidate_validation_quota_by_origin": recommended["prm_negative"]["candidate_validation_quota_by_origin"],
            "prm_selected_new_positive_prefix_quota": recommended["prm_class_balance"]["selected_new_positive_prefix_quota"],
            "projected_prm_positive_to_negative_ratio": recommended["prm_class_balance"]["projected_positive_to_negative_ratio"],
            "sft_expected_trajectory_split": recommended["dataset_split_plan"]["sft_expected_trajectory_quota"],
            "prm_negative_trajectory_split": recommended["dataset_split_plan"]["prm_negative_trajectory_quota"],
            "program_stop_line_cny": recommended["budget"]["program_stop_line_cny"],
            "absolute_hard_cap_cny": recommended["budget"]["absolute_hard_cap_cny"],
            "status": "planning_frozen_but_not_authorized_for_paid_execution",
        },
        "paid_canary_proposal": {
            "authorized": False,
            "new_questions": canary_normal["new_questions"],
            "questions_per_benchmark": canary_normal["questions_per_benchmark"],
            "teacher_candidate_events": canary_normal["teacher_candidate_events"],
            "negative_candidates": canary_mix,
            "negative_recovery_request_reserve": canary_recovery,
            "local_generation_attempts": canary_local_attempts,
            "budget": canary_budget,
            "quality_gates": config["paid_canary_proposal"]["quality_gates"],
        },
        "authorization": config["authorization"],
    }


def render_markdown(report: dict[str, Any]) -> str:
    observed = report["observed"]
    pilot = observed["pilot"]
    negative = observed["negative_canary"]
    lines = [
        "# Step 2 scale-up budget precheck",
        "",
        "Aggregate-only report; no question, ID, prompt, response, or trajectory text.",
        "",
        "## Real-log observations",
        "",
        f"- Pilot funnel: {pilot['questions']} questions -> "
        f"{pilot['teacher_candidate_events']} teacher candidates -> "
        f"{pilot['teacher_rule_passed']} rule-pass -> "
        f"{pilot['canonical_trajectories']} canonical -> "
        f"{pilot['strict_sft_trajectories']} strict SFT.",
        f"- Pilot API cost: CNY {pilot['total_api_cost_cny']:.8f}.",
        f"- Negative canary: {negative['validated_candidates']} candidates + "
        f"{negative['recovery_validator_requests']} recovery requests -> "
        f"{negative['human_approved_negatives']} adjudicated/materialized negatives.",
        f"- Negative canary/recovery API cost: CNY {negative['combined_cost_cny']:.8f}.",
        "",
        "## Scale-up scenarios",
        "",
        "| Scenario | New questions | Expected strict SFT | Negative target | PRM +:- ratio | Negative validator requests incl. recovery | Expected API | Stop line | Hard cap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scenario in report["scenarios"]:
        normal = scenario["normal_verified_cot_sft"]
        prm = scenario["prm_negative"]
        budget = scenario["budget"]
        balance = scenario["prm_class_balance"]
        lines.append(
            f"| {scenario['name']} | {normal['new_questions']} | "
            f"{normal['expected_strict_sft_trajectories']:.2f} | "
            f"{prm['target_total_negative_trajectories']} | "
            f"{balance['projected_positive_to_negative_ratio']:.2f}:1 | "
            f"{prm['total_validator_request_budget']} | "
            f"CNY {budget['raw_expected_api_cny']:.2f} | "
            f"CNY {budget['program_stop_line_cny']:.0f} | "
            f"CNY {budget['absolute_hard_cap_cny']:.0f} |"
        )
    recommendation = report["recommendation"]
    canary = report["paid_canary_proposal"]
    recommended_scenario = next(
        scenario for scenario in report["scenarios"]
        if scenario["name"] == recommendation["scenario"]
    )
    provider_budget = recommended_scenario["budget"]["provider_budget"]
    provider_requests = recommended_scenario["budget"]["provider_physical_requests"]
    lines.extend(
        [
            "",
            "## Frozen planning recommendation",
            "",
            f"- Scenario: `{recommendation['scenario']}`.",
            f"- New train questions: {recommendation['new_questions']} "
            f"({recommendation['questions_per_benchmark']['medqa']} MedQA + "
            f"{recommendation['questions_per_benchmark']['medmcqa']} MedMCQA).",
            f"- Teacher candidates per question: {recommendation['teacher_candidates_per_question']}.",
            f"- Total negative-trajectory target: {recommendation['target_total_negative_trajectories']}.",
            f"- Negative validation quotas: `"
            f"{json.dumps(recommendation['candidate_validation_quota_by_origin'], sort_keys=True)}`.",
            f"- PRM selected new positive-prefix quota: "
            f"{recommendation['prm_selected_new_positive_prefix_quota']} "
            f"(projected ratio {recommendation['projected_prm_positive_to_negative_ratio']}:1).",
            f"- Expected SFT train/validation split: `"
            f"{json.dumps(recommendation['sft_expected_trajectory_split'], sort_keys=True)}`.",
            f"- PRM negative train/validation/test split: `"
            f"{json.dumps(recommendation['prm_negative_trajectory_split'], sort_keys=True)}`.",
            "- Split unit is source question; prefixes from one question cannot cross splits.",
            f"- API stop line / hard cap: CNY {recommendation['program_stop_line_cny']:.0f} / "
            f"CNY {recommendation['absolute_hard_cap_cny']:.0f}.",
            f"- DashScope requests / stop / cap: {provider_requests['dashscope_teacher']} / "
            f"CNY {provider_budget['dashscope_teacher']['program_stop_line_cny']:.0f} / "
            f"CNY {provider_budget['dashscope_teacher']['absolute_hard_cap_cny']:.0f}.",
            f"- OpenRouter requests / stop / cap: {provider_requests['openrouter_validator']} / "
            f"CNY {provider_budget['openrouter_validator']['program_stop_line_cny']:.0f} / "
            f"CNY {provider_budget['openrouter_validator']['absolute_hard_cap_cny']:.0f}.",
            f"- Conservative local GPU upper bound / cost: "
            f"{recommended_scenario['budget']['conservative_local_gpu_hours']:.2f} hours / "
            f"CNY {recommended_scenario['budget']['conservative_local_gpu_cost_cny_at_observed_rate']:.2f}; "
            "hourly rate must be reconfirmed.",
            "- This recommendation is frozen for planning only; paid execution is not authorized.",
            "",
            "## Separate paid-canary proposal",
            "",
            f"- 100 new questions, {canary['teacher_candidate_events']} teacher candidates.",
            f"- Negative candidates: `{json.dumps(canary['negative_candidates'], sort_keys=True)}`.",
            f"- API stop line / hard cap: CNY {canary['budget']['program_stop_line_cny']:.0f} / "
            f"CNY {canary['budget']['absolute_hard_cap_cny']:.0f}.",
            "- Paid canary authorized: false.",
            "- Training/full-scale generation authorized: false.",
            "",
        ]
    )
    return "\n".join(lines)


def preflight(config: dict[str, Any], *, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    observations = collect_observations(config, repo_root=repo_root)
    return {
        "schema_version": "medtrace.step2-scaleup-zero-call-preflight.v1",
        "status": "passed",
        "contains_private_text_or_ids": False,
        "source_gates": observations["source_gates"],
        "network_or_model_calls_made": 0,
        "gpu_inference_calls_made": 0,
        "training_records_written": 0,
    }


def main() -> None:
    args = parse_args()
    config_path = _repo_path(args.config, REPO_ROOT)
    config = load_config(config_path)
    if args.preflight_only:
        print(json.dumps(preflight(config), indent=2, sort_keys=True))
        return
    observations = collect_observations(config)
    report = build_plan(config, observations)
    markdown = render_markdown(report)
    print(markdown, end="")
    if not args.write_approved_report:
        print("Preview only; no aggregate report was written.")
        return
    runtime = dict(config)
    runtime["write_enabled"] = True
    validate_config(runtime, write=True)
    outputs = config["outputs"]
    _write_json(_repo_path(str(outputs["aggregate_json"]), REPO_ROOT), report)
    _write_text(_repo_path(str(outputs["aggregate_markdown"]), REPO_ROOT), markdown)
    print("Wrote aggregate-only scale-up budget report.")


if __name__ == "__main__":
    main()
