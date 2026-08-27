"""Fail-closed validation for post-lock PRM human adjudication."""

from __future__ import annotations

from typing import Any


SCHEMA_VERSION = "medtrace.cot.prm-negative-human-adjudication.v1"


def validate_prm_negative_human_adjudication_config(
    config: dict[str, Any], *, write: bool = False
) -> None:
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported PRM human-adjudication schema_version")
    enabled = config.get("write_enabled")
    if type(enabled) is not bool:
        raise ValueError("human-adjudication write_enabled must be boolean")
    if write and not enabled:
        raise ValueError("human-adjudication write requires an enabled runtime copy")
    if not write and enabled:
        raise ValueError(
            "committed human-adjudication config must keep writes disabled"
        )
    if config.get("approval_status") != "approved_2026-08-27":
        raise ValueError("human-adjudication phase approval is missing")
    if config.get("source_human_review_config") != (
        "configs/cot/prm_negative_human_review_v2.yaml"
    ):
        raise ValueError("unexpected source human-review config")
    if config.get("private_files") != {
        "adjudication": "human_review_adjudication_v1.jsonl",
        "adjudication_lock": "human_review_adjudication_v1.lock.json",
        "aggregate_audit_json": "human_review_adjudication_audit_v1.json",
        "aggregate_audit_markdown": "human_review_adjudication_audit_v1.md",
        "approved_negative_candidates": (
            "human_adjudicated_negative_candidates_v1.jsonl"
        ),
    }:
        raise ValueError("human-adjudication private filenames changed")
    if config.get("source_gate") != {
        "total_annotations": 24,
        "human_problem_ok": 19,
        "raw_trajectory_correct": 19,
        "raw_trajectory_total": 19,
        "raw_first_error_correct": 9,
        "raw_first_error_total": 12,
        "raw_conservative_candidates": 9,
        "required_failed_checks": [
            "exact_first_error_accuracy_at_least_80_percent"
        ],
        "total_disagreements": 3,
    }:
        raise ValueError("human-adjudication source gate changed")
    if config.get("adjudication_contract") != {
        "total_cases": 3,
        "require_unblinded_attestation": True,
        "require_original_review_preserved": True,
        "decision_protocol": "earliest_unambiguously_incorrect_step",
        "decision_source_values": ["human", "validator"],
        "adjudicated_step_must_be_original_or_validator": True,
        "require_rationale": True,
    }:
        raise ValueError("human-adjudication contract changed")
    if config.get("scoring") != {
        "minimum_adjudicated_exact_first_error_accuracy": 0.80,
        "minimum_conservative_negative_candidates": 8,
        "raw_blind_metrics_must_remain_reported": True,
    }:
        raise ValueError("human-adjudication scoring protocol changed")
    if config.get("candidate_policy") != {
        "require_human_problem_status_ok": True,
        "require_human_trajectory_negative": True,
        "require_human_error_type_process": True,
        "require_validator_strict_process_negative": True,
        "require_adjudicated_exact_first_error_agreement": True,
        "write_training_records": False,
    }:
        raise ValueError("human-adjudication candidate policy changed")
    if config.get("authorization") != {
        "offline_only": True,
        "model_or_api_calls_authorized": False,
        "training_merge_authorized": False,
        "full_scale_generation_authorized": False,
    }:
        raise ValueError("human-adjudication authorization boundary changed")
