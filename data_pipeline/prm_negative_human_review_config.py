"""Fail-closed validation for the approved PRM human-review protocol."""

from __future__ import annotations

from typing import Any


SCHEMA_VERSION = "medtrace.cot.prm-negative-human-review.v2"


def validate_prm_negative_human_review_config(
    config: dict[str, Any], *, write: bool = False
) -> None:
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported PRM human-review schema_version")
    enabled = config.get("write_enabled")
    if type(enabled) is not bool:
        raise ValueError("human-review write_enabled must be boolean")
    if write and not enabled:
        raise ValueError("human-review write requires an enabled runtime copy")
    if not write and enabled:
        raise ValueError("committed human-review config must keep writes disabled")
    if config.get("approval_status") != "approved_2026-08-27":
        raise ValueError("human-review phase approval is missing")

    expected_paths = {
        "source_recovery_config": (
            "configs/cot/prm_negative_validator_recovery_v1.yaml"
        ),
        "source_run_dir": "results/cot/pilot_v1_real",
        "source_canary_subdir": "prm_negative_enrichment_v1/canary_v1",
        "source_recovery_subdir": (
            "prm_negative_enrichment_v1/canary_v1/validator_recovery_v1"
        ),
    }
    for field, expected in expected_paths.items():
        if config.get(field) != expected:
            raise ValueError(f"unexpected human-review path: {field}")
    if config.get("private_files") != {
        "blind_review": "human_review_blind.md",
        "annotations": "human_review_annotations_v2.jsonl",
        "annotation_lock": "human_review_annotations_v2.lock.json",
        "recovered_key": "human_review_key_recovered.md",
        "approved_negative_candidates": (
            "human_approved_negative_candidates_v2.jsonl"
        ),
        "aggregate_audit_json": "human_review_quality_audit_v2.json",
        "aggregate_audit_markdown": "human_review_quality_audit_v2.md",
    }:
        raise ValueError("human-review private filenames changed")
    if config.get("annotation_contract") != {
        "total_cases": 24,
        "problem_status_values": ["ok", "ambiguous", "bad_gold"],
        "trajectory_label_values": [0, 1],
        "negative_error_type_values": ["process", "answer_only"],
        "require_blind_attestation": True,
        "require_reviewer_role": True,
        "require_completed_at_utc": True,
        "non_ok_label_must_be_null": True,
        "non_ok_error_type_must_be_null": True,
        "non_ok_first_error_must_be_null": True,
        "positive_error_type_must_be_null": True,
        "positive_first_error_must_be_null": True,
        "process_negative_first_error_must_be_in_range": True,
        "answer_only_negative_first_error_must_be_null": True,
    }:
        raise ValueError("human-review annotation contract changed")
    if config.get("scoring") != {
        "trajectory_accuracy_denominator": "human_problem_status_ok",
        "first_error_accuracy_denominator": (
            "human_problem_status_ok_and_process_negative"
        ),
        "minimum_trajectory_label_accuracy": 0.90,
        "minimum_exact_first_error_accuracy": 0.80,
    }:
        raise ValueError("human-review scoring protocol changed")
    if config.get("candidate_policy") != {
        "require_human_problem_status_ok": True,
        "require_human_trajectory_negative": True,
        "require_human_error_type_process": True,
        "require_validator_strict_process_negative": True,
        "require_exact_first_error_agreement": True,
        "write_training_records": False,
    }:
        raise ValueError("human-review candidate policy changed")
    if config.get("authorization") != {
        "offline_only": True,
        "model_or_api_calls_authorized": False,
        "training_merge_authorized": False,
        "full_scale_generation_authorized": False,
    }:
        raise ValueError("human-review authorization boundary changed")
