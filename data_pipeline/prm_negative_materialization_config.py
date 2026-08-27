"""Fail-closed validation for approved offline PRM materialization."""

from __future__ import annotations

from typing import Any


SCHEMA_VERSION = "medtrace.cot.prm-negative-materialization.v1"


def validate_prm_negative_materialization_config(
    config: dict[str, Any], *, execute: bool = False
) -> None:
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported PRM negative-materialization schema_version")
    enabled = config.get("execution_enabled")
    if type(enabled) is not bool:
        raise ValueError("PRM materialization execution_enabled must be boolean")
    if execute and not enabled:
        raise ValueError("PRM materialization requires an enabled runtime copy")
    if not execute and enabled:
        raise ValueError("committed PRM materialization must remain disabled")
    if config.get("approval_status") != "approved_2026-08-27":
        raise ValueError("PRM materialization approval is missing")
    expected_paths = {
        "source_adjudication_config": (
            "configs/cot/prm_negative_human_adjudication_v1.yaml"
        ),
        "source_canary_config": "configs/cot/prm_negative_canary_v1.yaml",
        "source_run_dir": "results/cot/pilot_v1_real",
        "strict_source_subdir": "prm_negative_enrichment_v1/strict_source_v1",
        "source_canary_subdir": "prm_negative_enrichment_v1/canary_v1",
        "output_subdir": (
            "prm_negative_enrichment_v1/prm_negative_materialization_v1"
        ),
    }
    for field, value in expected_paths.items():
        if config.get(field) != value:
            raise ValueError(f"unexpected PRM materialization path: {field}")
    if config.get("source_expectations") != {
        "approved_negative_trajectories": 11,
        "strict_prm_records": 721,
        "strict_positive_prefix_records": 717,
        "strict_negative_prefix_records": 4,
    }:
        raise ValueError("PRM materialization source expectations changed")
    if config.get("output_files") != {
        "candidate_prefix_records": "process_train_negative_candidates_v1.jsonl",
        "enriched_process_train": "process_train_negative_enriched_canary_v1.jsonl",
        "manifest": "manifest.json",
        "aggregate_audit_json": "audit.json",
        "aggregate_audit_markdown": "audit.md",
    }:
        raise ValueError("PRM materialization output filenames changed")
    if config.get("materialization_contract") != {
        "record_keys": [
            "trajectory_id", "step_index", "prefix", "label", "error_codes",
            "source",
        ],
        "label_semantics": "prefix_correctness",
        "prefixes_before_first_error": 1,
        "prefixes_from_first_error": 0,
        "strict_integer_labels": True,
        "require_contiguous_step_indices": True,
        "require_distinct_candidate_trajectories": True,
        "reject_overlap_with_strict_source": True,
        "preserve_strict_source": True,
        "preserve_sft_source": True,
    }:
        raise ValueError("PRM materialization record contract changed")
    if config.get("quality_gates") != {
        "exact_approved_trajectories": 11,
        "minimum_negative_prefix_records": 11,
        "minimum_negative_trajectories": 11,
        "require_both_benchmarks": True,
        "minimum_negative_origins": 2,
        "require_zero_duplicate_records": True,
        "require_source_artifacts_unchanged": True,
    }:
        raise ValueError("PRM materialization quality gates changed")
    if config.get("authorization") != {
        "offline_only": True,
        "model_or_api_calls_authorized": False,
        "gpu_inference_authorized": False,
        "derivative_prm_records_authorized": True,
        "training_use_authorized": False,
        "sft_changes_authorized": False,
        "full_scale_generation_authorized": False,
    }:
        raise ValueError("PRM materialization authorization boundary changed")
