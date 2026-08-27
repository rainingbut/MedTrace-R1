"""Fail-closed validation for the approved three-case 429 recovery."""

from __future__ import annotations

from typing import Any


SCHEMA_VERSION = "medtrace.cot.prm-negative-validator-recovery.v1"
VALIDATOR_MODEL = "deepseek/deepseek-v4-pro"


def validate_prm_negative_recovery_config(
    config: dict[str, Any], *, execute: bool = False
) -> None:
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported PRM negative recovery schema_version")
    enabled = config.get("execution_enabled")
    if type(enabled) is not bool:
        raise ValueError("recovery execution_enabled must be boolean")
    if execute and not enabled:
        raise ValueError("recovery execution requires an enabled runtime copy")
    if not execute and enabled:
        raise ValueError("committed recovery config must keep execution disabled")
    if config.get("approval_status") != "approved_2026-08-27":
        raise ValueError("three-case recovery approval is missing")

    expected_paths = {
        "source_canary_config": "configs/cot/prm_negative_canary_v1.yaml",
        "source_run_dir": "results/cot/pilot_v1_real",
        "source_canary_subdir": "prm_negative_enrichment_v1/canary_v1",
        "output_subdir": (
            "prm_negative_enrichment_v1/canary_v1/validator_recovery_v1"
        ),
    }
    for field, expected in expected_paths.items():
        if config.get(field) != expected:
            raise ValueError(f"unexpected PRM recovery path: {field}")

    if config.get("source_expectations") != {
        "candidates": 24,
        "validator_events": 24,
        "strict_contract_valid": 21,
        "unavailable": 3,
        "unavailable_event_status": "api_or_parse_error",
        "unavailable_error_category": "http_429",
        "unavailable_content_present": False,
    }:
        raise ValueError("PRM recovery source expectations changed")
    if config.get("selection") != {
        "total": 3,
        "only_source_unavailable": True,
        "only_http_429": True,
    }:
        raise ValueError("PRM recovery selection changed")

    validator = config.get("validator") or {}
    expected_validator = {
        "provider": "openrouter",
        "model_id": VALIDATOR_MODEL,
        "provider_version": "OpenRouter-DeepSeek-V4-Pro",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "prompt_version": "validator_v2",
        "reasoning_effort": "high",
        "require_zero_data_retention": True,
        "allow_provider_fallbacks": True,
        "response_format": "json_schema_strict",
        "temperature": 0,
        "max_output_tokens": 8192,
        "max_retries": 0,
        "timeout_seconds": 600,
    }
    if validator != expected_validator:
        raise ValueError("PRM recovery validator settings changed")

    if config.get("throttle") != {
        "initial_delay_seconds": 60,
        "inter_candidate_delay_seconds": 45,
        "http_429_retry_delay_seconds": 90,
        "max_total_attempts_per_candidate": 2,
    }:
        raise ValueError("PRM recovery throttle settings changed")
    if config.get("budget") != {
        "api_hard_cap_cny_equivalent": 2,
        "stop_before_limit_fraction": 0.90,
        "usd_to_cny": 7.20,
        "validator_usd_per_million_input_tokens": 2.10,
        "validator_usd_per_million_output_tokens": 4.40,
    }:
        raise ValueError("PRM recovery budget settings changed")
    if config.get("quality_gates") != {
        "canonical_transport_and_contract_success_rate": 1.0,
        "minimum_distinct_strict_negative_trajectories": 8,
        "require_both_benchmarks": True,
        "minimum_negative_origins": 2,
        "human_trajectory_label_accuracy": 0.90,
        "human_exact_first_error_accuracy": 0.80,
    }:
        raise ValueError("PRM recovery quality gates changed")
    if config.get("authorization") != {
        "recovery_only": True,
        "training_merge_authorized": False,
        "full_scale_generation_authorized": False,
    }:
        raise ValueError("PRM recovery authorization boundary changed")
