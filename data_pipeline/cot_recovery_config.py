"""Validation for the isolated validator v2 recovery configuration."""

from __future__ import annotations

from typing import Any


def validate_recovery_config(config: dict[str, Any], *, execute: bool = False) -> None:
    if config.get("schema_version") != "medtrace.cot.validator-recovery.v2":
        raise ValueError("unsupported validator recovery schema_version")
    enabled = config.get("execution_enabled")
    if type(enabled) is not bool:
        raise ValueError("recovery execution_enabled must be boolean")
    if execute and not enabled:
        raise ValueError("recovery execution requires an enabled runtime copy")
    if not execute and enabled:
        raise ValueError("committed recovery config must keep execution disabled")
    if config.get("source_run_dir") != "results/cot/pilot_v1_real":
        raise ValueError("recovery source run directory is frozen")
    if config.get("output_subdir") != "validator_recovery_v2":
        raise ValueError("recovery output must remain isolated")
    identity = config.get("source_identity") or {}
    for field, length in (
        ("config_sha256", 64), ("questions_sha256", 64),
        ("generation_git_commit", 40),
    ):
        value = str(identity.get(field) or "")
        if len(value) != length or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"invalid recovery source identity: {field}")
    if int(identity.get("expected_failed_validator_events", 0)) != 25:
        raise ValueError("recovery must bind the 25 failed validator events")
    canary = config.get("canary") or {}
    if int(canary.get("total", 0)) != 6:
        raise ValueError("validator recovery canary must contain six events")
    if canary.get("by_benchmark") != {"medqa": 4, "medmcqa": 2}:
        raise ValueError("validator recovery canary stratification changed")
    if canary.get("selection_method") != "benchmark_stratified_error_detail_round_robin":
        raise ValueError("validator recovery canary selection method changed")
    validator = config.get("validator") or {}
    expected = {
        "provider": "openrouter",
        "model_id": "deepseek/deepseek-v4-pro",
        "api_key_env": "OPENROUTER_API_KEY",
        "prompt_version": "validator_v2",
        "reasoning_effort": "high",
        "response_format": "json_schema_strict",
        "max_output_tokens": 8192,
        "max_retries": 0,
        "temperature": 0,
    }
    for field, value in expected.items():
        if validator.get(field) != value:
            raise ValueError(f"unexpected validator recovery setting: {field}")
    if validator.get("require_zero_data_retention") is not True:
        raise ValueError("validator recovery must require ZDR")
    if validator.get("allow_provider_fallbacks") is not True:
        raise ValueError("validator recovery provider fallback policy changed")
    if not validator.get("base_url") or not validator.get("timeout_seconds"):
        raise ValueError("validator recovery endpoint settings are incomplete")
    budget = config.get("budget") or {}
    if float(budget.get("validator_usd_per_million_input_tokens", 0)) != 2.10:
        raise ValueError("validator recovery input price ceiling changed")
    if float(budget.get("validator_usd_per_million_output_tokens", 0)) != 4.40:
        raise ValueError("validator recovery output price ceiling changed")
    if float(budget.get("api_hard_cap_cny_equivalent", 0)) != 5:
        raise ValueError("validator recovery canary hard cap must remain CNY 5")
    if float(budget.get("stop_before_limit_fraction", 0)) != 0.90:
        raise ValueError("validator recovery stop fraction must remain 0.90")
