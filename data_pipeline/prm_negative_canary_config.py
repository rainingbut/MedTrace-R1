"""Fail-closed validation for the frozen PRM negative canary configuration."""

from __future__ import annotations

from typing import Any


SCHEMA_VERSION = "medtrace.cot.prm-negative-canary.v1"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
VALIDATOR_MODEL = "deepseek/deepseek-v4-pro"


def validate_prm_negative_canary_config(
    config: dict[str, Any], *, execute: bool = False
) -> None:
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported PRM negative canary schema_version")
    enabled = config.get("execution_enabled")
    if type(enabled) is not bool:
        raise ValueError("canary execution_enabled must be boolean")
    if execute and not enabled:
        raise ValueError("canary execution requires an enabled runtime copy")
    if not execute and enabled:
        raise ValueError("committed canary config must keep execution disabled")
    if config.get("source_run_dir") != "results/cot/pilot_v1_real":
        raise ValueError("canary source run changed")
    if config.get("strict_source_subdir") != (
        "prm_negative_enrichment_v1/strict_source_v1"
    ):
        raise ValueError("strict source output changed")
    if config.get("output_subdir") != "prm_negative_enrichment_v1/canary_v1":
        raise ValueError("canary output must remain isolated")

    identity = config.get("source_identity") or {}
    expected_identity = {
        "config_sha256": "168243919b2912ba76b1e70d4228d011cd18b2a2d3583ed9f026b712677ca36f",
        "questions_sha256": "eb7f4c2ba71d4651929af89558bfa64c295dcdab902e218483f931c040df9f76",
        "generation_git_commit": "461a95cb28837f8072b6fc3d927f57fd8fb80f87",
        "questions": 40,
        "teacher_events": 160,
        "canonical_trajectories": 109,
        "prm_records": 727,
    }
    if identity != expected_identity:
        raise ValueError("canary source identity changed")
    if config.get("strict_source_expected") != {
        "canonical_trajectories": 108,
        "sft_records": 107,
        "prm_records": 721,
    }:
        raise ValueError("strict source expectations changed")

    selection = config.get("selection") or {}
    if int(selection.get("seed", 0)) != 20260826:
        raise ValueError("canary seed changed")
    if int(selection.get("total", 0)) != 24:
        raise ValueError("canary must contain 24 candidates")
    if selection.get("by_origin") != {
        "existing_teacher_answer_mismatch": 8,
        "local_student": 8,
        "controlled_single_error": 8,
    }:
        raise ValueError("canary origin quotas changed")
    if selection.get("each_origin_by_benchmark") != {
        "medqa": 4,
        "medmcqa": 4,
    }:
        raise ValueError("canary benchmark quotas changed")
    if selection.get("prefer_distinct_questions") is not True:
        raise ValueError("canary must prefer distinct questions")

    local = config.get("local_model") or {}
    required_local = {
        "provider": "local_vllm",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "expected_vllm_version": "0.24.0",
    }
    for field, expected in required_local.items():
        if local.get(field) != expected:
            raise ValueError(f"unexpected local model setting: {field}")
    if local.get("api_key_env") != "MEDTRACE_API_KEY":
        raise ValueError("local model must use MEDTRACE_API_KEY")
    if local.get("base_url") != "http://127.0.0.1:8000/v1":
        raise ValueError("local model endpoint changed")
    if local.get("runtime_manifest") != (
        "results/runtime/cot_screener_runtime_manifest.json"
    ):
        raise ValueError("local model endpoint/runtime is incomplete")
    if float(local.get("temperature", -1)) != 0.7:
        raise ValueError("local model temperature changed")
    if float(local.get("top_p", -1)) != 0.95:
        raise ValueError("local model top_p changed")
    if int(local.get("max_output_tokens", 0)) != 1536:
        raise ValueError("local model output limit changed")
    if int(local.get("max_retries", -1)) != 1:
        raise ValueError("local model retry count changed")
    if int(local.get("max_attempted_questions_per_origin_and_benchmark", 0)) != 12:
        raise ValueError("local generation attempt cap changed")

    if (config.get("student") or {}) != {
        "prompt_version": "prm_student_v1",
        "expose_gold_answer": False,
    }:
        raise ValueError("student settings changed")
    if (config.get("controlled_mutator") or {}) != {
        "prompt_version": "prm_controlled_mutator_v1",
        "target_position_cycle": ["early", "middle", "late", "middle"],
        "preserve_non_target_steps_exactly": True,
        "preserve_answer_exactly": True,
    }:
        raise ValueError("controlled mutator settings changed")

    validator = config.get("validator") or {}
    expected_validator = {
        "provider": "openrouter",
        "model_id": VALIDATOR_MODEL,
        "provider_version": "OpenRouter-DeepSeek-V4-Pro",
        "api_key_env": "OPENROUTER_API_KEY",
        "prompt_version": "validator_v2",
        "reasoning_effort": "high",
        "require_zero_data_retention": True,
        "allow_provider_fallbacks": True,
        "response_format": "json_schema_strict",
        "max_retries": 0,
    }
    for field, expected in expected_validator.items():
        if validator.get(field) != expected:
            raise ValueError(f"unexpected validator setting: {field}")
    if validator.get("base_url") != "https://openrouter.ai/api/v1":
        raise ValueError("validator endpoint changed")
    if int(validator.get("max_output_tokens", 0)) != 8192:
        raise ValueError("validator endpoint/token limit changed")
    if float(validator.get("temperature", -1)) != 0:
        raise ValueError("validator temperature changed")

    budget = config.get("budget") or {}
    if float(budget.get("api_hard_cap_cny_equivalent", 0)) != 20:
        raise ValueError("canary hard cap must remain CNY 20")
    if float(budget.get("stop_before_limit_fraction", 0)) != 0.90:
        raise ValueError("canary stop fraction must remain 0.90")
    if float(budget.get("local_gpu_hours_cap", 0)) != 1:
        raise ValueError("canary local GPU cap must remain one hour")
    if float(budget.get("usd_to_cny", 0)) != 7.20:
        raise ValueError("canary USD/CNY conversion changed")
    if float(budget.get("validator_usd_per_million_input_tokens", 0)) != 2.10:
        raise ValueError("validator input price ceiling changed")
    if float(budget.get("validator_usd_per_million_output_tokens", 0)) != 4.40:
        raise ValueError("validator output price ceiling changed")

    gates = config.get("quality_gates") or {}
    if float(gates.get("transport_and_contract_success_rate", 0)) != 1.0:
        raise ValueError("canary must require perfect contract success")
    if int(gates.get("minimum_distinct_strict_negative_trajectories", 0)) != 8:
        raise ValueError("canary strict-negative minimum changed")
    if gates.get("require_both_benchmarks") is not True:
        raise ValueError("canary must cover both benchmarks")
    if int(gates.get("minimum_negative_origins", 0)) != 2:
        raise ValueError("canary must cover at least two negative origins")
    if float(gates.get("human_trajectory_label_accuracy", 0)) != 0.90:
        raise ValueError("human trajectory-label gate changed")
    if float(gates.get("human_exact_first_error_accuracy", 0)) != 0.80:
        raise ValueError("human first-error gate changed")
