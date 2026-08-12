"""Validation for the frozen stage-2 pilot configuration."""

from __future__ import annotations

from typing import Any


EXPECTED_MODELS = {
    "teacher": "qwen3-max-2026-01-23",
    "screener": "Qwen/Qwen2.5-7B-Instruct",
    "validator": "deepseek/deepseek-v4-pro",
}
EXPECTED_PROMPTS = {
    "teacher": "teacher_v1",
    "screener": "screener_v1",
    "validator": "validator_v1",
}


def validate_pilot_config(config: dict[str, Any], real_run: bool = False) -> None:
    if config.get("schema_version") != "medtrace.cot.pilot.v1":
        raise ValueError("unsupported CoT pilot schema_version")
    execution_enabled = config.get("execution_enabled")
    if not isinstance(execution_enabled, bool):
        raise ValueError("execution_enabled must be a boolean")
    if real_run and not execution_enabled:
        raise ValueError("real pilot run requires an explicitly enabled runtime config")
    if not real_run and execution_enabled:
        raise ValueError("committed pilot config must keep execution_enabled: false")

    sources = config.get("sources")
    if not isinstance(sources, dict) or set(sources) != {"medqa", "medmcqa"}:
        raise ValueError("pilot requires exactly the medqa and medmcqa sources")
    for benchmark, source in sources.items():
        if source.get("split") != "train":
            raise ValueError(f"{benchmark} source must use split: train")
        revision = str(source.get("revision", ""))
        if len(revision) != 40 or any(ch not in "0123456789abcdef" for ch in revision):
            raise ValueError(f"{benchmark} revision must be an exact commit SHA")
        source_hash = str(source.get("source_file_sha256", ""))
        if real_run and source_hash.startswith("REPLACE_"):
            raise ValueError(f"{benchmark} source_file_sha256 is not pinned")

    sampling = config.get("sampling") or {}
    per_benchmark = sampling.get("questions_per_benchmark") or {}
    total = int(sampling.get("total_questions", 0))
    if total != 40 or sum(map(int, per_benchmark.values())) != total:
        raise ValueError("pilot must contain 40 questions across both benchmarks")
    if int(sampling.get("candidates_per_question", 0)) != 4:
        raise ValueError("pilot requires four candidates per question")
    if sampling.get("independent_teacher_requests") is not True:
        raise ValueError("teacher candidates must use independent requests")

    for role, expected_model in EXPECTED_MODELS.items():
        role_config = config.get(role) or {}
        if role_config.get("model_id") != expected_model:
            raise ValueError(f"unexpected {role} model")
        if role_config.get("prompt_version") != EXPECTED_PROMPTS[role]:
            raise ValueError(f"unexpected {role} prompt version")
        for connection_field in ("base_url", "api_key_env", "timeout_seconds"):
            if not role_config.get(connection_field):
                raise ValueError(f"{role} is missing {connection_field}")
    if (config.get("teacher") or {}).get("expose_gold_answer") is not False:
        raise ValueError("teacher must remain gold-blind")
    screener = config.get("screener") or {}
    if not screener.get("runtime_manifest") or not screener.get("expected_vllm_version"):
        raise ValueError("screener runtime manifest and vLLM version must be pinned")
    validator = config.get("validator") or {}
    if validator.get("provider") != "openrouter":
        raise ValueError("validator must use the approved OpenRouter provider")
    if validator.get("api_key_env") != "OPENROUTER_API_KEY":
        raise ValueError("validator must use OPENROUTER_API_KEY")
    if validator.get("reasoning_effort") != "high":
        raise ValueError("validator reasoning_effort must remain high")
    if validator.get("require_zero_data_retention") is not True:
        raise ValueError("validator must require zero-data-retention routing")

    budget = config.get("budget") or {}
    if float(budget.get("api_hard_cap_cny_equivalent", 0)) != 10:
        raise ValueError("pilot API hard cap must remain CNY 10 equivalent")
    if not 0 < float(budget.get("stop_before_limit_fraction", 0)) < 1:
        raise ValueError("budget stop fraction must be between zero and one")
    pricing_fields = {
        "usd_to_cny",
        "teacher_cny_per_million_input_tokens",
        "teacher_cny_per_million_output_tokens",
        "validator_usd_per_million_input_tokens",
        "validator_usd_per_million_output_tokens",
    }
    if any(float(budget.get(field, 0)) <= 0 for field in pricing_fields):
        raise ValueError("all pilot pricing assumptions must be positive")
    if float(budget["validator_usd_per_million_input_tokens"]) != 2.10:
        raise ValueError("validator maximum input price must remain USD 2.10/M")
    if float(budget["validator_usd_per_million_output_tokens"]) != 4.40:
        raise ValueError("validator maximum output price must remain USD 4.40/M")

    rule_filter = config.get("rule_filter") or {}
    if int(rule_filter.get("min_steps", 0)) != 3:
        raise ValueError("pilot rule filter requires exactly three minimum steps")
    if int(rule_filter.get("max_steps", 0)) != 8:
        raise ValueError("pilot rule filter requires exactly eight maximum steps")
    if int(rule_filter.get("max_trajectory_characters", 0)) <= 0:
        raise ValueError("max_trajectory_characters must be positive")
    if int(rule_filter.get("max_step_characters", 0)) <= 0:
        raise ValueError("max_step_characters must be positive")
