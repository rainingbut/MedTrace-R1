"""Frozen, non-executing policy for identifying potential PRM negatives."""

from __future__ import annotations

from typing import Any


POLICY_SCHEMA_VERSION = "medtrace.cot.prm-negative-enrichment.v1"

STRUCTURAL_FAILURE_CODES = frozenset(
    {
        "api_error",
        "truncated_output",
        "trajectory_too_long",
        "missing_step_tags",
        "missing_answer_tag",
        "malformed_tag_nesting",
        "step_count_out_of_range",
        "duplicate_step",
        "multiple_answer_tags",
        "invalid_answer_label",
    }
)


def validate_prm_negative_policy(config: dict[str, Any]) -> None:
    """Fail closed if the committed offline policy is relaxed."""

    if config.get("schema_version") != POLICY_SCHEMA_VERSION:
        raise ValueError("unsupported PRM negative enrichment schema_version")
    if config.get("execution_enabled") is not False:
        raise ValueError("committed PRM negative enrichment config must not execute")
    if config.get("source_run_dir") != "results/cot/pilot_v1_real":
        raise ValueError("PRM negative audit source run changed")
    if config.get("output_subdir") != "prm_negative_enrichment_v1":
        raise ValueError("PRM negative audit output must remain isolated")
    if config.get("output_stem") != "prm_negative_opportunity_audit":
        raise ValueError("PRM negative audit output stem changed")
    if config.get("original_artifacts_immutable") is not True:
        raise ValueError("original pilot artifacts must remain immutable")

    natural = config.get("natural_candidate_policy") or {}
    expected_natural = {
        "answer_mismatch_only": "requires_independent_validation",
        "screener_reject": "requires_independent_validation",
        "validator_unavailable": "recovery_only_not_negative_evidence",
        "malformed_or_truncated": "exclude",
    }
    if natural != expected_natural:
        raise ValueError("natural PRM candidate policy changed")

    labels = config.get("label_policy") or {}
    if labels.get("label_semantics") != "prefix_correctness":
        raise ValueError("PRM labels must represent prefix correctness")
    if labels.get("uncertain_first_error") != "human_review":
        raise ValueError("uncertain validator results must require human review")
    if labels.get("ambiguous_or_bad_gold") != "human_review":
        raise ValueError("non-ok problems must require human review")
    if labels.get("wrong_answer_without_reasoning_error") != "exclude_from_prm_negative":
        raise ValueError("answer-only errors must not become PRM step negatives")
    if labels.get("positive_requires") != {
        "trajectory_label": 1,
        "problem_status": "ok",
        "first_error_step": None,
    }:
        raise ValueError("strict PRM positive requirements changed")
    negative = labels.get("negative_requires") or {}
    if negative != {
        "trajectory_label": 0,
        "problem_status": "ok",
        "first_error_step": "integer",
        "first_error_local_verdict": "incorrect",
    }:
        raise ValueError("strict PRM negative requirements changed")
    if labels.get("prefixes_before_first_error") != 1:
        raise ValueError("prefixes before the first error must remain positive")
    if labels.get("prefixes_from_first_error") != 0:
        raise ValueError("prefixes from the first error must remain negative")

    canary = config.get("future_canary") or {}
    if canary.get("requires_separate_user_approval") is not True:
        raise ValueError("future model calls require separate approval")
    if int(canary.get("target_total", 0)) != 24:
        raise ValueError("future canary target must remain 24")
    if canary.get("target_by_benchmark") != {"medqa": 12, "medmcqa": 12}:
        raise ValueError("future canary benchmark targets changed")
    if canary.get("preferred_origin_targets") != {
        "existing_natural": 8,
        "local_student": 8,
        "controlled_single_error": 8,
    }:
        raise ValueError("future canary origin targets changed")
    if float(canary.get("transport_and_contract_success_rate", 0)) != 1.0:
        raise ValueError("future canary must require perfect contract success")
    if int(canary.get("minimum_distinct_strict_negative_trajectories", 0)) != 8:
        raise ValueError("future canary strict-negative minimum changed")
    if float(canary.get("minimum_human_trajectory_label_accuracy", 0)) < 0.90:
        raise ValueError("future canary trajectory-label quality gate was relaxed")
    if float(canary.get("minimum_human_exact_first_error_accuracy", 0)) < 0.80:
        raise ValueError("future canary first-error quality gate was relaxed")


def structurally_usable_wrong_answer(event: dict[str, Any]) -> bool:
    """Return whether a failed teacher event is usable only as a candidate.

    This never assigns a negative label. It merely identifies a structurally
    valid, wrong-answer trajectory that could be sent to an independent judge.
    """

    if event.get("status") != "complete":
        return False
    rule = event.get("rule_check") or {}
    failures = set(str(code) for code in rule.get("failure_codes") or [])
    if failures != {"gold_answer_mismatch"}:
        return False
    steps = rule.get("steps")
    predicted = rule.get("predicted_answer")
    record = event.get("record") or {}
    choices = record.get("choices") or {}
    return (
        isinstance(steps, list)
        and 3 <= len(steps) <= 8
        and all(isinstance(step, str) and bool(step.strip()) for step in steps)
        and isinstance(predicted, str)
        and predicted in choices
    )


def verification_disposition(result: object) -> str:
    """Classify a contract-shaped result under the conservative PRM policy."""

    if not isinstance(result, dict):
        return "unavailable"
    if result.get("problem_status") != "ok":
        return "human_review_non_ok_problem"

    label = result.get("trajectory_label")
    first = result.get("first_error_step")
    steps = result.get("steps")
    if type(label) is not int or label not in {0, 1} or not isinstance(steps, list):
        return "invalid_contract"
    if label == 1:
        if first is not None or result.get("answer_consistent") is not True:
            return "invalid_contract"
        if any(
            not isinstance(step, dict)
            or step.get("local_verdict") != "correct"
            or type(step.get("prefix_label")) is not int
            or step.get("prefix_label") != 1
            for step in steps
        ):
            return "invalid_contract"
        return "strict_positive"
    if first is None:
        return "answer_only_or_inconsistent_negative"
    if type(first) is not int or not 0 <= first < len(steps):
        return "invalid_contract"
    step = steps[first]
    if not isinstance(step, dict):
        return "invalid_contract"
    verdict = step.get("local_verdict")
    for index, value in enumerate(steps):
        if (
            not isinstance(value, dict)
            or type(value.get("prefix_label")) is not int
            or value.get("prefix_label") != int(index < first)
        ):
            return "invalid_contract"
    if verdict == "incorrect":
        return "strict_process_negative"
    if verdict == "uncertain":
        return "human_review_uncertain"
    return "invalid_contract"
