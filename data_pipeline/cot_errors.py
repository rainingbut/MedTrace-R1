"""Frozen failure taxonomy for stage-2 generation and verification."""

FORMAT_ERROR_CODES = (
    "missing_step_tags",
    "missing_answer_tag",
    "malformed_tag_nesting",
    "invalid_answer_label",
    "multiple_answer_tags",
    "step_count_out_of_range",
    "trajectory_too_long",
    "duplicate_step",
    "duplicate_candidate",
    "truncated_output",
    "model_refusal",
)

REASONING_ERROR_CODES = (
    "gold_answer_mismatch",
    "medical_fact_error",
    "causal_error",
    "calculation_error",
    "logical_gap",
    "internal_contradiction",
    "irrelevant_reasoning",
    "unsupported_claim",
    "answer_not_supported_by_steps",
)

DATA_ERROR_CODES = (
    "ambiguous_question",
    "suspected_bad_gold",
    "evaluation_overlap",
    "unverified_source",
    "non_train_split",
)

SYSTEM_ERROR_CODES = (
    "api_error",
    "judge_parse_error",
    "budget_stop",
)

ALL_ERROR_CODES = (
    FORMAT_ERROR_CODES
    + REASONING_ERROR_CODES
    + DATA_ERROR_CODES
    + SYSTEM_ERROR_CODES
)
