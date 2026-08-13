"""Stable, privacy-safe error codes for CoT judge attempts."""

from __future__ import annotations

import re


def error_category(error: str) -> str:
    exception = error.partition(":")[0].strip() or "UnknownError"
    lowered = error.casefold()
    if "http error" in lowered or exception == "HTTPError":
        match = re.search(r"\b([45]\d\d)\b", error)
        return f"http_{match.group(1)}" if match else "http_error"
    if "json" in lowered:
        return "json_parse_or_contract"
    if "timed out" in lowered or "timeout" in lowered:
        return "timeout"
    if exception in {"URLError", "ConnectionError"}:
        return "connection_error"
    if "usage.cost" in lowered:
        return "missing_provider_cost"
    if exception == "ValueError":
        return "response_contract"
    return exception


def error_detail(error: str) -> str:
    """Map a preserved exception message to a fixed code without emitting it."""

    exception = error.partition(":")[0].strip() or "UnknownError"
    lowered = error.casefold()
    if "usage.cost" in lowered:
        return "missing_provider_cost"
    if exception == "JSONDecodeError":
        syntax_patterns = (
            ("unterminated string", "json_syntax_unterminated_string"),
            ("expecting property name", "json_syntax_property_name"),
            ("expecting ',' delimiter", "json_syntax_missing_comma"),
            ("expecting ':' delimiter", "json_syntax_missing_colon"),
            ("expecting value", "json_syntax_missing_value"),
            ("extra data", "json_syntax_extra_data"),
            ("invalid \\escape", "json_syntax_invalid_escape"),
            ("invalid control character", "json_syntax_control_character"),
        )
        return next(
            (code for pattern, code in syntax_patterns if pattern in lowered),
            "json_syntax_other",
        )
    if exception == "ValueError":
        contract_patterns = (
            ("chat completion response has no text content", "response_no_text_content"),
            ("chat completion response content has invalid type", "response_content_type"),
            ("judge response must be one json object", "top_level_not_object"),
            ("validator json keys differ", "top_level_keys_differ"),
            ("invalid trajectory_label", "trajectory_label_invalid"),
            ("answer_consistent must be boolean", "answer_consistent_not_boolean"),
            ("invalid problem_status", "problem_status_invalid"),
            ("validator must return exactly one result per step", "step_count_mismatch"),
            ("validator step keys differ", "step_keys_differ"),
            ("validator step indices are not consecutive", "step_indices_invalid"),
            ("invalid local_verdict", "local_verdict_invalid"),
            ("invalid prefix_label", "prefix_label_invalid"),
            (
                "prefix labels do not become and remain zero after first error",
                "prefix_label_inconsistent",
            ),
            ("invalid validator step details", "step_details_invalid"),
            ("first_error_step must be integer or null", "first_error_step_type"),
            ("first_error_step differs from step verdicts", "first_error_step_inconsistent"),
            (
                "trajectory_label is inconsistent with validation details",
                "trajectory_label_inconsistent",
            ),
        )
        return next(
            (code for pattern, code in contract_patterns if pattern in lowered),
            "response_contract_other",
        )
    return error_category(error)
