"""Deterministic rule checks for teacher trajectory responses."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Iterable

from data_pipeline.cot_isolation import normalise_text


TAG_PATTERN = re.compile(r"<(step|answer)>(.*?)</\1>", re.DOTALL)


@dataclass(frozen=True)
class RuleCheckResult:
    passed: bool
    steps: tuple[str, ...]
    predicted_answer: str | None
    failure_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["steps"] = list(self.steps)
        value["failure_codes"] = list(self.failure_codes)
        return value


def check_teacher_response(
    raw_response: str,
    available_labels: Iterable[str],
    gold_answer: str,
    *,
    min_steps: int = 3,
    max_steps: int = 8,
    max_trajectory_characters: int = 12000,
    max_step_characters: int = 3000,
    finish_reason: str | None = None,
) -> RuleCheckResult:
    failures: list[str] = []
    text = str(raw_response)
    labels = {str(label).strip().upper() for label in available_labels}

    if finish_reason == "length":
        failures.append("truncated_output")
    if len(text) > max_trajectory_characters:
        failures.append("trajectory_too_long")
    if "<step>" not in text:
        failures.append("missing_step_tags")
    if "<answer>" not in text:
        failures.append("missing_answer_tag")

    matches = list(TAG_PATTERN.finditer(text))
    remainder = TAG_PATTERN.sub("", text)
    if remainder.strip() or any("<" in match.group(2) or ">" in match.group(2) for match in matches):
        failures.append("malformed_tag_nesting")

    steps = tuple(match.group(2).strip() for match in matches if match.group(1) == "step")
    answers = [match.group(2).strip() for match in matches if match.group(1) == "answer"]
    tag_order = [match.group(1) for match in matches]
    expected_order = ["step"] * len(steps) + (["answer"] if answers else [])
    if tag_order != expected_order:
        failures.append("malformed_tag_nesting")
    if not min_steps <= len(steps) <= max_steps:
        failures.append("step_count_out_of_range")
    if any(not step or len(step) > max_step_characters for step in steps):
        failures.append("trajectory_too_long")

    normalized_steps = [normalise_text(step) for step in steps]
    if len(set(normalized_steps)) != len(normalized_steps):
        failures.append("duplicate_step")

    if len(answers) > 1:
        failures.append("multiple_answer_tags")
    predicted_answer = answers[0].upper() if len(answers) == 1 else None
    if predicted_answer is not None and (
        len(predicted_answer) != 1 or predicted_answer not in labels
    ):
        failures.append("invalid_answer_label")
    if predicted_answer is not None and predicted_answer in labels:
        if predicted_answer != str(gold_answer).strip().upper():
            failures.append("gold_answer_mismatch")

    unique_failures = tuple(dict.fromkeys(failures))
    return RuleCheckResult(
        passed=not unique_failures,
        steps=steps,
        predicted_answer=predicted_answer,
        failure_codes=unique_failures,
    )
