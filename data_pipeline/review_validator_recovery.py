"""Render the six validator recovery cases for local, read-only human review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from data_pipeline.cot_api import validate_validator_result
from data_pipeline.cot_recovery_config import validate_recovery_config
from data_pipeline.run_cot_pilot_real import _load_completed
from data_pipeline.run_validator_recovery import REPO_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/cot/validator_recovery_v2.yaml"
    )
    parser.add_argument(
        "--case",
        type=int,
        action="append",
        dest="cases",
        help="show only this one-based case number; may be repeated",
    )
    return parser.parse_args()


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected one JSON object: {path}")
    return value


def _format_value(value: object) -> str:
    if value is None:
        return "null"
    if type(value) is bool:
        return str(value).lower()
    return str(value)


def build_review(
    config_path: Path, selected_cases: list[int] | None = None
) -> str:
    """Build a private review view without emitting source/request identifiers."""

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    validate_recovery_config(config)
    run_dir = _repo_path(str(config["source_run_dir"]))
    output_dir = run_dir / str(config["output_subdir"])
    manifest = _load_json(output_dir / "private_manifest.json")
    metadata = _load_json(output_dir / "canary_metadata.json")
    if metadata.get("status") != "complete":
        raise RuntimeError("recovery canary is not complete")

    selected = [
        (str(key[0]), int(key[1])) for key in manifest.get("selected_keys") or []
    ]
    expected = int(config["canary"]["total"])
    if len(selected) != expected or len(set(selected)) != expected:
        raise RuntimeError("private recovery selection does not match the config")
    if selected_cases:
        if any(type(case) is not int or not 1 <= case <= expected for case in selected_cases):
            raise ValueError(f"--case must be between 1 and {expected}")
        case_numbers = list(dict.fromkeys(selected_cases))
    else:
        case_numbers = list(range(1, expected + 1))

    teachers = _load_completed(
        run_dir / "teacher_events.jsonl", ("record_id", "candidate_index")
    )
    screeners = _load_completed(
        run_dir / "screener_events.jsonl", ("record_id", "candidate_index")
    )
    recovered = _load_completed(
        output_dir / "canary_events.jsonl", ("record_id", "candidate_index")
    )
    if set(recovered) != set(selected):
        raise RuntimeError("recovery event keys do not match the frozen selection")

    lines = [
        "VALIDATOR RECOVERY — LOCAL PRIVATE HUMAN REVIEW",
        "Read-only view. No API request or file write is performed.",
        "Record IDs, request IDs, hashes, costs, and provider metadata are omitted.",
        "",
        "For each case decide: AGREE, DISAGREE, or UNCERTAIN.",
        "If disagree, note the earliest flawed step and a short reason.",
    ]
    for case_number in case_numbers:
        key = selected[case_number - 1]
        if key not in teachers or key not in screeners or key not in recovered:
            raise RuntimeError(f"case {case_number} is missing a required event")
        teacher = teachers[key]
        record = teacher.get("record") or {}
        rule = teacher.get("rule_check") or {}
        screener = (screeners[key].get("result") or {})
        recovery_event = recovered[key]
        result = recovery_event.get("result")
        if recovery_event.get("status") != "complete" or not isinstance(result, dict):
            raise RuntimeError(f"case {case_number} recovery is not complete")
        steps = rule.get("steps") or []
        validate_validator_result(result, len(steps))

        lines.extend([
            "",
            "=" * 78,
            f"CASE {case_number}/{expected} | benchmark={record.get('benchmark', 'unknown')}",
            "=" * 78,
            "",
            "QUESTION",
            str(record.get("question") or ""),
            "",
            "OPTIONS",
        ])
        choices = record.get("choices") or {}
        for label, text in choices.items():
            lines.append(f"{label}. {text}")
        lines.extend([
            "",
            f"GOLD ANSWER: {record.get('answer')}",
            f"CANDIDATE ANSWER: {rule.get('predicted_answer')}",
            "",
            "CANDIDATE TRAJECTORY",
        ])
        for index, step in enumerate(steps):
            lines.append(f"[{index}] {step}")

        lines.extend([
            "",
            "ORIGINAL SCREENER",
            f"verdict: {_format_value(screener.get('verdict'))}",
            "suspected_first_error_step: "
            f"{_format_value(screener.get('suspected_first_error_step'))}",
            "error_codes: "
            f"{json.dumps(screener.get('error_codes') or [], ensure_ascii=False)}",
            f"reason: {screener.get('concise_reason') or ''}",
            "",
            "RECOVERY VALIDATOR V2",
            f"trajectory_label: {_format_value(result.get('trajectory_label'))}",
            f"first_error_step: {_format_value(result.get('first_error_step'))}",
            f"answer_consistent: {_format_value(result.get('answer_consistent'))}",
            f"problem_status: {_format_value(result.get('problem_status'))}",
        ])
        for step in result["steps"]:
            lines.extend([
                "",
                f"step {step['index']}: local={step['local_verdict']}; "
                f"prefix={step['prefix_label']}; "
                f"errors={json.dumps(step.get('error_codes') or [], ensure_ascii=False)}",
                f"validator reason: {step['concise_reason']}",
            ])
        lines.extend([
            "",
            f"HUMAN DECISION FOR CASE {case_number}: "
            "AGREE / DISAGREE(step=?, reason=...) / UNCERTAIN",
        ])

    lines.extend([
        "",
        "=" * 78,
        "REPORT ONLY THIS NON-PRIVATE SUMMARY",
        "=" * 78,
        "cases 1-6: AGREE / DISAGREE(step number + short category) / UNCERTAIN",
        "Do not paste questions, options, trajectory text, validator reasons, IDs, or hashes.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    print(build_review(_repo_path(args.config), args.cases))


if __name__ == "__main__":
    main()
