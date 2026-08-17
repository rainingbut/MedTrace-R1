"""Audit PRM negative-sample opportunities without calling a model or leaking text."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import yaml

from data_pipeline.cot_api import validate_validator_result
from data_pipeline.prm_negative_policy import (
    STRUCTURAL_FAILURE_CODES,
    structurally_usable_wrong_answer,
    validate_prm_negative_policy,
    verification_disposition,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "metadata.json",
    "teacher_events.jsonl",
    "screener_events.jsonl",
    "validator_events.jsonl",
    "canonical_trajectories.jsonl",
    "sft_verified.jsonl",
    "process_train.jsonl",
)
SAFE_RULE_FAILURE_CODES = STRUCTURAL_FAILURE_CODES | {"gold_answer_mismatch"}
VALIDATOR_RESULT_FIELDS = (
    "trajectory_label",
    "first_error_step",
    "answer_consistent",
    "problem_status",
    "steps",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/cot/prm_negative_enrichment_v1.yaml"
    )
    return parser.parse_args()


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected one JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            records.append(value)
    return records


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _write_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(value)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hashes(run_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in SOURCE_FILES:
        path = run_dir / name
        if not path.is_file():
            raise RuntimeError(f"PRM opportunity audit source is missing: {name}")
        hashes[name] = _sha256(path)
    return hashes


def _event_key(event: dict[str, Any]) -> tuple[str, int]:
    return str(event["record_id"]), int(event["candidate_index"])


def _benchmark(event: dict[str, Any]) -> str:
    value = str((event.get("record") or {}).get("benchmark") or "unknown")
    return value if value in {"medqa", "medmcqa"} else "unknown"


def _counts(values: Iterable[object]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def _strict_labels(values: Iterable[object]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for value in values:
        if type(value) is int and value in {0, 1}:
            counts[f"int:{value}"] += 1
        elif type(value) is bool:
            counts[f"bool:{str(value).lower()}"] += 1
        else:
            counts[f"invalid:{type(value).__name__}"] += 1
    return dict(sorted(counts.items()))


def _verified_counts(
    events: list[dict[str, Any]],
    teachers: dict[tuple[str, int], dict[str, Any]],
) -> tuple[dict[str, int], int]:
    dispositions: Counter[str] = Counter()
    contract_valid = 0
    for event in events:
        key = _event_key(event)
        teacher = teachers.get(key)
        result = event.get("result")
        if teacher is None or event.get("status") != "complete":
            dispositions["unavailable"] += 1
            continue
        step_count = len(((teacher.get("rule_check") or {}).get("steps")) or [])
        try:
            validate_validator_result(result, step_count)
        except (KeyError, TypeError, ValueError):
            dispositions["invalid_contract"] += 1
            continue
        contract_valid += 1
        dispositions[verification_disposition(result)] += 1
    return dict(sorted(dispositions.items())), contract_valid


def _recovery_summary(
    run_dir: Path,
    teachers: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    path = run_dir / "validator_recovery_v2" / "canary_events.jsonl"
    if not path.is_file():
        return {
            "present": False,
            "events": 0,
            "strict_contract_valid": 0,
            "dispositions": {},
        }
    events = _load_jsonl(path)
    dispositions, contract_valid = _verified_counts(events, teachers)
    return {
        "present": True,
        "events": len(events),
        "strict_contract_valid": contract_valid,
        "dispositions": dispositions,
    }


def _canonical_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    dispositions: Counter[str] = Counter()
    contract_valid = 0
    for record in records:
        verification = record.get("verification")
        trajectory = record.get("trajectory") or {}
        trajectory_steps = trajectory.get("steps") or []
        if not isinstance(verification, dict) or not isinstance(trajectory_steps, list):
            dispositions["invalid_contract"] += 1
            continue
        payload = {field: verification.get(field) for field in VALIDATOR_RESULT_FIELDS}
        try:
            validate_validator_result(payload, len(trajectory_steps))
        except (KeyError, TypeError, ValueError):
            dispositions["invalid_contract"] += 1
            continue
        contract_valid += 1
        dispositions[verification_disposition(payload)] += 1
    return {
        "strict_contract_valid": contract_valid,
        "dispositions": dict(sorted(dispositions.items())),
    }


def audit_opportunities(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("PRM negative enrichment config must be an object")
    validate_prm_negative_policy(config)

    run_dir = _repo_path(config["source_run_dir"])
    hashes_before = _source_hashes(run_dir)
    metadata = _load_json(run_dir / "metadata.json")
    teachers_list = _load_jsonl(run_dir / "teacher_events.jsonl")
    screeners_list = _load_jsonl(run_dir / "screener_events.jsonl")
    validators_list = _load_jsonl(run_dir / "validator_events.jsonl")
    canonical = _load_jsonl(run_dir / "canonical_trajectories.jsonl")
    sft = _load_jsonl(run_dir / "sft_verified.jsonl")
    prm = _load_jsonl(run_dir / "process_train.jsonl")

    teachers = {_event_key(event): event for event in teachers_list}
    screeners = {_event_key(event): event for event in screeners_list}
    validators = {_event_key(event): event for event in validators_list}
    canonical_ids = [str(record.get("trajectory_id")) for record in canonical]
    sft_ids = [str(record.get("trajectory_id")) for record in sft]
    prm_keys = [
        (str(record.get("trajectory_id")), int(record.get("step_index", -1)))
        for record in prm
    ]

    answer_mismatch = [
        event for event in teachers_list if structurally_usable_wrong_answer(event)
    ]
    screener_reject: list[dict[str, Any]] = []
    for key, screener in screeners.items():
        if ((screener.get("result") or {}).get("verdict")) == "reject":
            teacher = teachers.get(key)
            if teacher is not None:
                screener_reject.append(teacher)

    failure_codes = Counter(
        str(code) if str(code) in SAFE_RULE_FAILURE_CODES else "unknown"
        for event in teachers_list
        for code in ((event.get("rule_check") or {}).get("failure_codes") or [])
    )
    natural_by_benchmark: dict[str, Counter[str]] = defaultdict(Counter)
    for event in answer_mismatch:
        natural_by_benchmark[_benchmark(event)]["answer_mismatch_only"] += 1
    for event in screener_reject:
        natural_by_benchmark[_benchmark(event)]["screener_reject"] += 1

    validator_dispositions, validator_contract_valid = _verified_counts(
        validators_list, teachers
    )
    canonical_summary = _canonical_summary(canonical)

    negative_prefix_by_trajectory: Counter[str] = Counter()
    for record in prm:
        if type(record.get("label")) is int and record["label"] == 0:
            negative_prefix_by_trajectory[str(record.get("trajectory_id"))] += 1
    negative_prefix_multiplicity = _counts(negative_prefix_by_trajectory.values())

    expected = config["source_identity"]
    count_checks = {
        "metadata_complete": metadata.get("status") == "complete",
        "questions": int(metadata.get("questions", -1)) == int(expected["questions"]),
        "teacher_events": len(teachers_list) == int(expected["teacher_events"]),
        "teacher_events_match_metadata": len(teachers_list)
        == int(metadata.get("teacher_events", -1)),
        "screener_events_match_metadata": len(screeners_list)
        == int(metadata.get("screener_events", -1)),
        "validator_events_match_metadata": len(validators_list)
        == int(metadata.get("validator_events", -1)),
        "rule_passed": sum(
            (event.get("rule_check") or {}).get("passed") is True
            for event in teachers_list
        ) == int(expected["rule_passed"]),
        "canonical_trajectories": len(canonical)
        == int(expected["canonical_trajectories"]),
        "canonical_matches_metadata": len(canonical)
        == int(metadata.get("canonical_trajectories", -1)),
        "prm_records": len(prm) == int(expected["prm_records"]),
        "prm_matches_metadata": len(prm) == int(metadata.get("prm_records", -1)),
        "sft_matches_metadata": len(sft) == int(metadata.get("sft_records", -1)),
        "teacher_event_keys_unique": len(teachers) == len(teachers_list),
        "screener_event_keys_unique": len(screeners) == len(screeners_list),
        "validator_event_keys_unique": len(validators) == len(validators_list),
        "canonical_trajectory_ids_unique": len(set(canonical_ids)) == len(canonical_ids),
        "sft_trajectory_ids_unique": len(set(sft_ids)) == len(sft_ids),
        "sft_is_canonical_subset": set(sft_ids).issubset(canonical_ids),
        "prm_trajectory_step_keys_unique": len(set(prm_keys)) == len(prm_keys),
        "prm_labels_are_strict_binary_integers": all(
            type(record.get("label")) is int and record["label"] in {0, 1}
            for record in prm
        ),
    }
    hashes_after = _source_hashes(run_dir)
    source_unchanged = hashes_before == hashes_after

    report = {
        "schema_version": "medtrace.prm-negative-opportunity-audit.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contains_private_text_or_ids": False,
        "source": {
            "status": metadata.get("status"),
            "questions": int(metadata.get("questions", 0)),
            "teacher_events": len(teachers_list),
            "screener_events": len(screeners_list),
            "validator_events": len(validators_list),
            "canonical_trajectories": len(canonical),
            "sft_records": len(sft),
            "prm_records": len(prm),
        },
        "natural_opportunities": {
            "total": len(answer_mismatch) + len(screener_reject),
            "answer_mismatch_only": len(answer_mismatch),
            "screener_reject": len(screener_reject),
            "by_benchmark": {
                benchmark: dict(sorted(counts.items()))
                for benchmark, counts in sorted(natural_by_benchmark.items())
            },
            "teacher_rule_failure_codes": dict(sorted(failure_codes.items())),
            "automatic_negative_labels_assigned": 0,
            "note": "Every opportunity still requires independent validation.",
        },
        "validator": {
            "strict_contract_valid": validator_contract_valid,
            "dispositions": validator_dispositions,
        },
        "recovery_canary": _recovery_summary(run_dir, teachers),
        "canonical": canonical_summary,
        "prm": {
            "strict_labels": _strict_labels(record.get("label") for record in prm),
            "negative_prefix_records": sum(negative_prefix_by_trajectory.values()),
            "distinct_trajectories_with_negative_prefix": len(
                negative_prefix_by_trajectory
            ),
            "negative_prefix_records_per_negative_trajectory":
                negative_prefix_multiplicity,
            "readiness_basis": "distinct strict first-error trajectories, not row balance",
        },
        "integrity": {
            "count_checks": count_checks,
            "source_artifacts_unchanged": source_unchanged,
            "passed": all(count_checks.values()) and source_unchanged,
        },
        "decision": {
            "full_scale_authorized": False,
            "model_or_api_calls_authorized": False,
            "next_gate": "Review this aggregate audit before freezing a paid canary.",
        },
    }
    return report


def render_markdown(report: dict[str, Any]) -> str:
    source = report["source"]
    opportunities = report["natural_opportunities"]
    validator = report["validator"]
    recovery = report["recovery_canary"]
    canonical = report["canonical"]
    prm = report["prm"]
    by_benchmark = json.dumps(opportunities["by_benchmark"], sort_keys=True)
    negative_multiplicity = json.dumps(
        prm["negative_prefix_records_per_negative_trajectory"], sort_keys=True
    )
    return "\n".join(
        [
            "# PRM negative opportunity audit",
            "",
            "This report is aggregate-only and contains no private question, "
            "ID, or trajectory text.",
            "",
            "## Source snapshot",
            "",
            f"- Questions: {source['questions']}",
            f"- Teacher / canonical / SFT: {source['teacher_events']} / "
            f"{source['canonical_trajectories']} / {source['sft_records']}",
            f"- PRM records: {source['prm_records']}",
            "",
            "## Natural negative opportunities",
            "",
            f"- Total candidates requiring validation: {opportunities['total']}",
            f"- Structurally usable answer mismatch only: "
            f"{opportunities['answer_mismatch_only']}",
            f"- Screener reject: {opportunities['screener_reject']}",
            f"- By benchmark: `{by_benchmark}`",
            "- Automatic negative labels assigned: 0",
            "",
            "## Validator evidence",
            "",
            f"- Original validator dispositions: `"
            f"{json.dumps(validator['dispositions'], sort_keys=True)}`",
            f"- Recovery canary present/events/contract-valid: "
            f"{str(recovery['present']).lower()} / {recovery['events']} / "
            f"{recovery['strict_contract_valid']}",
            f"- Recovery dispositions: `"
            f"{json.dumps(recovery['dispositions'], sort_keys=True)}`",
            f"- Canonical dispositions: `"
            f"{json.dumps(canonical['dispositions'], sort_keys=True)}`",
            "",
            "## Existing PRM evidence",
            "",
            f"- Strict labels: `{json.dumps(prm['strict_labels'], sort_keys=True)}`",
            f"- Negative prefix records: {prm['negative_prefix_records']}",
            f"- Distinct trajectories with a negative prefix: "
            f"{prm['distinct_trajectories_with_negative_prefix']}",
            f"- Negative multiplicity: `{negative_multiplicity}`",
            "",
            "## Gate",
            "",
            f"- Integrity passed: {str(report['integrity']['passed']).lower()}",
            "- Model/API calls authorized: false",
            "- Full-scale generation authorized: false",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    config_path = _repo_path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("PRM negative enrichment config must be an object")
    validate_prm_negative_policy(config)
    report = audit_opportunities(config_path)
    run_dir = _repo_path(config["source_run_dir"])
    output_dir = run_dir / str(config["output_subdir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = str(config["output_stem"])
    json_path = output_dir / f"{stem}.json"
    markdown_path = output_dir / f"{stem}.md"
    _write_json(json_path, report)
    _write_text(markdown_path, render_markdown(report))
    print(render_markdown(report), end="")
    if not report["integrity"]["passed"]:
        raise RuntimeError("PRM negative opportunity audit integrity checks failed")


if __name__ == "__main__":
    main()
