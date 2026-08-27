"""Private state and immutable contracts for PRM disagreement adjudication."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

import yaml

from data_pipeline.prm_negative_human_adjudication_config import (
    validate_prm_negative_human_adjudication_config,
)
from data_pipeline.prm_negative_human_review_state import (
    load_annotation_jsonl,
    load_human_review_config,
    load_review_context,
    validate_annotation_lock,
    validate_annotations,
)
from data_pipeline.prm_negative_recovery_state import sha256_file


ADJUDICATION_SCHEMA = "medtrace.prm-negative-human-adjudication.v1"
LOCK_SCHEMA = "medtrace.prm-negative-human-adjudication-lock.v1"


def load_adjudication_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("PRM human-adjudication config must be an object")
    validate_prm_negative_human_adjudication_config(config)
    return config


def _validate_raw_audit(report: dict[str, Any], expected: dict[str, Any]) -> None:
    annotations = report.get("annotations") or {}
    scores = report.get("scores") or {}
    trajectory = scores.get("trajectory_label") or {}
    first_error = scores.get("exact_first_error") or {}
    candidates = report.get("candidate_negative_list") or {}
    gate = report.get("quality_gate") or {}
    actual = {
        "total_annotations": annotations.get("total"),
        "human_problem_ok": annotations.get("human_problem_ok"),
        "raw_trajectory_correct": trajectory.get("correct"),
        "raw_trajectory_total": trajectory.get("total"),
        "raw_first_error_correct": first_error.get("correct"),
        "raw_first_error_total": first_error.get("total"),
        "raw_conservative_candidates": candidates.get("records"),
        "required_failed_checks": gate.get("failed_checks"),
    }
    for field, value in actual.items():
        if value != expected[field]:
            raise RuntimeError(f"source human-review audit changed: {field}")
    if gate.get("passed") is not False:
        raise RuntimeError("source human-review audit must preserve its failed gate")


def load_adjudication_context(
    config: dict[str, Any], repo_root: Path
) -> dict[str, Any]:
    human_config_path = repo_root / str(config["source_human_review_config"])
    human_config = load_human_review_config(human_config_path)
    context = load_review_context(human_config, repo_root)
    files = human_config["private_files"]
    annotation_path = context["canary_dir"] / str(files["annotations"])
    records = load_annotation_jsonl(annotation_path)
    metadata, annotations = validate_annotations(
        records, context, require_complete=True
    )
    annotation_lock_path = context["canary_dir"] / str(files["annotation_lock"])
    annotation_lock = json.loads(
        annotation_lock_path.read_text(encoding="utf-8")
    )
    validate_annotation_lock(
        annotation_lock, annotation_path, context, metadata
    )
    recovered_key_path = context["canary_dir"] / str(files["recovered_key"])
    if not recovered_key_path.is_file():
        raise RuntimeError("recovered validator key is required after blind lock")
    raw_audit_path = context["canary_dir"] / str(files["aggregate_audit_json"])
    raw_audit = json.loads(raw_audit_path.read_text(encoding="utf-8"))
    _validate_raw_audit(raw_audit, config["source_gate"])
    from data_pipeline.audit_prm_negative_human_review import score_review

    recomputed_raw_audit, _ = score_review(
        human_config_path, repo_root=repo_root
    )
    if raw_audit != recomputed_raw_audit:
        raise RuntimeError("stored raw human-review audit differs from recomputation")

    disagreements: list[dict[str, Any]] = []
    for case_number, (candidate, annotation) in enumerate(
        zip(context["source"]["candidates"], annotations, strict=True), start=1
    ):
        if annotation["human_error_type"] != "process":
            continue
        result = context["canonical"][str(candidate["candidate_id"])]["result"]
        human_first = annotation["human_first_error_step"]
        validator_first = result["first_error_step"]
        if human_first != validator_first:
            disagreements.append(
                {
                    "case_number": case_number,
                    "original_human_first_error_step": human_first,
                    "validator_first_error_step": validator_first,
                }
            )
    if len(disagreements) != config["source_gate"]["total_disagreements"]:
        raise RuntimeError("source first-error disagreement count changed")
    return {
        "human_config": human_config,
        "review_context": context,
        "annotations": annotations,
        "annotation_path": annotation_path,
        "annotation_lock_path": annotation_lock_path,
        "recovered_key_path": recovered_key_path,
        "raw_audit_path": raw_audit_path,
        "raw_audit": raw_audit,
        "disagreements": disagreements,
        "canary_dir": context["canary_dir"],
    }


def adjudication_template(disagreements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = [
        {
            "schema_version": ADJUDICATION_SCHEMA,
            "record_type": "adjudication_metadata",
            "reviewer_role": "",
            "unblinded_to_validator_outputs": None,
            "original_blind_review_preserved": None,
            "decision_protocol": "earliest_unambiguously_incorrect_step",
            "adjudication_completed_at_utc": None,
        }
    ]
    for disagreement in disagreements:
        records.append(
            {
                "schema_version": ADJUDICATION_SCHEMA,
                "record_type": "case_adjudication",
                **disagreement,
                "adjudicated_first_error_step": None,
                "decision_source": None,
                "rationale": "",
            }
        )
    return records


def validate_adjudications(
    records: list[dict[str, Any]],
    adjudication_context: dict[str, Any],
    *,
    require_complete: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    disagreements = adjudication_context["disagreements"]
    if len(records) != len(disagreements) + 1:
        raise ValueError("human adjudication record count changed")
    metadata = records[0]
    if set(metadata) != {
        "schema_version", "record_type", "reviewer_role",
        "unblinded_to_validator_outputs", "original_blind_review_preserved",
        "decision_protocol", "adjudication_completed_at_utc",
    }:
        raise ValueError("human adjudication metadata keys differ")
    if metadata["schema_version"] != ADJUDICATION_SCHEMA:
        raise ValueError("human adjudication metadata schema changed")
    if metadata["record_type"] != "adjudication_metadata":
        raise ValueError("first adjudication record must be metadata")
    if metadata["decision_protocol"] != "earliest_unambiguously_incorrect_step":
        raise ValueError("human adjudication decision protocol changed")
    if require_complete:
        if not isinstance(metadata["reviewer_role"], str) or not metadata[
            "reviewer_role"
        ].strip():
            raise ValueError("adjudication reviewer_role is required")
        if metadata["unblinded_to_validator_outputs"] is not True:
            raise ValueError("unblinded adjudication attestation must be true")
        if metadata["original_blind_review_preserved"] is not True:
            raise ValueError("original blind review must remain preserved")
        completed = metadata["adjudication_completed_at_utc"]
        if not isinstance(completed, str) or not completed.strip():
            raise ValueError("adjudication_completed_at_utc is required")
        try:
            parsed = datetime.fromisoformat(completed.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("adjudication timestamp must be ISO-8601") from exc
        if parsed.tzinfo is None:
            raise ValueError("adjudication timestamp must include a timezone")

    expected_keys = {
        "schema_version", "record_type", "case_number",
        "original_human_first_error_step", "validator_first_error_step",
        "adjudicated_first_error_step", "decision_source", "rationale",
    }
    adjudications = records[1:]
    for expected, record in zip(disagreements, adjudications, strict=True):
        if set(record) != expected_keys:
            raise ValueError("human adjudication case keys differ")
        if record["schema_version"] != ADJUDICATION_SCHEMA:
            raise ValueError("human adjudication case schema changed")
        if record["record_type"] != "case_adjudication":
            raise ValueError("human adjudication record_type changed")
        for field, value in expected.items():
            if record[field] != value:
                raise ValueError(f"source disagreement changed: {field}")
        decision = record["adjudicated_first_error_step"]
        source = record["decision_source"]
        rationale = record["rationale"]
        if (
            not require_complete and decision is None and source is None
            and rationale == ""
        ):
            continue
        allowed = {
            record["original_human_first_error_step"],
            record["validator_first_error_step"],
        }
        if type(decision) is not int or decision not in allowed:
            raise ValueError("adjudicated step must be the human or validator step")
        expected_source = (
            "human"
            if decision == record["original_human_first_error_step"]
            else "validator"
        )
        if source != expected_source:
            raise ValueError("adjudication decision_source is inconsistent")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError("adjudication rationale is required")
        if len(rationale) > 4000:
            raise ValueError("adjudication rationale is too long")
    return metadata, adjudications


def expected_adjudication_lock(
    adjudication_path: Path,
    adjudication_context: dict[str, Any],
    metadata: dict[str, Any],
    *,
    locked_at_utc: str,
) -> dict[str, Any]:
    return {
        "schema_version": LOCK_SCHEMA,
        "adjudication_sha256": sha256_file(adjudication_path),
        "human_annotation_lock_sha256": sha256_file(
            adjudication_context["annotation_lock_path"]
        ),
        "recovered_validator_key_sha256": sha256_file(
            adjudication_context["recovered_key_path"]
        ),
        "raw_human_review_audit_sha256": sha256_file(
            adjudication_context["raw_audit_path"]
        ),
        "total_disagreements": len(adjudication_context["disagreements"]),
        "reviewer_role": metadata["reviewer_role"].strip(),
        "unblinded_to_validator_outputs": True,
        "original_blind_review_preserved": True,
        "adjudication_completed_at_utc": metadata[
            "adjudication_completed_at_utc"
        ],
        "locked_at_utc": locked_at_utc,
    }


def validate_adjudication_lock(
    lock: dict[str, Any],
    adjudication_path: Path,
    adjudication_context: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    locked_at = lock.get("locked_at_utc")
    if not isinstance(locked_at, str) or not locked_at:
        raise ValueError("human adjudication lock timestamp is missing")
    if lock != expected_adjudication_lock(
        adjudication_path,
        adjudication_context,
        metadata,
        locked_at_utc=locked_at,
    ):
        raise RuntimeError("human adjudication lock or frozen inputs changed")
