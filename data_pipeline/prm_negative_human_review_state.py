"""Shared private state and contract helpers for PRM human review."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

import yaml

from data_pipeline.cot_api import validate_validator_result
from data_pipeline.prm_negative_human_review_config import (
    validate_prm_negative_human_review_config,
)
from data_pipeline.prm_negative_recovery_config import (
    validate_prm_negative_recovery_config,
)
from data_pipeline.prm_negative_recovery_state import (
    canonical_event_map,
    load_source_state,
    sha256_file,
)
from data_pipeline.run_cot_pilot_real import _load_jsonl


ANNOTATION_SCHEMA = "medtrace.prm-negative-human-annotation.v1"
LOCK_SCHEMA = "medtrace.prm-negative-human-annotation-lock.v1"


def load_human_review_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("PRM human-review config must be an object")
    validate_prm_negative_human_review_config(config)
    return config


def load_review_context(
    config: dict[str, Any], repo_root: Path
) -> dict[str, Any]:
    recovery_config_path = repo_root / str(config["source_recovery_config"])
    recovery_config = yaml.safe_load(
        recovery_config_path.read_text(encoding="utf-8")
    )
    if not isinstance(recovery_config, dict):
        raise ValueError("PRM recovery config must be an object")
    validate_prm_negative_recovery_config(recovery_config)
    expected = {
        "source_run_dir": config["source_run_dir"],
        "source_canary_subdir": config["source_canary_subdir"],
        "output_subdir": config["source_recovery_subdir"],
    }
    for field, value in expected.items():
        if recovery_config.get(field) != value:
            raise ValueError(f"human review and recovery differ: {field}")
    source = load_source_state(recovery_config, repo_root)
    recovery_dir = repo_root / str(config["source_run_dir"]) / str(
        config["source_recovery_subdir"]
    )
    recovery_metadata = json.loads(
        (recovery_dir / "metadata.json").read_text(encoding="utf-8")
    )
    if recovery_metadata.get("status") != "complete":
        raise RuntimeError("PRM recovery must be terminal before human review")
    attempts_path = recovery_dir / "recovery_attempts.jsonl"
    attempts = _load_jsonl(attempts_path)
    canonical, provenance = canonical_event_map(source, attempts)
    if set(canonical) != set(source["candidate_by_id"]):
        raise RuntimeError("canonical review keys differ from candidates")
    for candidate_id, candidate in source["candidate_by_id"].items():
        event = canonical[candidate_id]
        if event.get("status") != "complete":
            raise RuntimeError("human review requires 24 complete canonical results")
        validate_validator_result(
            event.get("result"), len(candidate["trajectory"]["steps"])
        )
    canary_dir = repo_root / str(config["source_run_dir"]) / str(
        config["source_canary_subdir"]
    )
    return {
        "source": source,
        "canonical": canonical,
        "provenance": provenance,
        "canary_dir": canary_dir,
        "recovery_dir": recovery_dir,
        "attempts_path": attempts_path,
    }


def annotation_template(total: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = [
        {
            "schema_version": ANNOTATION_SCHEMA,
            "record_type": "review_metadata",
            "reviewer_role": "",
            "blinded_to_validator_outputs": None,
            "review_completed_at_utc": None,
        }
    ]
    records.extend(
        {
            "schema_version": ANNOTATION_SCHEMA,
            "record_type": "case_annotation",
            "case_number": case_number,
            "human_problem_status": None,
            "human_trajectory_label": None,
            "human_first_error_step": None,
            "notes": "",
        }
        for case_number in range(1, total + 1)
    )
    return records


def load_annotation_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate_annotations(
    records: list[dict[str, Any]],
    context: dict[str, Any],
    *,
    require_complete: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates = context["source"]["candidates"]
    if len(records) != len(candidates) + 1:
        raise ValueError("human annotation record count changed")
    metadata = records[0]
    if set(metadata) != {
        "schema_version", "record_type", "reviewer_role",
        "blinded_to_validator_outputs", "review_completed_at_utc",
    }:
        raise ValueError("human review metadata keys differ")
    if metadata["schema_version"] != ANNOTATION_SCHEMA:
        raise ValueError("human review metadata schema changed")
    if metadata["record_type"] != "review_metadata":
        raise ValueError("first human review record must be metadata")
    if require_complete:
        if not isinstance(metadata["reviewer_role"], str) or not metadata[
            "reviewer_role"
        ].strip():
            raise ValueError("reviewer_role is required")
        if metadata["blinded_to_validator_outputs"] is not True:
            raise ValueError("blind-review attestation must be true")
        completed = metadata["review_completed_at_utc"]
        if not isinstance(completed, str) or not completed.strip():
            raise ValueError("review_completed_at_utc is required")
        try:
            parsed = datetime.fromisoformat(completed.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("review_completed_at_utc must be ISO-8601") from exc
        if parsed.tzinfo is None:
            raise ValueError("review_completed_at_utc must include a timezone")

    annotations = records[1:]
    expected_cases = list(range(1, len(candidates) + 1))
    actual_cases = [record.get("case_number") for record in annotations]
    if actual_cases != expected_cases:
        raise ValueError("human annotation case order changed")
    required_keys = {
        "schema_version", "record_type", "case_number",
        "human_problem_status", "human_trajectory_label",
        "human_first_error_step", "notes",
    }
    for candidate, record in zip(candidates, annotations, strict=True):
        if set(record) != required_keys:
            raise ValueError("human case annotation keys differ")
        if record["schema_version"] != ANNOTATION_SCHEMA:
            raise ValueError("human case annotation schema changed")
        if record["record_type"] != "case_annotation":
            raise ValueError("human case record_type changed")
        if not isinstance(record["notes"], str) or len(record["notes"]) > 4000:
            raise ValueError("human annotation notes are invalid")
        status = record["human_problem_status"]
        label = record["human_trajectory_label"]
        first = record["human_first_error_step"]
        if not require_complete and status is None and label is None and first is None:
            continue
        if status not in {"ok", "ambiguous", "bad_gold"}:
            raise ValueError("human problem status is invalid")
        if status != "ok":
            if label is not None or first is not None:
                raise ValueError("non-ok human cases must not have trajectory labels")
            continue
        if type(label) is not int or label not in {0, 1}:
            raise ValueError("ok human cases require a strict integer label")
        if label == 1:
            if first is not None:
                raise ValueError("positive human cases must have null first error")
            continue
        step_count = len(candidate["trajectory"]["steps"])
        if type(first) is not int or not 0 <= first < step_count:
            raise ValueError("negative human first error is outside the trajectory")
    return metadata, annotations


def expected_lock(
    annotation_path: Path,
    context: dict[str, Any],
    metadata: dict[str, Any],
    *,
    locked_at_utc: str,
) -> dict[str, Any]:
    source_hashes = context["source"]["source_canary_hashes"]
    return {
        "schema_version": LOCK_SCHEMA,
        "annotation_sha256": sha256_file(annotation_path),
        "source_candidates_sha256": source_hashes["candidates.jsonl"],
        "source_validator_events_sha256": source_hashes["validator_events.jsonl"],
        "recovery_attempts_sha256": sha256_file(context["attempts_path"]),
        "total_cases": len(context["source"]["candidates"]),
        "reviewer_role": metadata["reviewer_role"].strip(),
        "blinded_to_validator_outputs": True,
        "review_completed_at_utc": metadata["review_completed_at_utc"],
        "locked_at_utc": locked_at_utc,
    }


def validate_annotation_lock(
    lock: dict[str, Any],
    annotation_path: Path,
    context: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    locked_at = lock.get("locked_at_utc")
    if not isinstance(locked_at, str) or not locked_at:
        raise ValueError("human annotation lock timestamp is missing")
    if lock != expected_lock(
        annotation_path, context, metadata, locked_at_utc=locked_at
    ):
        raise RuntimeError("human annotation lock or its frozen inputs changed")
