"""Source validation and pure record construction for PRM materialization."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from data_pipeline.audit_prm_negative_human_adjudication import score_adjudication
from data_pipeline.build_strict_pilot_view import _source_hashes
from data_pipeline.prm_negative_canary_config import (
    validate_prm_negative_canary_config,
)
from data_pipeline.prm_negative_human_adjudication_state import (
    load_adjudication_config,
    load_adjudication_context,
    validate_adjudication_lock,
    validate_adjudications,
)
from data_pipeline.prm_negative_materialization_config import (
    validate_prm_negative_materialization_config,
)
from data_pipeline.prm_negative_recovery_state import sha256_file
from data_pipeline.prm_negative_human_review_state import load_annotation_jsonl
from data_pipeline.prm_negative_policy import verification_disposition


RECORD_KEYS = {
    "trajectory_id", "step_index", "prefix", "label", "error_codes", "source"
}


def load_materialization_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("PRM materialization config must be an object")
    validate_prm_negative_materialization_config(config)
    return config


def _json_fingerprint(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_source(source: object) -> dict[str, Any]:
    if not isinstance(source, dict):
        raise ValueError("PRM source must be an object")
    if source.get("dataset") not in {"medqa", "medmcqa"}:
        raise ValueError("PRM source benchmark is invalid")
    if source.get("split") != "train":
        raise ValueError("PRM source must remain on the train split")
    content_hash = source.get("content_sha256")
    if not isinstance(content_hash, str) or len(content_hash) != 64:
        raise ValueError("PRM source content hash is invalid")
    if not isinstance(source.get("source_id"), str):
        raise ValueError("PRM source ID is invalid")
    return source


def _validate_record_shape(record: dict[str, Any]) -> None:
    if set(record) != RECORD_KEYS:
        raise ValueError("PRM materialized record keys differ")
    if not isinstance(record["trajectory_id"], str) or not record["trajectory_id"]:
        raise ValueError("PRM trajectory ID is invalid")
    if type(record["step_index"]) is not int or record["step_index"] < 0:
        raise ValueError("PRM step index is invalid")
    prefix = record["prefix"]
    if (
        not isinstance(prefix, list)
        or len(prefix) != record["step_index"] + 1
        or not all(isinstance(step, str) and step.strip() for step in prefix)
    ):
        raise ValueError("PRM prefix is invalid")
    if type(record["label"]) is not int or record["label"] not in {0, 1}:
        raise ValueError("PRM label must be a strict binary integer")
    if not isinstance(record["error_codes"], list) or not all(
        isinstance(code, str) for code in record["error_codes"]
    ):
        raise ValueError("PRM error codes are invalid")
    _validate_source(record["source"])


def validate_prm_record_set(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    record_keys: set[tuple[str, int]] = set()
    duplicate_records = 0
    for record in records:
        _validate_record_shape(record)
        key = (record["trajectory_id"], record["step_index"])
        if key in record_keys:
            duplicate_records += 1
        record_keys.add(key)
        grouped[record["trajectory_id"]].append(record)
    negative_trajectories = 0
    fingerprints: dict[str, str] = {}
    for trajectory_id, trajectory_records in grouped.items():
        trajectory_records.sort(key=lambda item: item["step_index"])
        indices = [record["step_index"] for record in trajectory_records]
        if indices != list(range(len(trajectory_records))):
            raise ValueError("PRM step indices are not contiguous")
        for index, record in enumerate(trajectory_records):
            if index and record["prefix"][:-1] != trajectory_records[index - 1][
                "prefix"
            ]:
                raise ValueError("PRM prefixes do not grow by exactly one step")
        labels = [record["label"] for record in trajectory_records]
        if 0 in labels:
            first = labels.index(0)
            if labels != [1] * first + [0] * (len(labels) - first):
                raise ValueError("PRM prefix labels recover after the first error")
            negative_trajectories += 1
        full = trajectory_records[-1]
        fingerprints[trajectory_id] = _json_fingerprint(
            {"source": full["source"], "steps": full["prefix"]}
        )
    labels = Counter(record["label"] for record in records)
    return {
        "records": len(records),
        "trajectories": len(grouped),
        "labels": {str(key): labels[key] for key in sorted(labels)},
        "negative_trajectories": negative_trajectories,
        "duplicate_records": duplicate_records,
        "fingerprints": fingerprints,
    }


def build_candidate_prefix_records(
    approved: list[dict[str, Any]],
    candidate_by_id: dict[str, dict[str, Any]],
    canonical: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    origins: Counter[str] = Counter()
    benchmarks: Counter[str] = Counter()
    for item in approved:
        candidate_id = str(item["candidate_id"])
        candidate = candidate_by_id.get(candidate_id)
        event = canonical.get(candidate_id)
        if candidate is None or event is None:
            raise RuntimeError("approved candidate is absent from frozen canary")
        if item.get("status") != "candidate_only_not_merged":
            raise RuntimeError("approved candidate status changed")
        result = event.get("result")
        if verification_disposition(result) != "strict_process_negative":
            raise RuntimeError("approved candidate is no longer a strict negative")
        first = item.get("final_human_first_error_step")
        if first != result.get("first_error_step"):
            raise RuntimeError("approved first-error agreement changed")
        steps = (candidate.get("trajectory") or {}).get("steps")
        if not isinstance(steps, list) or not 3 <= len(steps) <= 8:
            raise ValueError("approved candidate step count is invalid")
        if not all(isinstance(step, str) and step.strip() for step in steps):
            raise ValueError("approved candidate contains an invalid step")
        source = _validate_source(candidate.get("source"))
        origins[str(item["origin"])] += 1
        benchmarks[str(source["dataset"])] += 1
        validator_steps = result["steps"]
        if len(validator_steps) != len(steps):
            raise RuntimeError("candidate and validator step counts differ")
        for index, text in enumerate(steps):
            validator_step = validator_steps[index]
            if validator_step.get("index") != index:
                raise RuntimeError("validator step indices changed")
            records.append(
                {
                    "trajectory_id": candidate_id,
                    "step_index": index,
                    "prefix": steps[: index + 1],
                    "label": int(index < first),
                    "error_codes": validator_step["error_codes"],
                    "source": source,
                }
            )
    stats = validate_prm_record_set(records)
    if stats["trajectories"] != len(approved):
        raise RuntimeError("approved candidate trajectories are not distinct")
    if len(set(stats["fingerprints"].values())) != len(approved):
        raise RuntimeError("approved candidates contain duplicate full trajectories")
    stats["origins"] = dict(sorted(origins.items()))
    stats["benchmarks"] = dict(sorted(benchmarks.items()))
    return records, stats


def load_materialization_context(
    config: dict[str, Any], repo_root: Path
) -> dict[str, Any]:
    adjudication_config_path = repo_root / str(
        config["source_adjudication_config"]
    )
    adjudication_config = load_adjudication_config(adjudication_config_path)
    adjudication_context = load_adjudication_context(
        adjudication_config, repo_root
    )
    adjudication_files = adjudication_config["private_files"]
    canary_dir = adjudication_context["canary_dir"]
    adjudication_path = canary_dir / str(adjudication_files["adjudication"])
    adjudication_records = load_annotation_jsonl(adjudication_path)
    metadata, _ = validate_adjudications(
        adjudication_records, adjudication_context, require_complete=True
    )
    adjudication_lock_path = canary_dir / str(
        adjudication_files["adjudication_lock"]
    )
    adjudication_lock = json.loads(
        adjudication_lock_path.read_text(encoding="utf-8")
    )
    validate_adjudication_lock(
        adjudication_lock, adjudication_path, adjudication_context, metadata
    )
    recomputed_audit, recomputed_candidates = score_adjudication(
        adjudication_config_path, repo_root=repo_root
    )
    if not recomputed_audit["quality_gate"]["passed"]:
        raise RuntimeError("source adjudication quality gate is not passed")
    adjudication_audit_path = canary_dir / str(
        adjudication_files["aggregate_audit_json"]
    )
    stored_audit = json.loads(
        adjudication_audit_path.read_text(encoding="utf-8")
    )
    if stored_audit != recomputed_audit:
        raise RuntimeError("stored adjudication audit differs from recomputation")
    approved_path = canary_dir / str(
        adjudication_files["approved_negative_candidates"]
    )
    approved = load_annotation_jsonl(approved_path)
    if approved != recomputed_candidates:
        raise RuntimeError("approved candidate list differs from recomputation")
    expected = config["source_expectations"]
    if len(approved) != expected["approved_negative_trajectories"]:
        raise RuntimeError("approved negative trajectory count changed")

    canary_config_path = repo_root / str(config["source_canary_config"])
    canary_config = yaml.safe_load(canary_config_path.read_text(encoding="utf-8"))
    if not isinstance(canary_config, dict):
        raise ValueError("source canary config must be an object")
    validate_prm_negative_canary_config(canary_config)
    run_dir = repo_root / str(config["source_run_dir"])
    strict_dir = run_dir / str(config["strict_source_subdir"])
    strict_manifest_path = strict_dir / "manifest.json"
    strict_manifest = json.loads(
        strict_manifest_path.read_text(encoding="utf-8")
    )
    if strict_manifest.get("counts") != canary_config["strict_source_expected"]:
        raise RuntimeError("strict source manifest counts changed")
    if strict_manifest.get("source_artifact_sha256") != _source_hashes(run_dir):
        raise RuntimeError("strict source no longer matches original pilot")
    if strict_manifest.get("config_sha256") != sha256_file(canary_config_path):
        raise RuntimeError("strict source config hash changed")
    strict_process_path = strict_dir / "process_train.jsonl"
    strict_sft_path = strict_dir / "sft_verified.jsonl"
    strict_records = load_annotation_jsonl(strict_process_path)
    strict_sft_records = load_annotation_jsonl(strict_sft_path)
    strict_stats = validate_prm_record_set(strict_records)
    expected_labels = {
        "0": expected["strict_negative_prefix_records"],
        "1": expected["strict_positive_prefix_records"],
    }
    if strict_stats["records"] != expected["strict_prm_records"]:
        raise RuntimeError("strict PRM record count changed")
    if strict_stats["labels"] != expected_labels:
        raise RuntimeError("strict PRM label counts changed")
    if len(strict_sft_records) != canary_config["strict_source_expected"][
        "sft_records"
    ]:
        raise RuntimeError("strict SFT record count changed")
    return {
        "adjudication_config": adjudication_config,
        "adjudication_context": adjudication_context,
        "adjudication_lock_path": adjudication_lock_path,
        "adjudication_audit_path": adjudication_audit_path,
        "approved_path": approved_path,
        "approved": approved,
        "run_dir": run_dir,
        "canary_dir": canary_dir,
        "strict_dir": strict_dir,
        "strict_manifest_path": strict_manifest_path,
        "strict_process_path": strict_process_path,
        "strict_sft_path": strict_sft_path,
        "strict_records": strict_records,
        "strict_stats": strict_stats,
        "original_source_hashes": _source_hashes(run_dir),
        "output_dir": run_dir / str(config["output_subdir"]),
    }


def source_binding(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "original_source_artifact_sha256": context["original_source_hashes"],
        "strict_manifest_sha256": sha256_file(context["strict_manifest_path"]),
        "strict_process_train_sha256": sha256_file(context["strict_process_path"]),
        "strict_sft_verified_sha256": sha256_file(context["strict_sft_path"]),
        "adjudication_lock_sha256": sha256_file(
            context["adjudication_lock_path"]
        ),
        "adjudication_audit_sha256": sha256_file(
            context["adjudication_audit_path"]
        ),
        "approved_candidates_sha256": sha256_file(context["approved_path"]),
    }
