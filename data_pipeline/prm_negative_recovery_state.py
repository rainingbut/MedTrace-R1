"""Shared source and canonical-state helpers for PRM validator recovery."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from data_pipeline.build_strict_pilot_view import _source_hashes
from data_pipeline.cot_api import validate_validator_result
from data_pipeline.run_cot_pilot_real import _load_jsonl


SOURCE_CANARY_FILES = (
    "candidates.jsonl",
    "validator_events.jsonl",
    "metadata.json",
    "private_manifest.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_canary_hashes(output_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in SOURCE_CANARY_FILES:
        path = output_dir / name
        if not path.is_file():
            raise RuntimeError(f"PRM recovery source file is missing: {name}")
        hashes[name] = sha256_file(path)
    return hashes


def load_source_state(
    config: dict[str, Any], repo_root: Path
) -> dict[str, Any]:
    run_dir = repo_root / str(config["source_run_dir"])
    canary_dir = run_dir / str(config["source_canary_subdir"])
    candidates = _load_jsonl(canary_dir / "candidates.jsonl")
    events = _load_jsonl(canary_dir / "validator_events.jsonl")
    metadata = json.loads((canary_dir / "metadata.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (canary_dir / "private_manifest.json").read_text(encoding="utf-8")
    )
    candidate_by_id = {
        str(candidate["candidate_id"]): candidate for candidate in candidates
    }
    event_by_id = {str(event["candidate_id"]): event for event in events}
    expected = config["source_expectations"]
    if metadata.get("status") != "complete":
        raise RuntimeError("source PRM canary is not complete")
    if len(candidates) != int(expected["candidates"]):
        raise RuntimeError("source PRM candidate count changed")
    if len(events) != int(expected["validator_events"]):
        raise RuntimeError("source PRM validator count changed")
    if len(candidate_by_id) != len(candidates) or len(event_by_id) != len(events):
        raise RuntimeError("source PRM canary contains duplicate candidate IDs")
    if set(candidate_by_id) != set(event_by_id):
        raise RuntimeError("source PRM validator keys differ from candidates")
    if _source_hashes(run_dir) != manifest.get("source_artifact_sha256"):
        raise RuntimeError("original pilot artifacts changed after PRM canary")

    strict_valid = 0
    unavailable: list[str] = []
    for candidate_id, candidate in candidate_by_id.items():
        event = event_by_id[candidate_id]
        if event.get("status") == "complete":
            validate_validator_result(
                event.get("result"), len(candidate["trajectory"]["steps"])
            )
            strict_valid += 1
            continue
        if event.get("status") != expected["unavailable_event_status"]:
            raise RuntimeError("source unavailable status changed")
        if event.get("result") is not None:
            raise RuntimeError("source unavailable event unexpectedly has a result")
        diagnostics = event.get("attempt_diagnostics") or []
        if not diagnostics:
            raise RuntimeError("source unavailable event lacks diagnostics")
        if any(
            item.get("error_category") != expected["unavailable_error_category"]
            or item.get("content_present")
            is not expected["unavailable_content_present"]
            for item in diagnostics
        ):
            raise RuntimeError("source unavailable failure signature changed")
        unavailable.append(candidate_id)
    if strict_valid != int(expected["strict_contract_valid"]):
        raise RuntimeError("source strict-contract count changed")
    if len(unavailable) != int(expected["unavailable"]):
        raise RuntimeError("source unavailable count changed")
    ordered_unavailable = [
        str(candidate["candidate_id"])
        for candidate in candidates
        if str(candidate["candidate_id"]) in set(unavailable)
    ]
    return {
        "run_dir": run_dir,
        "canary_dir": canary_dir,
        "candidates": candidates,
        "candidate_by_id": candidate_by_id,
        "source_events": events,
        "source_event_by_id": event_by_id,
        "selected_ids": ordered_unavailable,
        "source_canary_hashes": source_canary_hashes(canary_dir),
    }


def recovery_attempts_by_candidate(
    attempts: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    seen: set[tuple[str, int]] = set()
    for event in attempts:
        candidate_id = str(event["candidate_id"])
        attempt_number = int(event["recovery_attempt"])
        key = (candidate_id, attempt_number)
        if key in seen:
            raise RuntimeError("duplicate PRM recovery attempt key")
        seen.add(key)
        grouped.setdefault(candidate_id, []).append(event)
    for values in grouped.values():
        values.sort(key=lambda event: int(event["recovery_attempt"]))
    return grouped


def canonical_event_map(
    source_state: dict[str, Any], attempts: list[dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    grouped = recovery_attempts_by_candidate(attempts)
    canonical: dict[str, dict[str, Any]] = {}
    provenance: dict[str, str] = {}
    for candidate_id, source in source_state["source_event_by_id"].items():
        if source.get("status") == "complete":
            canonical[candidate_id] = source
            provenance[candidate_id] = "source_canary"
            continue
        completed = [
            event for event in grouped.get(candidate_id, [])
            if event.get("status") == "complete"
        ]
        if completed:
            canonical[candidate_id] = completed[0]
            provenance[candidate_id] = "validator_recovery_v1"
        else:
            canonical[candidate_id] = source
            provenance[candidate_id] = "source_unavailable"
    return canonical, provenance
