"""Build an isolated strict-contract view of the immutable 40-question pilot."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import yaml

from data_pipeline.cot_api import validate_validator_result
from data_pipeline.prm_negative_canary_config import (
    validate_prm_negative_canary_config,
)
from data_pipeline.run_validator_recovery import SOURCE_FILES


REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_FIELDS = (
    "trajectory_label",
    "first_error_step",
    "answer_consistent",
    "problem_status",
    "steps",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cot/prm_negative_canary_v1.yaml")
    return parser.parse_args()


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hashes(run_dir: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in SOURCE_FILES:
        path = run_dir / name
        if not path.is_file():
            raise RuntimeError(f"strict pilot source is missing: {name}")
        result[name] = _sha256(path)
    return result


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected one JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            records.append(value)
    return records


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def strict_records(
    canonical: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    strict: list[dict[str, Any]] = []
    excluded: Counter[str] = Counter()
    for record in canonical:
        verification = record.get("verification")
        trajectory = record.get("trajectory") or {}
        trajectory_steps = trajectory.get("steps") or []
        if not isinstance(verification, dict) or not isinstance(trajectory_steps, list):
            excluded["invalid_shape"] += 1
            continue
        payload = {field: verification.get(field) for field in VALIDATOR_FIELDS}
        try:
            validate_validator_result(payload, len(trajectory_steps))
        except (KeyError, TypeError, ValueError):
            excluded["validator_contract_invalid"] += 1
            continue
        strict.append(record)

    sft = [
        record
        for record in strict
        if record["verification"]["trajectory_label"] == 1
        and record.get("disposition") == "sft_accept"
    ]
    prm: list[dict[str, Any]] = []
    for record in strict:
        for step in record["verification"]["steps"]:
            index = int(step["index"])
            prm.append(
                {
                    "trajectory_id": record["trajectory_id"],
                    "step_index": index,
                    "prefix": [
                        value["text"]
                        for value in record["trajectory"]["steps"][: index + 1]
                    ],
                    "label": int(step["prefix_label"]),
                    "error_codes": step["error_codes"],
                    "source": record["source"],
                }
            )
    return strict, sft, prm, dict(sorted(excluded.items()))


def build_strict_view(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("strict-view config must be an object")
    validate_prm_negative_canary_config(config)
    run_dir = _repo_path(str(config["source_run_dir"]))
    hashes_before = _source_hashes(run_dir)
    metadata = _load_json(run_dir / "metadata.json")
    identity = config["source_identity"]
    for field in ("questions", "teacher_events", "canonical_trajectories", "prm_records"):
        if int(metadata.get(field, -1)) != int(identity[field]):
            raise RuntimeError(f"strict-view source identity mismatch: {field}")
    canonical = _load_jsonl(run_dir / "canonical_trajectories.jsonl")
    strict, sft, prm, excluded = strict_records(canonical)
    actual = {
        "canonical_trajectories": len(strict),
        "sft_records": len(sft),
        "prm_records": len(prm),
    }
    if actual != config["strict_source_expected"]:
        raise RuntimeError(f"strict-view counts changed: {actual}")
    if any(type(record["label"]) is not int for record in prm):
        raise RuntimeError("strict-view PRM contains a non-integer label")
    output_dir = run_dir / str(config["strict_source_subdir"])
    _write_jsonl(output_dir / "canonical_trajectories.jsonl", strict)
    _write_jsonl(output_dir / "sft_verified.jsonl", sft)
    _write_jsonl(output_dir / "process_train.jsonl", prm)
    if _source_hashes(run_dir) != hashes_before:
        raise RuntimeError("an immutable pilot artifact changed during strict derivation")
    manifest = {
        "schema_version": "medtrace.strict-pilot-view.v1",
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contains_private_text_or_ids": False,
        "config_sha256": _sha256(config_path),
        "source_artifact_sha256": hashes_before,
        "source_artifacts_unchanged": True,
        "counts": actual,
        "excluded": excluded,
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    args = parse_args()
    manifest = build_strict_view(_repo_path(args.config))
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
