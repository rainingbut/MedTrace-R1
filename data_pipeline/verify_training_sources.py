"""Read-only verification of private stage-2 training sources and isolation output."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

import yaml

from data_pipeline.cot_isolation import EvaluationIsolationIndex, validate_train_record


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cot/pilot_v1.yaml")
    parser.add_argument("--manifest", default="data/source/stage2_train_manifest.json")
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


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc


def _require_hash(path: Path, expected: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"hash mismatch for {path}: {actual} != {expected}")


def main() -> None:
    args = parse_args()
    config_path = _repo_path(args.config)
    manifest_path = _repo_path(args.manifest)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    if manifest.get("schema_version") != "medtrace.training-source.v1":
        raise ValueError("unsupported training-source manifest")
    if manifest.get("privacy_boundary") != config.get("privacy_boundary"):
        raise ValueError("manifest privacy boundary differs from pilot config")

    evaluation = manifest["evaluation_index"]
    evaluation_files = [_repo_path(value) for value in evaluation["files"]]
    if len(evaluation_files) != 1:
        raise ValueError("v1 verifier requires one combined evaluation JSONL")
    _require_hash(evaluation_files[0], evaluation["combined_sha256"])
    evaluation_records = list(_iter_jsonl(evaluation_files[0]))
    if len(evaluation_records) != int(evaluation["records"]):
        raise ValueError("evaluation record count differs from manifest")
    evaluation_index = EvaluationIsolationIndex(evaluation_records)

    config_sources = config["sources"]
    seen_ids: set[str] = set()
    seen_content: set[str] = set()
    accepted_by_benchmark: dict[str, int] = {}
    raw_records = 0
    for source in manifest["sources"]:
        benchmark = source["benchmark"]
        expected = config_sources[benchmark]
        if source["split"] != "train" or expected["split"] != "train":
            raise ValueError(f"non-train source found for {benchmark}")
        if source["revision"] != expected["revision"]:
            raise ValueError(f"revision mismatch for {benchmark}")
        if source["source_file_sha256"] != expected["source_file_sha256"]:
            raise ValueError(f"source hash differs from config for {benchmark}")
        raw_path = _repo_path(source["source_file"])
        normalized_path = _repo_path(source["normalized_file"])
        _require_hash(raw_path, source["source_file_sha256"])
        _require_hash(normalized_path, source["normalized_file_sha256"])
        raw_records += int(source["records"])

        accepted = 0
        for record in _iter_jsonl(normalized_path):
            validate_train_record(record, evaluation_index)
            record_id = str(record["id"])
            digest = str(record["content_sha256"])
            if record_id in seen_ids:
                raise ValueError(f"duplicate normalized id: {record_id}")
            if digest in seen_content:
                raise ValueError(f"duplicate normalized content: {digest}")
            seen_ids.add(record_id)
            seen_content.add(digest)
            accepted += 1
        accepted_by_benchmark[benchmark] = accepted

    isolation_report = manifest["isolation_report"]
    rejection_path = _repo_path(isolation_report["file"])
    _require_hash(rejection_path, isolation_report["sha256"])
    rejections = list(_iter_jsonl(rejection_path))
    if len(rejections) != int(isolation_report["records"]):
        raise ValueError("rejection report count differs from manifest")

    summary = manifest["isolation_summary"]
    accepted_total = sum(accepted_by_benchmark.values())
    rejected_total = sum(
        int(value) for key, value in summary.items() if key.startswith("rejected_")
    )
    if raw_records != int(summary["input_records"]):
        raise ValueError("raw source count differs from isolation summary")
    if accepted_total != int(summary["accepted_records"]):
        raise ValueError("normalized count differs from isolation summary")
    if rejected_total != len(rejections):
        raise ValueError("rejection count differs from isolation summary")
    if raw_records != accepted_total + rejected_total:
        raise ValueError("accepted and rejected records do not partition all inputs")

    result = {
        "status": "verified",
        "raw_records": raw_records,
        "accepted_records": accepted_total,
        "rejected_records": rejected_total,
        "unique_ids": len(seen_ids),
        "unique_content_sha256": len(seen_content),
        "accepted_by_benchmark": accepted_by_benchmark,
        "evaluation_records": len(evaluation_records),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
