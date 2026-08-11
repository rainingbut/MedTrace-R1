"""Select the deterministic, private 40-question stage-2 pilot input."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import yaml

from data_pipeline.cot_config import validate_pilot_config
from data_pipeline.cot_isolation import validate_train_record


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cot/pilot_v1.yaml")
    parser.add_argument("--source-manifest", default=None)
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


def _stable_rank(seed: int, *values: object) -> str:
    value = "\x1f".join([str(seed), *(str(item) for item in values)])
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            validate_train_record(record)
            records.append(record)
    return records


def _length_buckets(records: list[dict[str, Any]]) -> dict[str, int]:
    ordered = sorted(records, key=lambda record: (len(record["question"]), record["id"]))
    count = len(ordered)
    return {
        str(record["id"]): min(3, index * 4 // count)
        for index, record in enumerate(ordered)
    }


def select_medqa(
    records: list[dict[str, Any]], count: int, seed: int
) -> list[dict[str, Any]]:
    if count % 4:
        raise ValueError("MedQA pilot count must be divisible by four")
    buckets = _length_buckets(records)
    selected: list[dict[str, Any]] = []
    per_bucket = count // 4
    for bucket in range(4):
        candidates = [record for record in records if buckets[record["id"]] == bucket]
        candidates.sort(key=lambda record: _stable_rank(seed, "medqa", record["id"]))
        selected.extend(candidates[:per_bucket])
    if len(selected) != count:
        raise ValueError(f"unable to select {count} stratified MedQA records")
    return selected


def select_medmcqa(
    records: list[dict[str, Any]], count: int, seed: int
) -> list[dict[str, Any]]:
    by_subject: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        subject = str(record.get("subject") or "").strip()
        if not subject or subject.casefold() == "unknown":
            continue
        by_subject.setdefault(subject, []).append(record)
    if len(by_subject) < count:
        raise ValueError("not enough MedMCQA subjects for subject-stratified pilot")

    subjects = sorted(
        by_subject,
        key=lambda subject: _stable_rank(seed, "medmcqa-subject", subject),
    )[:count]
    buckets = _length_buckets(records)
    selected: list[dict[str, Any]] = []
    for index, subject in enumerate(subjects):
        target_bucket = index % 4
        candidates = sorted(
            by_subject[subject],
            key=lambda record: (
                abs(buckets[record["id"]] - target_bucket),
                _stable_rank(seed, "medmcqa", subject, record["id"]),
            ),
        )
        selected.append(candidates[0])
    return selected


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    config_path = _repo_path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    validate_pilot_config(config)

    source_manifest_path = _repo_path(
        args.source_manifest or config["outputs"]["source_manifest"]
    )
    with source_manifest_path.open("r", encoding="utf-8") as handle:
        source_manifest = json.load(handle)
    sources = {source["benchmark"]: source for source in source_manifest["sources"]}
    records = {
        benchmark: _load_jsonl(_repo_path(source["normalized_file"]))
        for benchmark, source in sources.items()
    }

    sampling = config["sampling"]
    seed = int(sampling["seed"])
    counts = sampling["questions_per_benchmark"]
    selected = select_medqa(records["medqa"], int(counts["medqa"]), seed)
    selected.extend(
        select_medmcqa(records["medmcqa"], int(counts["medmcqa"]), seed)
    )
    selected.sort(key=lambda record: (record["benchmark"], record["id"]))
    expected_total = int(sampling["total_questions"])
    if len(selected) != expected_total:
        raise ValueError(f"pilot selection count mismatch: {len(selected)} != {expected_total}")
    if len({record["id"] for record in selected}) != len(selected):
        raise ValueError("pilot selection contains duplicate ids")

    output_path = _repo_path(config["outputs"]["pilot_questions"])
    _write_jsonl(output_path, selected)
    medqa_buckets = _length_buckets(records["medqa"])
    medmcqa_buckets = _length_buckets(records["medmcqa"])
    manifest = {
        "schema_version": "medtrace.cot-pilot-input.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_file": config_path.relative_to(REPO_ROOT).as_posix(),
        "config_sha256": _sha256(config_path),
        "source_manifest_file": source_manifest_path.relative_to(REPO_ROOT).as_posix(),
        "source_manifest_sha256": _sha256(source_manifest_path),
        "sampling_seed": seed,
        "sampling_method": "length_quartiles_medqa_subject_and_length_medmcqa_v1",
        "questions_file": output_path.relative_to(REPO_ROOT).as_posix(),
        "questions_sha256": _sha256(output_path),
        "records": len(selected),
        "records_per_benchmark": {
            benchmark: sum(record["benchmark"] == benchmark for record in selected)
            for benchmark in ("medqa", "medmcqa")
        },
        "length_quartiles": {
            "medqa": [
                sum(
                    record["benchmark"] == "medqa"
                    and medqa_buckets[record["id"]] == bucket
                    for record in selected
                )
                for bucket in range(4)
            ],
            "medmcqa": [
                sum(
                    record["benchmark"] == "medmcqa"
                    and medmcqa_buckets[record["id"]] == bucket
                    for record in selected
                )
                for bucket in range(4)
            ],
        },
        "medmcqa_subjects": sorted(
            str(record.get("subject"))
            for record in selected
            if record["benchmark"] == "medmcqa"
        ),
    }
    manifest_path = _repo_path(config["outputs"]["pilot_input_manifest"])
    _write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
