"""Normalise the inherited evaluation data into deterministic JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "MedQA_USLME_test": {
        "benchmark": "medqa",
        "split": "test",
        "filename": "medqa_test.jsonl",
    },
    "MedMCQA_validation": {
        "benchmark": "medmcqa",
        "split": "validation",
        "filename": "medmcqa_validation.jsonl",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="evaluation/data/eval_data.json",
        help="Source JSON path, relative to the repository root by default",
    )
    parser.add_argument(
        "--output-dir",
        default="data/benchmark",
        help="Output directory, relative to the repository root by default",
    )
    return parser.parse_args()


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_digest(question: str, choices: dict[str, str]) -> str:
    canonical = json.dumps(
        {"question": question, "choices": choices},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def normalise_record(
    raw: dict[str, Any], source_key: str, source_index: int, dataset: dict[str, str]
) -> dict[str, Any]:
    required = {"question", "options", "answer_idx"}
    missing = required - set(raw)
    if missing:
        raise ValueError(f"{source_key}[{source_index}] missing fields {sorted(missing)}")

    question = str(raw["question"]).strip()
    if not question:
        raise ValueError(f"{source_key}[{source_index}] has an empty question")
    if not isinstance(raw["options"], dict) or len(raw["options"]) < 2:
        raise ValueError(f"{source_key}[{source_index}] has invalid options")

    choices = {
        str(label).strip().upper(): str(text).strip()
        for label, text in raw["options"].items()
    }
    if any(len(label) != 1 or not label.isalpha() for label in choices):
        raise ValueError(f"{source_key}[{source_index}] has invalid option labels")
    answer = str(raw["answer_idx"]).strip().upper()
    if answer not in choices:
        raise ValueError(f"{source_key}[{source_index}] answer is not in options")

    digest = _content_digest(question, choices)
    benchmark = dataset["benchmark"]
    split = dataset["split"]
    return {
        "id": f"{benchmark}_{split}_{source_index:06d}_{digest[:10]}",
        "benchmark": benchmark,
        "split": split,
        "question": question,
        "choices": choices,
        "answer": answer,
        "source_key": source_key,
        "source_index": source_index,
        "content_sha256": digest,
    }


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    source_path = _repo_path(args.source).resolve()
    output_dir = _repo_path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with source_path.open("r", encoding="utf-8") as handle:
        source_data = json.load(handle)
    if not isinstance(source_data, dict):
        raise ValueError("source evaluation data must be a JSON object")

    all_records: list[dict[str, Any]] = []
    dataset_manifest: list[dict[str, Any]] = []
    content_locations: dict[str, list[str]] = {}

    for source_key, dataset in DATASETS.items():
        raw_records = source_data.get(source_key)
        if not isinstance(raw_records, list):
            raise ValueError(f"source dataset is missing or invalid: {source_key}")
        records = [
            normalise_record(raw, source_key, index, dataset)
            for index, raw in enumerate(raw_records)
        ]
        output_path = output_dir / dataset["filename"]
        _write_jsonl(output_path, records)
        all_records.extend(records)

        for record in records:
            content_locations.setdefault(record["content_sha256"], []).append(record["id"])
        dataset_manifest.append(
            {
                "benchmark": dataset["benchmark"],
                "split": dataset["split"],
                "source_key": source_key,
                "records": len(records),
                "output_file": output_path.relative_to(REPO_ROOT).as_posix(),
                "output_sha256": sha256_file(output_path),
            }
        )

    combined_path = output_dir / "medical_mcq_eval.jsonl"
    _write_jsonl(combined_path, all_records)
    duplicate_groups = [ids for ids in content_locations.values() if len(ids) > 1]
    manifest = {
        "schema_version": 1,
        "provenance_status": "inherited_unverified",
        "provenance_note": (
            "Records were inherited from the HuatuoGPT-o1 evaluation snapshot. "
            "Verify the original dataset revisions and licenses before redistribution."
        ),
        "source_file": source_path.relative_to(REPO_ROOT).as_posix(),
        "source_sha256": sha256_file(source_path),
        "datasets": dataset_manifest,
        "combined_file": combined_path.relative_to(REPO_ROOT).as_posix(),
        "combined_records": len(all_records),
        "combined_sha256": sha256_file(combined_path),
        "duplicate_content_groups": duplicate_groups,
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
