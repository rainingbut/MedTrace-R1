"""Download, pin, normalise, and isolate private stage-2 training sources."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable, Iterable
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

import yaml

from data_pipeline.cot_config import validate_pilot_config
from data_pipeline.cot_isolation import (
    EvaluationIsolationIndex,
    content_sha256,
    validate_train_record,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
HF_API = "https://huggingface.co/api/datasets"
HF_DATASETS = "https://huggingface.co/datasets"
OPTION_LABELS = "ABCD"
SOURCE_FILES = {
    "medqa": "data/train-00000-of-00001.parquet",
    "medmcqa": "data/train-00000-of-00001.parquet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cot/pilot_v1.yaml")
    parser.add_argument("--raw-dir", default="data/source/raw")
    parser.add_argument("--normalized-dir", default="data/source/normalized")
    parser.add_argument(
        "--evaluation-file", default="data/benchmark/medical_mcq_eval.jsonl"
    )
    parser.add_argument("--manifest", default="data/source/stage2_train_manifest.json")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument(
        "--discover-source-hashes",
        action="store_true",
        help="download pinned files and print hashes, but do not normalise data",
    )
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


def _get_json(url: str, timeout: float, max_retries: int) -> dict[str, Any]:
    request = urllib_request.Request(url, headers={"User-Agent": "MEDTRACE-R1/0.2"})
    for attempt in range(max_retries + 1):
        try:
            with urllib_request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (OSError, json.JSONDecodeError, urllib_error.HTTPError) as exc:
            if attempt == max_retries:
                raise RuntimeError(f"failed to retrieve {url}: {exc}") from exc
            time.sleep(min(2**attempt, 8))
    raise AssertionError("retry loop ended unexpectedly")


def _download(url: str, destination: Path, timeout: float, max_retries: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    request = urllib_request.Request(url, headers={"User-Agent": "MEDTRACE-R1/0.2"})
    for attempt in range(max_retries + 1):
        try:
            with urllib_request.urlopen(request, timeout=timeout) as response:
                with temporary.open("wb") as handle:
                    while chunk := response.read(1024 * 1024):
                        handle.write(chunk)
            temporary.replace(destination)
            return
        except (OSError, urllib_error.HTTPError) as exc:
            temporary.unlink(missing_ok=True)
            if attempt == max_retries:
                raise RuntimeError(f"failed to retrieve {url}: {exc}") from exc
            time.sleep(min(2**attempt, 8))
    raise AssertionError("retry loop ended unexpectedly")


def _normalise_medqa(row: dict[str, Any]) -> dict[str, Any]:
    sent1 = str(row["sent1"]).strip()
    sent2 = str(row.get("sent2") or "").strip()
    question = f"{sent1} {sent2}".strip() if sent2 else sent1
    return {
        "source_id": str(row["id"]),
        "question": question,
        "choices": {
            label: str(row[f"ending{index}"]).strip()
            for index, label in enumerate(OPTION_LABELS)
        },
        "answer": OPTION_LABELS[int(row["label"])],
    }


def _normalise_medmcqa(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_id": str(row["id"]),
        "question": str(row["question"]).strip(),
        "choices": {
            "A": str(row["opa"]).strip(),
            "B": str(row["opb"]).strip(),
            "C": str(row["opc"]).strip(),
            "D": str(row["opd"]).strip(),
        },
        "answer": OPTION_LABELS[int(row["cop"])],
        "subject": str(row.get("subject_name") or "").strip() or None,
        "topic": str(row.get("topic_name") or "").strip() or None,
        "choice_type": str(row.get("choice_type") or "").strip() or None,
        "reference_explanation_present": bool(str(row.get("exp") or "").strip()),
    }


NORMALISERS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "medqa": _normalise_medqa,
    "medmcqa": _normalise_medmcqa,
}


def _load_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required; activate the project environment and install requirements.txt"
        ) from exc
    return parquet.read_table(path).to_pylist()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
    if not records:
        raise ValueError(f"no evaluation records found in {path}")
    return records


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


def _source_url(repository: str, revision: str, data_file: str) -> str:
    repo = urllib_parse.quote(repository, safe="/")
    revision = urllib_parse.quote(revision, safe="")
    data_file = urllib_parse.quote(data_file, safe="/")
    return f"{HF_DATASETS}/{repo}/resolve/{revision}/{data_file}"


def _download_sources(
    config: dict[str, Any], raw_dir: Path, timeout: float, max_retries: int
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for benchmark, source in config["sources"].items():
        repository = str(source["repository"])
        revision = str(source["revision"])
        repo_id = urllib_parse.quote(repository, safe="/")
        repo_revision = urllib_parse.quote(revision, safe="")
        info = _get_json(
            f"{HF_API}/{repo_id}/revision/{repo_revision}", timeout, max_retries
        )
        if info.get("sha") != revision:
            raise ValueError(
                f"{benchmark} resolved revision changed: {info.get('sha')} != {revision}"
            )
        data_file = SOURCE_FILES[benchmark]
        destination = raw_dir / f"{benchmark}-{revision[:12]}-train.parquet"
        if not destination.exists():
            print(f"{benchmark}: downloading pinned train Parquet", flush=True)
            _download(
                _source_url(repository, revision, data_file),
                destination,
                timeout,
                max_retries,
            )
        digest = _sha256(destination)
        expected = str(source["source_file_sha256"])
        if not expected.startswith("REPLACE_") and digest != expected:
            raise ValueError(f"{benchmark} source hash mismatch: {digest} != {expected}")
        reports.append(
            {
                "benchmark": benchmark,
                "repository": repository,
                "revision": revision,
                "split": "train",
                "data_file": data_file,
                "local_file": destination,
                "source_file_sha256": digest,
                "expected_records": int(source["expected_records"]),
            }
        )
    return reports


def _normalise_source(
    report: dict[str, Any],
    evaluation_index: EvaluationIsolationIndex,
    seen_content: set[str],
    normalized_dir: Path,
) -> tuple[dict[str, Any], dict[str, int], list[dict[str, Any]]]:
    benchmark = str(report["benchmark"])
    raw_rows = _load_parquet(report["local_file"])
    expected_rows = int(report["expected_records"])
    if len(raw_rows) != expected_rows:
        raise ValueError(f"{benchmark} row count mismatch: {len(raw_rows)} != {expected_rows}")

    summary = {
        "input_records": len(raw_rows),
        "accepted_records": 0,
        "rejected_non_train": 0,
        "rejected_exact_overlap": 0,
        "rejected_normalized_overlap": 0,
        "rejected_near_overlap": 0,
        "rejected_duplicate_within_train": 0,
    }
    accepted: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    normalise = NORMALISERS[benchmark]
    for source_index, raw in enumerate(raw_rows):
        value = normalise(raw)
        digest = content_sha256(value["question"], value["choices"])
        record = {
            "id": f"{benchmark}_train_{source_index:06d}_{digest[:10]}",
            "benchmark": benchmark,
            "split": "train",
            "question": value.pop("question"),
            "choices": value.pop("choices"),
            "answer": value.pop("answer"),
            "source_id": value.pop("source_id"),
            "source_index": source_index,
            "source_revision": report["revision"],
            "source_file_sha256": report["source_file_sha256"],
            "content_sha256": digest,
            **value,
        }
        validate_train_record(record)
        if digest in seen_content:
            summary["rejected_duplicate_within_train"] += 1
            rejections.append(
                {
                    "benchmark": benchmark,
                    "source_id": record["source_id"],
                    "source_index": source_index,
                    "content_sha256": digest,
                    "reason": "duplicate_within_train",
                }
            )
            continue
        overlap = evaluation_index.find_overlap(record)
        if overlap is not None:
            overlap_counter = {
                "exact_content": "rejected_exact_overlap",
                "normalised_question": "rejected_normalized_overlap",
                "near_question": "rejected_near_overlap",
            }[overlap.kind]
            summary[overlap_counter] += 1
            rejections.append(
                {
                    "benchmark": benchmark,
                    "source_id": record["source_id"],
                    "source_index": source_index,
                    "content_sha256": digest,
                    "reason": overlap.kind,
                    "evaluation_id": overlap.evaluation_id,
                    "similarity": round(overlap.score, 6),
                }
            )
            continue
        seen_content.add(digest)
        accepted.append(record)
    summary["accepted_records"] = len(accepted)

    output = normalized_dir / f"{benchmark}_train.jsonl"
    _write_jsonl(output, accepted)
    source_manifest = {
        "benchmark": benchmark,
        "repository": report["repository"],
        "revision": report["revision"],
        "split": "train",
        "source_file": report["local_file"].relative_to(REPO_ROOT).as_posix(),
        "source_file_sha256": report["source_file_sha256"],
        "records": len(raw_rows),
        "normalized_file": output.relative_to(REPO_ROOT).as_posix(),
        "normalized_file_sha256": _sha256(output),
    }
    return source_manifest, summary, rejections


def main() -> None:
    args = parse_args()
    config_path = _repo_path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    validate_pilot_config(config)

    reports = _download_sources(
        config,
        _repo_path(args.raw_dir),
        args.timeout,
        args.max_retries,
    )
    discovery = [
        {
            key: value
            for key, value in report.items()
            if key not in {"local_file"}
        }
        | {"local_file": report["local_file"].relative_to(REPO_ROOT).as_posix()}
        for report in reports
    ]
    if args.discover_source_hashes:
        print(json.dumps(discovery, ensure_ascii=False, indent=2))
        return

    for report, source in zip(reports, config["sources"].values()):
        if str(source["source_file_sha256"]).startswith("REPLACE_"):
            raise ValueError(
                f"{report['benchmark']} hash is not frozen in {config_path}; "
                "run --discover-source-hashes, review, then update the config"
            )

    evaluation_file = _repo_path(args.evaluation_file)
    evaluation_records = _load_jsonl(evaluation_file)
    evaluation_index = EvaluationIsolationIndex(evaluation_records)
    seen_content: set[str] = set()
    source_manifests: list[dict[str, Any]] = []
    summaries: list[dict[str, int]] = []
    all_rejections: list[dict[str, Any]] = []
    for report in reports:
        source_manifest, summary, rejections = _normalise_source(
            report,
            evaluation_index,
            seen_content,
            _repo_path(args.normalized_dir),
        )
        source_manifests.append(source_manifest)
        summaries.append(summary)
        all_rejections.extend(rejections)
        print(
            f"{report['benchmark']}: accepted {summary['accepted_records']}/"
            f"{summary['input_records']}",
            flush=True,
        )

    combined_summary = {
        key: sum(summary[key] for summary in summaries) for key in summaries[0]
    }
    rejection_path = _repo_path(config["outputs"]["isolation_rejections"])
    _write_jsonl(rejection_path, all_rejections)
    manifest = {
        "schema_version": "medtrace.training-source.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "privacy_boundary": config["privacy_boundary"],
        "sources": source_manifests,
        "evaluation_index": {
            "files": [evaluation_file.relative_to(REPO_ROOT).as_posix()],
            "combined_sha256": _sha256(evaluation_file),
            "records": len(evaluation_records),
        },
        "isolation_summary": combined_summary,
        "isolation_report": {
            "file": rejection_path.relative_to(REPO_ROOT).as_posix(),
            "sha256": _sha256(rejection_path),
            "records": len(all_rejections),
        },
    }
    manifest_path = _repo_path(args.manifest)
    _write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Private manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()
