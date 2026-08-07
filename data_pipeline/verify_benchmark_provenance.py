"""Verify every inherited benchmark row against pinned Hugging Face sources."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request


REPO_ROOT = Path(__file__).resolve().parents[1]
HF_API = "https://huggingface.co/api/datasets"
HF_DATASETS = "https://huggingface.co/datasets"
OPTION_LABELS = "ABCD"


def _normalise_medqa(row: dict[str, Any]) -> dict[str, Any]:
    sent1 = str(row["sent1"]).strip()
    sent2 = str(row.get("sent2") or "").strip()
    question = f"{sent1} {sent2}".strip() if sent2 else sent1
    choices = {
        label: str(row[f"ending{index}"]).strip()
        for index, label in enumerate(OPTION_LABELS)
    }
    return {
        "question": question,
        "choices": choices,
        "answer": OPTION_LABELS[int(row["label"])],
    }


def _normalise_medmcqa(row: dict[str, Any]) -> dict[str, Any]:
    choices = {
        "A": str(row["opa"]).strip(),
        "B": str(row["opb"]).strip(),
        "C": str(row["opc"]).strip(),
        "D": str(row["opd"]).strip(),
    }
    return {
        "question": str(row["question"]).strip(),
        "choices": choices,
        "answer": OPTION_LABELS[int(row["cop"])],
    }


SOURCES: tuple[dict[str, Any], ...] = (
    {
        "benchmark": "medqa",
        "local_key": "MedQA_USLME_test",
        "repo_id": "openlifescienceai/MedQA-USMLE-4-options-hf",
        "revision": "20a8f4d6b851f6391751f6e76c06bc3a26c83e0b",
        "config": "default",
        "split": "test",
        "rows": 1273,
        "data_file": "data/test-00000-of-00001.parquet",
        "file_sha256": "1177dd34cb298cfe4f4a286797832a41e397ffde6847009203a8d1a0914327f5",
        "normalise": _normalise_medqa,
        "license": {
            "hf_dataset_card": "not_declared",
            "original_repository": "MIT",
            "original_repository_url": "https://github.com/jind11/MedQA",
            "review_status": "manual_review_required",
        },
    },
    {
        "benchmark": "medmcqa",
        "local_key": "MedMCQA_validation",
        "repo_id": "openlifescienceai/medmcqa",
        "revision": "91c6572c454088bf71b679ad90aa8dffcd0d5868",
        "config": "default",
        "split": "validation",
        "rows": 4183,
        "data_file": "data/validation-00000-of-00001.parquet",
        "file_sha256": "b768a1ea34afc9f80d3106d9b21f80fa8a00ec450a1f6cd641af72ca9e591021",
        "normalise": _normalise_medmcqa,
        "license": {
            "hf_dataset_card": "Apache-2.0",
            "original_repository": "MIT",
            "original_repository_url": "https://github.com/medmcqa/medmcqa",
            "review_status": "conflicting_metadata_manual_review_required",
        },
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="evaluation/data/eval_data.json")
    parser.add_argument("--output", default="data/benchmark/provenance.json")
    parser.add_argument("--cache-dir", default=".cache/provenance")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument(
        "--update-report",
        action="store_true",
        help="replace the committed report after an intentional provenance review",
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
    request = urllib_request.Request(
        url,
        headers={
            "User-Agent": "MEDTRACE-R1/0.1",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        },
    )
    for attempt in range(max_retries + 1):
        try:
            with urllib_request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (OSError, json.JSONDecodeError, urllib_error.HTTPError) as exc:
            if attempt == max_retries:
                raise RuntimeError(f"failed to retrieve {url}: {exc}") from exc
            headers = getattr(exc, "headers", None)
            retry_after = headers.get("Retry-After") if headers else None
            delay = float(retry_after) if retry_after else min(2**attempt, 8)
            time.sleep(delay)
    raise AssertionError("retry loop ended unexpectedly")


def _cached_file_matches(path: Path, expected_sha256: str) -> bool:
    return path.is_file() and _sha256(path) == expected_sha256


def _download_file(
    url: str,
    destination: Path,
    timeout: float,
    max_retries: int,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    request = urllib_request.Request(
        url,
        headers={
            "User-Agent": "MEDTRACE-R1/0.1",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        },
    )
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
            headers = getattr(exc, "headers", None)
            retry_after = headers.get("Retry-After") if headers else None
            delay = float(retry_after) if retry_after else min(2**attempt, 8)
            time.sleep(delay)
    raise AssertionError("retry loop ended unexpectedly")


def _load_pinned_rows(
    source: dict[str, Any], args: argparse.Namespace
) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required for provenance verification; install requirements.txt"
        ) from exc

    cache_dir = _repo_path(args.cache_dir).resolve()
    cache_name = (
        f"{source['benchmark']}-{source['revision'][:12]}-"
        f"{Path(source['data_file']).name}"
    )
    cached_file = cache_dir / cache_name
    expected_sha256 = str(source["file_sha256"])
    if not _cached_file_matches(cached_file, expected_sha256):
        repo_id = urllib_parse.quote(str(source["repo_id"]), safe="/")
        revision = urllib_parse.quote(str(source["revision"]), safe="")
        data_file = urllib_parse.quote(str(source["data_file"]), safe="/")
        url = f"{HF_DATASETS}/{repo_id}/resolve/{revision}/{data_file}"
        print(f"{source['benchmark']}: downloading pinned Parquet file", flush=True)
        _download_file(url, cached_file, args.timeout, args.max_retries)
    actual_sha256 = _sha256(cached_file)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{source['repo_id']} file hash changed: "
            f"{actual_sha256} != {expected_sha256}"
        )

    rows = parquet.read_table(cached_file).to_pylist()
    expected_rows = int(source["rows"])
    if len(rows) != expected_rows:
        raise ValueError(
            f"{source['repo_id']} row count changed: {len(rows)} != {expected_rows}"
        )
    print(f"{source['benchmark']}: loaded {len(rows)}/{expected_rows}", flush=True)
    return rows


def _normalise_local(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "question": str(row["question"]).strip(),
        "choices": {
            str(label).strip().upper(): str(text).strip()
            for label, text in row["options"].items()
        },
        "answer": str(row["answer_idx"]).strip().upper(),
    }


def _compare_dataset(
    local_rows: list[dict[str, Any]],
    official_rows: list[dict[str, Any]],
    normalise: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    matched = 0
    for index, (local, official) in enumerate(zip(local_rows, official_rows)):
        local_normalised = _normalise_local(local)
        official_normalised = normalise(official)
        differing_fields = [
            field
            for field in ("question", "choices", "answer")
            if local_normalised[field] != official_normalised[field]
        ]
        if differing_fields:
            if len(mismatches) < 20:
                mismatches.append({"index": index, "fields": differing_fields})
        else:
            matched += 1
    return {
        "local_rows": len(local_rows),
        "official_rows": len(official_rows),
        "matched_rows": matched,
        "mismatched_rows": max(len(local_rows), len(official_rows)) - matched,
        "first_mismatches": mismatches,
        "status": (
            "exact_match"
            if len(local_rows) == len(official_rows) and matched == len(local_rows)
            else "mismatch"
        ),
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    source_path = _repo_path(args.source).resolve()
    output_path = _repo_path(args.output).resolve()
    with source_path.open("r", encoding="utf-8") as handle:
        local_data = json.load(handle)
    existing_report: dict[str, Any] | None = None
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as handle:
            existing_report = json.load(handle)

    dataset_reports: list[dict[str, Any]] = []
    for source in SOURCES:
        repo_id = urllib_parse.quote(source["repo_id"], safe="/")
        revision = urllib_parse.quote(source["revision"], safe="")
        repo_url = f"{HF_API}/{repo_id}/revision/{revision}"
        repo_info = _get_json(repo_url, args.timeout, args.max_retries)
        actual_revision = str(repo_info["sha"])
        if actual_revision != source["revision"]:
            raise ValueError(
                f"{source['repo_id']} revision changed: "
                f"{actual_revision} != {source['revision']}"
            )

        official_rows = _load_pinned_rows(source, args)
        local_rows = local_data[source["local_key"]]
        comparison = _compare_dataset(local_rows, official_rows, source["normalise"])
        dataset_reports.append(
            {
                "benchmark": source["benchmark"],
                "local_key": source["local_key"],
                "official_repo": source["repo_id"],
                "official_revision": actual_revision,
                "official_config": source["config"],
                "official_split": source["split"],
                "official_file": source["data_file"],
                "official_file_sha256": source["file_sha256"],
                "comparison": comparison,
                "license": source["license"],
            }
        )

    content_verified = all(
        report["comparison"]["status"] == "exact_match"
        for report in dataset_reports
    )
    if args.update_report or existing_report is None:
        verified_at_utc = datetime.now(timezone.utc).isoformat()
    else:
        verified_at_utc = existing_report.get("verified_at_utc")
    report = {
        "schema_version": 2,
        "verified_at_utc": verified_at_utc,
        "local_source_file": source_path.relative_to(REPO_ROOT).as_posix(),
        "local_source_sha256": _sha256(source_path),
        "content_verification": "exact_match" if content_verified else "mismatch",
        "provenance_status": (
            "content_verified_license_review_required"
            if content_verified
            else "content_mismatch"
        ),
        "datasets": dataset_reports,
    }
    if not content_verified:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(1)
    if args.update_report:
        _write_json(output_path, report)
        print(f"Updated provenance report: {output_path}")
    elif existing_report is None:
        raise RuntimeError(
            "committed provenance report is missing; rerun with --update-report "
            "only after reviewing the pinned sources"
        )
    elif report != existing_report:
        raise RuntimeError(
            "live verification passed, but the committed provenance report differs; "
            "review the diff before using --update-report"
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("Provenance verification passed without modifying the committed report.")


if __name__ == "__main__":
    main()
