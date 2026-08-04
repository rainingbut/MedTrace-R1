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
DATASETS_SERVER = "https://datasets-server.huggingface.co/rows"
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
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-retries", type=int, default=4)
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
    request = urllib_request.Request(url, headers={"User-Agent": "MEDTRACE-R1/0.1"})
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


def _fetch_rows(source: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    expected_rows = int(source["rows"])
    for offset in range(0, expected_rows, args.page_size):
        query = urllib_parse.urlencode(
            {
                "dataset": source["repo_id"],
                "config": source["config"],
                "split": source["split"],
                "offset": offset,
                "length": min(args.page_size, expected_rows - offset),
            }
        )
        response = _get_json(
            f"{DATASETS_SERVER}?{query}", args.timeout, args.max_retries
        )
        if int(response["num_rows_total"]) != expected_rows:
            raise ValueError(
                f"{source['repo_id']} row count changed: "
                f"{response['num_rows_total']} != {expected_rows}"
            )
        rows.extend(item["row"] for item in response["rows"])
        print(f"{source['benchmark']}: fetched {len(rows)}/{expected_rows}", flush=True)
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
    if not 1 <= args.page_size <= 100:
        raise ValueError("--page-size must be between 1 and 100")

    source_path = _repo_path(args.source).resolve()
    output_path = _repo_path(args.output).resolve()
    with source_path.open("r", encoding="utf-8") as handle:
        local_data = json.load(handle)

    dataset_reports: list[dict[str, Any]] = []
    for source in SOURCES:
        repo_url = f"{HF_API}/{urllib_parse.quote(source['repo_id'], safe='/')}"
        repo_info = _get_json(repo_url, args.timeout, args.max_retries)
        actual_revision = str(repo_info["sha"])
        if actual_revision != source["revision"]:
            raise ValueError(
                f"{source['repo_id']} revision changed: "
                f"{actual_revision} != {source['revision']}"
            )

        official_rows = _fetch_rows(source, args)
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
                "comparison": comparison,
                "license": source["license"],
            }
        )

    content_verified = all(
        report["comparison"]["status"] == "exact_match"
        for report in dataset_reports
    )
    report = {
        "schema_version": 1,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
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
    _write_json(output_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not content_verified:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
