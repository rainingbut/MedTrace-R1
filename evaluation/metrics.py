"""Metrics for MEDTRACE-R1 multiple-choice evaluation runs."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Iterable, Mapping


def _safe_rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _optional_mean(values: list[float]) -> float | None:
    return round(mean(values), 6) if values else None


def _score_group(records: list[Mapping[str, object]]) -> dict[str, object]:
    total = len(records)
    correct = 0
    parsed = 0
    format_valid = 0
    errors = 0
    truncated = 0
    completion_tokens: list[float] = []
    latencies: list[float] = []

    for record in records:
        gold = str(record["answer"]).strip().upper()
        predicted_value = record.get("extracted_answer")
        predicted = (
            str(predicted_value).strip().upper()
            if predicted_value is not None
            else None
        )
        parsed += int(predicted is not None)
        correct += int(predicted == gold)
        format_valid += int(bool(record.get("format_valid", False)))
        errors += int(bool(record.get("error")))
        truncated += int(record.get("finish_reason") == "length")

        if record.get("completion_tokens") is not None:
            completion_tokens.append(float(record["completion_tokens"]))
        if record.get("latency_seconds") is not None:
            latencies.append(float(record["latency_seconds"]))

    return {
        "total": total,
        "correct": correct,
        "accuracy": _safe_rate(correct, total),
        "parsed": parsed,
        "parse_rate": _safe_rate(parsed, total),
        "format_valid": format_valid,
        "format_rate": _safe_rate(format_valid, total),
        "errors": errors,
        "truncated": truncated,
        "truncation_rate": _safe_rate(truncated, total),
        "average_completion_tokens": _optional_mean(completion_tokens),
        "average_latency_seconds": _optional_mean(latencies),
    }


def score_records(records: Iterable[Mapping[str, object]]) -> dict[str, object]:
    """Score predictions once, using one fixed answer extraction policy."""

    materialised = list(records)
    seen_ids: set[str] = set()
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)

    for record in materialised:
        missing = {"id", "benchmark", "answer"} - set(record)
        if missing:
            raise ValueError(f"prediction record is missing fields: {sorted(missing)}")
        record_id = str(record["id"])
        if record_id in seen_ids:
            raise ValueError(f"duplicate prediction id: {record_id}")
        seen_ids.add(record_id)
        grouped[str(record["benchmark"])].append(record)

    return {
        "schema_version": 1,
        "overall": _score_group(materialised),
        "by_benchmark": {
            benchmark: _score_group(grouped[benchmark])
            for benchmark in sorted(grouped)
        },
    }
