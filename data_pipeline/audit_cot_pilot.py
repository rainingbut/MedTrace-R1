"""Audit a completed private CoT pilot without emitting questions or trajectories."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Iterable
import unicodedata


REPO_ROOT = Path(__file__).resolve().parents[1]
NEAR_DUPLICATE_JACCARD = 0.90


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="results/cot/pilot_v1_real")
    return parser.parse_args()


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected one JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            records.append(value)
    return records


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _write_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(value)
    temporary.replace(path)


def _counts(values: Iterable[object]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def _sum_cost(events: Iterable[dict[str, Any]]) -> float:
    return round(
        sum(float(event.get("usage", {}).get("cost_cny", 0)) for event in events),
        8,
    )


def _error_category(error: str) -> str:
    exception = error.partition(":")[0].strip() or "UnknownError"
    lowered = error.casefold()
    if "http error" in lowered or exception == "HTTPError":
        match = re.search(r"\b([45]\d\d)\b", error)
        return f"http_{match.group(1)}" if match else "http_error"
    if "json" in lowered:
        return "json_parse_or_contract"
    if "timed out" in lowered or "timeout" in lowered:
        return "timeout"
    if exception in {"URLError", "ConnectionError"}:
        return "connection_error"
    if "usage.cost" in lowered:
        return "missing_provider_cost"
    if exception == "ValueError":
        return "response_contract"
    return exception


def _trajectory_text(event: dict[str, Any]) -> str:
    rule = event.get("rule_check") or {}
    return " ".join(str(step) for step in rule.get("steps") or [])


def _comparison_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text)).casefold()
    return " ".join(re.findall(r"[^\W_]+", value, flags=re.UNICODE))


def _tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for word in _comparison_text(text).split():
        if re.search(r"[\u3400-\u9fff]", word):
            tokens.update(word[index : index + 2] for index in range(len(word) - 1))
            if len(word) == 1:
                tokens.add(word)
        else:
            tokens.add(word)
    return tokens


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _duplicate_stats(teacher_events: list[dict[str, Any]]) -> dict[str, Any]:
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in teacher_events:
        if (event.get("rule_check") or {}).get("passed"):
            by_question[str(event["record_id"])].append(event)

    exact_pairs = 0
    near_pairs = 0
    compared_pairs = 0
    max_nonexact = 0.0
    questions_with_exact = 0
    questions_with_near = 0
    for events in by_question.values():
        question_exact = False
        question_near = False
        texts = [_comparison_text(_trajectory_text(event)) for event in events]
        token_sets = [_tokens(text) for text in texts]
        for left in range(len(events)):
            for right in range(left + 1, len(events)):
                compared_pairs += 1
                if texts[left] == texts[right]:
                    exact_pairs += 1
                    question_exact = True
                    continue
                score = _jaccard(token_sets[left], token_sets[right])
                max_nonexact = max(max_nonexact, score)
                if score >= NEAR_DUPLICATE_JACCARD:
                    near_pairs += 1
                    question_near = True
        questions_with_exact += int(question_exact)
        questions_with_near += int(question_near)
    return {
        "threshold": NEAR_DUPLICATE_JACCARD,
        "compared_pairs": compared_pairs,
        "exact_duplicate_pairs": exact_pairs,
        "near_duplicate_pairs_excluding_exact": near_pairs,
        "questions_with_exact_duplicates": questions_with_exact,
        "questions_with_near_duplicates": questions_with_near,
        "maximum_nonexact_jaccard": round(max_nonexact, 6),
    }


def _coverage(
    metadata: dict[str, Any],
    teachers: list[dict[str, Any]],
    canonical: list[dict[str, Any]],
    sft: list[dict[str, Any]],
) -> dict[str, Any]:
    def teacher_identity(event: dict[str, Any]) -> str:
        record = event.get("record") or {}
        return str(record.get("content_sha256") or event["record_id"])

    def canonical_identity(record: dict[str, Any]) -> str:
        source = record.get("source") or {}
        return str(source.get("content_sha256") or source["source_id"])

    all_ids = {teacher_identity(event) for event in teachers}
    rule_ids = {
        teacher_identity(event)
        for event in teachers
        if (event.get("rule_check") or {}).get("passed")
    }
    canonical_ids = {canonical_identity(record) for record in canonical}
    sft_ids = {canonical_identity(record) for record in sft}
    expected = int(metadata["questions"])
    return {
        "expected_questions": expected,
        "teacher_questions": len(all_ids),
        "questions_with_rule_pass": len(rule_ids),
        "questions_with_canonical": len(canonical_ids),
        "questions_with_sft": len(sft_ids),
        "questions_without_rule_pass": expected - len(rule_ids),
        "questions_without_canonical": expected - len(canonical_ids),
        "questions_without_sft": expected - len(sft_ids),
    }


def _invariants(
    metadata: dict[str, Any],
    teachers: list[dict[str, Any]],
    screeners: list[dict[str, Any]],
    validators: list[dict[str, Any]],
    canonical: list[dict[str, Any]],
    sft: list[dict[str, Any]],
    prm: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    checks = {
        "metadata_complete": metadata.get("status") == "complete",
        "teacher_count_matches": len(teachers) == int(metadata["teacher_events"]),
        "screener_count_matches": len(screeners) == int(metadata["screener_events"]),
        "validator_count_matches": len(validators) == int(metadata["validator_events"]),
        "canonical_count_matches": len(canonical)
        == int(metadata["canonical_trajectories"]),
        "sft_count_matches": len(sft) == int(metadata["sft_records"]),
        "prm_count_matches": len(prm) == int(metadata["prm_records"]),
        "teacher_resume_keys_unique": len(teachers)
        == len({(event["record_id"], event["candidate_index"]) for event in teachers}),
        "screener_resume_keys_unique": len(screeners)
        == len({(event["record_id"], event["candidate_index"]) for event in screeners}),
        "validator_resume_keys_unique": len(validators)
        == len({(event["record_id"], event["candidate_index"]) for event in validators}),
        "sft_is_canonical_subset": {
            record["trajectory_id"] for record in sft
        }.issubset({record["trajectory_id"] for record in canonical}),
        "prm_trajectory_is_canonical": {
            record["trajectory_id"] for record in prm
        }.issubset({record["trajectory_id"] for record in canonical}),
    }
    return [{"check": name, "passed": passed} for name, passed in checks.items()]


def audit_run(run_dir: Path) -> dict[str, Any]:
    metadata = _load_json(run_dir / "metadata.json")
    teachers = _load_jsonl(run_dir / "teacher_events.jsonl")
    screeners = _load_jsonl(run_dir / "screener_events.jsonl")
    validators = _load_jsonl(run_dir / "validator_events.jsonl")
    canonical = _load_jsonl(run_dir / "canonical_trajectories.jsonl")
    sft = _load_jsonl(run_dir / "sft_verified.jsonl")
    prm = _load_jsonl(run_dir / "process_train.jsonl")

    teacher_failures = Counter(
        code
        for event in teachers
        for code in (event.get("rule_check") or {}).get("failure_codes") or []
    )
    screener_codes = Counter(
        code
        for event in screeners
        for code in (event.get("result") or {}).get("error_codes") or []
    )
    validator_errors = Counter(
        _error_category(str(error))
        for event in validators
        for error in event.get("errors") or []
    )
    validator_error_events = Counter(
        category
        for event in validators
        for category in {
            _error_category(str(error)) for error in event.get("errors") or []
        }
    )
    complete_validators = [
        event for event in validators if event.get("status") == "complete"
    ]
    validator_results = [event["result"] for event in complete_validators]
    validator_step_codes = Counter(
        code
        for result in validator_results
        for step in result["steps"]
        for code in step.get("error_codes") or []
    )
    costs = {
        "teacher_cny": _sum_cost(teachers),
        "screener_cny": _sum_cost(screeners),
        "validator_cny": _sum_cost(validators),
    }
    costs["total_cny"] = round(sum(costs.values()), 8)

    report: dict[str, Any] = {
        "schema_version": "medtrace.cot-pilot-audit.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contains_private_text": False,
        "run_identity": {
            "config_sha256": metadata["config_sha256"],
            "questions_sha256": metadata["questions_sha256"],
            "git_commit": metadata["preflight"]["screener_runtime"]["git_commit"],
        },
        "counts": {
            "questions": int(metadata["questions"]),
            "teacher_events": len(teachers),
            "rule_passed": sum(
                bool((event.get("rule_check") or {}).get("passed"))
                for event in teachers
            ),
            "screener_events": len(screeners),
            "validator_events": len(validators),
            "validator_complete": len(complete_validators),
            "canonical_trajectories": len(canonical),
            "sft_records": len(sft),
            "prm_records": len(prm),
        },
        "coverage": _coverage(metadata, teachers, canonical, sft),
        "teacher": {
            "status": _counts(event.get("status") for event in teachers),
            "attempts": _counts(event.get("attempts") for event in teachers),
            "rule_failure_codes": dict(sorted(teacher_failures.items())),
        },
        "screener": {
            "status": _counts(event.get("status") for event in screeners),
            "verdict": _counts(
                (event.get("result") or {}).get("verdict") for event in screeners
            ),
            "attempts": _counts(event.get("attempts") for event in screeners),
            "error_codes": dict(sorted(screener_codes.items())),
        },
        "validator": {
            "status": _counts(event.get("status") for event in validators),
            "attempts": _counts(event.get("attempts") for event in validators),
            "error_categories_by_attempt": dict(sorted(validator_errors.items())),
            "error_categories_by_event": dict(sorted(validator_error_events.items())),
            "trajectory_label": _counts(
                result["trajectory_label"] for result in validator_results
            ),
            "problem_status": _counts(
                result["problem_status"] for result in validator_results
            ),
            "first_error_step": _counts(
                "none" if result["first_error_step"] is None else result["first_error_step"]
                for result in validator_results
            ),
            "step_error_codes": dict(sorted(validator_step_codes.items())),
            "cost_source": _counts(
                (event.get("usage") or {}).get("cost_source") for event in validators
            ),
            "routed_provider": _counts(
                (event.get("usage") or {}).get("routed_provider") for event in validators
            ),
        },
        "prm": {
            "labels": _counts(record.get("label") for record in prm),
            "error_codes": dict(
                sorted(
                    Counter(
                        code
                        for record in prm
                        for code in record.get("error_codes") or []
                    ).items()
                )
            ),
        },
        "candidate_diversity": _duplicate_stats(teachers),
        "cost": {
            **costs,
            "metadata_total_cny": float(metadata["spent_cny_equivalent"]),
            "difference_from_metadata_cny": round(
                costs["total_cny"] - float(metadata["spent_cny_equivalent"]), 8
            ),
            "stop_limit_cny": float(metadata["budget_stop_limit_cny"]),
        },
        "invariants": _invariants(
            metadata, teachers, screeners, validators, canonical, sft, prm
        ),
    }
    report["invariants_passed"] = all(
        check["passed"] for check in report["invariants"]
    ) and abs(report["cost"]["difference_from_metadata_cny"]) <= 1e-7
    return report


def render_markdown(report: dict[str, Any]) -> str:
    counts = report["counts"]
    coverage = report["coverage"]
    validator = report["validator"]
    prm = report["prm"]
    diversity = report["candidate_diversity"]
    cost = report["cost"]

    lines = [
        "# CoT pilot quality audit",
        "",
        "This report contains aggregate statistics only; it does not contain question or trajectory text.",
        "",
        "## Funnel",
        "",
        "| Measure | Count |",
        "|---|---:|",
    ]
    for key in (
        "questions", "teacher_events", "rule_passed", "screener_events",
        "validator_events", "validator_complete", "canonical_trajectories",
        "sft_records", "prm_records",
    ):
        lines.append(f"| {key} | {counts[key]} |")
    lines.extend([
        "",
        "## Coverage and labels",
        "",
        f"- Questions with at least one SFT trajectory: {coverage['questions_with_sft']}/{coverage['expected_questions']}",
        f"- Validator trajectory labels: `{json.dumps(validator['trajectory_label'], sort_keys=True)}`",
        f"- PRM prefix labels: `{json.dumps(prm['labels'], sort_keys=True)}`",
        "",
        "## Failures and routing",
        "",
        f"- Validator status: `{json.dumps(validator['status'], sort_keys=True)}`",
        f"- Validator error categories by attempt: `{json.dumps(validator['error_categories_by_attempt'], sort_keys=True)}`",
        f"- Validator routed providers: `{json.dumps(validator['routed_provider'], sort_keys=True)}`",
        "",
        "## Candidate diversity",
        "",
        f"- Compared same-question pairs: {diversity['compared_pairs']}",
        f"- Exact duplicate pairs: {diversity['exact_duplicate_pairs']}",
        f"- Near duplicate pairs (Jaccard >= {diversity['threshold']}): {diversity['near_duplicate_pairs_excluding_exact']}",
        f"- Maximum non-exact Jaccard: {diversity['maximum_nonexact_jaccard']}",
        "",
        "## Cost and integrity",
        "",
        f"- Teacher: CNY {cost['teacher_cny']:.8f}",
        f"- Validator: CNY {cost['validator_cny']:.8f}",
        f"- Total: CNY {cost['total_cny']:.8f} / {cost['stop_limit_cny']:.2f}",
        f"- All structural invariants passed: {str(report['invariants_passed']).lower()}",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    run_dir = _repo_path(args.run_dir)
    report = audit_run(run_dir)
    json_path = run_dir / "quality_audit.json"
    markdown_path = run_dir / "quality_audit.md"
    _write_json(json_path, report)
    _write_text(markdown_path, render_markdown(report))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Audit JSON: {json_path}")
    print(f"Audit Markdown: {markdown_path}")
    if not report["invariants_passed"]:
        raise RuntimeError("quality audit structural invariants failed")


if __name__ == "__main__":
    main()
