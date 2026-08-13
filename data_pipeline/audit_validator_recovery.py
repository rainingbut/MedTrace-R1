"""Audit the isolated validator recovery canary without calling an API."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import yaml

from data_pipeline.cot_api import validate_validator_result
from data_pipeline.cot_diagnostics import error_detail
from data_pipeline.cot_recovery_config import validate_recovery_config
from data_pipeline.run_cot_pilot_real import _load_completed
from data_pipeline.run_validator_recovery import REPO_ROOT, _source_hashes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/cot/validator_recovery_v2.yaml"
    )
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _counts(values: Iterable[object]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def _strict_binary_counts(values: Iterable[object]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for value in values:
        if type(value) is int and value in {0, 1}:
            counts[f"int:{value}"] += 1
        elif type(value) is bool:
            counts[f"bool:{str(value).lower()}"] += 1
        else:
            counts[f"invalid:{type(value).__name__}"] += 1
    return dict(sorted(counts.items()))


def _safe_diagnostics(events: list[dict[str, Any]]) -> dict[str, Any]:
    attempts = [
        diagnostic
        for event in events
        for diagnostic in event.get("attempt_diagnostics") or []
    ]
    failure_details = Counter(
        str(diagnostic.get("error_detail") or "unknown")
        for diagnostic in attempts
        if diagnostic.get("status") != "complete"
    )
    return {
        "attempts": len(attempts),
        "attempt_status": _counts(
            diagnostic.get("status") or "unknown" for diagnostic in attempts
        ),
        "failure_details": dict(sorted(failure_details.items())),
        "finish_reason": _counts(
            diagnostic.get("finish_reason") or "none" for diagnostic in attempts
        ),
        "routed_provider": _counts(
            diagnostic.get("routed_provider") or "unknown" for diagnostic in attempts
        ),
        "content_present": _counts(
            bool(diagnostic.get("content_present")) for diagnostic in attempts
        ),
        "reasoning_tokens": {
            "observed_attempts": sum(
                diagnostic.get("reasoning_tokens") is not None for diagnostic in attempts
            ),
            "missing_attempts": sum(
                diagnostic.get("reasoning_tokens") is None for diagnostic in attempts
            ),
            "sum": sum(
                int(diagnostic.get("reasoning_tokens") or 0)
                for diagnostic in attempts
            ),
        },
    }


def audit_recovery(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    validate_recovery_config(config)

    run_dir = _repo_path(str(config["source_run_dir"]))
    output_dir = run_dir / str(config["output_subdir"])
    pilot_metadata = _load_json(run_dir / "metadata.json")
    recovery_metadata = _load_json(output_dir / "canary_metadata.json")
    manifest = _load_json(output_dir / "private_manifest.json")

    teachers = _load_completed(
        run_dir / "teacher_events.jsonl", ("record_id", "candidate_index")
    )
    screeners = _load_completed(
        run_dir / "screener_events.jsonl", ("record_id", "candidate_index")
    )
    old_validators = _load_completed(
        run_dir / "validator_events.jsonl", ("record_id", "candidate_index")
    )
    recovered = _load_completed(
        output_dir / "canary_events.jsonl", ("record_id", "candidate_index")
    )
    selected = {
        (str(key[0]), int(key[1])) for key in manifest.get("selected_keys") or []
    }

    valid_contracts = 0
    benchmark_by_key: dict[tuple[Any, ...], str] = {}
    screener_by_key: dict[tuple[Any, ...], str] = {}
    source_details_by_key: dict[tuple[Any, ...], list[str]] = {}
    result_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for key, event in recovered.items():
        teacher = teachers.get(key) or {}
        benchmark_by_key[key] = str(
            ((teacher.get("record") or {}).get("benchmark")) or "unknown"
        )
        screener_by_key[key] = str(
            (((screeners.get(key) or {}).get("result") or {}).get("verdict"))
            or "missing"
        )
        old = old_validators.get(key) or {}
        source_details_by_key[key] = [
            error_detail(str(error)) for error in old.get("errors") or []
        ]
        result = event.get("result")
        if event.get("status") == "complete" and isinstance(result, dict):
            steps = ((teacher.get("rule_check") or {}).get("steps")) or []
            try:
                validate_validator_result(result, len(steps))
            except (TypeError, ValueError):
                pass
            else:
                valid_contracts += 1
                result_by_key[key] = result

    labels = [result["trajectory_label"] for result in result_by_key.values()]
    prefix_labels = [
        step["prefix_label"]
        for result in result_by_key.values()
        for step in result["steps"]
    ]
    local_verdicts = Counter(
        str(step["local_verdict"])
        for result in result_by_key.values()
        for step in result["steps"]
    )
    step_error_codes = Counter(
        str(code)
        for result in result_by_key.values()
        for step in result["steps"]
        for code in step.get("error_codes") or []
    )

    by_benchmark: dict[str, Counter[str]] = defaultdict(Counter)
    by_screener: dict[str, Counter[str]] = defaultdict(Counter)
    by_source_detail: dict[str, Counter[str]] = defaultdict(Counter)
    for key, result in result_by_key.items():
        outcome = f"label_{result['trajectory_label']}"
        by_benchmark[benchmark_by_key[key]][outcome] += 1
        by_screener[screener_by_key[key]][outcome] += 1
        details = source_details_by_key[key] or ["unknown"]
        by_source_detail[details[0]][outcome] += 1

    event_cost = round(
        sum(float((event.get("usage") or {}).get("cost_cny") or 0) for event in recovered.values()),
        8,
    )
    source_identity = {
        "config_sha256": pilot_metadata.get("config_sha256"),
        "questions_sha256": pilot_metadata.get("questions_sha256"),
        "generation_git_commit": (
            ((pilot_metadata.get("preflight") or {}).get("screener_runtime") or {})
            .get("git_commit")
        ),
    }
    expected_identity = {
        key: value
        for key, value in config["source_identity"].items()
        if key != "expected_failed_validator_events"
    }
    source_hashes = _source_hashes(run_dir)
    expected_total = int(config["canary"]["total"])
    complete_count = sum(event.get("status") == "complete" for event in recovered.values())
    old_failed_count = sum(
        (old_validators.get(key) or {}).get("status") != "complete" for key in selected
    )
    source_details_match = all(
        list(recovered[key].get("source_error_details") or [])
        == source_details_by_key.get(key, [])
        for key in recovered
        if key in old_validators
    )
    checks = {
        "recovery_metadata_complete": recovery_metadata.get("status") == "complete",
        "source_identity_matches_config": source_identity == expected_identity,
        "recovery_identity_matches_source": recovery_metadata.get("source_identity")
        == source_identity,
        "config_hash_matches_manifest": manifest.get("config_sha256")
        == _sha256(config_path),
        "source_hashes_match_manifest": manifest.get("source_artifact_sha256")
        == source_hashes,
        "source_artifacts_marked_unchanged": recovery_metadata.get(
            "source_artifacts_unchanged"
        ) is True,
        "selected_count_matches_config": len(selected) == expected_total,
        "event_keys_match_selection": set(recovered) == selected,
        "metadata_counts_match_events": (
            recovery_metadata.get("selected_events") == expected_total
            and recovery_metadata.get("completed_events") == complete_count
        ),
        "all_selected_old_validators_failed": old_failed_count == expected_total,
        "all_recovery_events_complete": complete_count == expected_total,
        "all_results_pass_strict_contract": valid_contracts == expected_total,
        "source_error_details_match": source_details_match,
        "cost_matches_metadata": abs(
            event_cost - float(recovery_metadata.get("spent_cny_equivalent") or 0)
        ) <= 1e-7,
    }
    integrity_passed = all(checks.values())
    transport_passed = (
        complete_count == expected_total and valid_contracts == expected_total
    )
    all_same_label = len(set(labels)) == 1 if labels else False

    report: dict[str, Any] = {
        "schema_version": "medtrace.validator-recovery-audit.v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contains_private_text_or_ids": False,
        "source_identity": source_identity,
        "counts": {
            "selected": len(selected),
            "old_validator_unavailable": old_failed_count,
            "recovery_complete": complete_count,
            "strict_contract_valid": valid_contracts,
        },
        "recovery": {
            "status": _counts(event.get("status") for event in recovered.values()),
            "strict_trajectory_label_counts": _strict_binary_counts(labels),
            "problem_status": _counts(
                result["problem_status"] for result in result_by_key.values()
            ),
            "answer_consistent": _counts(
                result["answer_consistent"] for result in result_by_key.values()
            ),
            "first_error_present": _counts(
                result["first_error_step"] is not None
                for result in result_by_key.values()
            ),
            "strict_prefix_label_counts": _strict_binary_counts(prefix_labels),
            "local_verdict": dict(sorted(local_verdicts.items())),
            "step_error_codes": dict(sorted(step_error_codes.items())),
            "labels_by_benchmark": {
                key: dict(sorted(value.items()))
                for key, value in sorted(by_benchmark.items())
            },
            "labels_by_screener_verdict": {
                key: dict(sorted(value.items()))
                for key, value in sorted(by_screener.items())
            },
            "labels_by_source_failure_detail": {
                key: dict(sorted(value.items()))
                for key, value in sorted(by_source_detail.items())
            },
        },
        "diagnostics": _safe_diagnostics(list(recovered.values())),
        "cost": {
            "events_cny_equivalent": event_cost,
            "metadata_cny_equivalent": float(
                recovery_metadata.get("spent_cny_equivalent") or 0
            ),
        },
        "gates": {
            "integrity_passed": integrity_passed,
            "transport_and_contract_passed": transport_passed,
            "semantic_sample_size": len(labels),
            "semantic_auto_approval": False,
            "semantic_review_note": (
                "All recovered trajectory labels are identical; inspect this small-sample "
                "signal before scaling."
                if all_same_label
                else "The six-event canary is too small for automatic semantic approval."
            ),
        },
        "invariants": [
            {"check": name, "passed": passed} for name, passed in checks.items()
        ],
    }
    return report


def render_markdown(report: dict[str, Any]) -> str:
    counts = report["counts"]
    recovery = report["recovery"]
    diagnostics = report["diagnostics"]
    gates = report["gates"]
    return "\n".join([
        "# Validator recovery canary audit",
        "",
        "This report contains aggregate statistics only; it contains no question text, trajectory text, IDs, or request IDs.",
        "",
        "## Outcome",
        "",
        f"- Old unavailable -> recovered complete: {counts['old_validator_unavailable']} -> {counts['recovery_complete']}",
        f"- Strictly contract-valid results: {counts['strict_contract_valid']}/{counts['selected']}",
        f"- Integrity passed: {str(gates['integrity_passed']).lower()}",
        f"- Transport and contract passed: {str(gates['transport_and_contract_passed']).lower()}",
        f"- Semantic auto-approval: {str(gates['semantic_auto_approval']).lower()}",
        f"- Semantic note: {gates['semantic_review_note']}",
        "",
        "## Labels",
        "",
        f"- Trajectory: `{json.dumps(recovery['strict_trajectory_label_counts'], sort_keys=True)}`",
        f"- Prefix: `{json.dumps(recovery['strict_prefix_label_counts'], sort_keys=True)}`",
        f"- Problem status: `{json.dumps(recovery['problem_status'], sort_keys=True)}`",
        f"- By benchmark: `{json.dumps(recovery['labels_by_benchmark'], sort_keys=True)}`",
        f"- By old failure detail: `{json.dumps(recovery['labels_by_source_failure_detail'], sort_keys=True)}`",
        "",
        "## Diagnostics and cost",
        "",
        f"- Attempt status: `{json.dumps(diagnostics['attempt_status'], sort_keys=True)}`",
        f"- Finish reason: `{json.dumps(diagnostics['finish_reason'], sort_keys=True)}`",
        f"- Routed provider: `{json.dumps(diagnostics['routed_provider'], sort_keys=True)}`",
        f"- Reasoning tokens: `{json.dumps(diagnostics['reasoning_tokens'], sort_keys=True)}`",
        f"- Cost: CNY {report['cost']['events_cny_equivalent']:.8f}",
        "",
    ])


def main() -> None:
    args = parse_args()
    config_path = _repo_path(args.config)
    report = audit_recovery(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = (
        _repo_path(str(config["source_run_dir"])) / str(config["output_subdir"])
    )
    json_path = output_dir / "canary_quality_audit.json"
    markdown_path = output_dir / "canary_quality_audit.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Audit JSON: {json_path}")
    print(f"Audit Markdown: {markdown_path}")
    if not report["gates"]["integrity_passed"]:
        raise RuntimeError("validator recovery audit integrity checks failed")


if __name__ == "__main__":
    main()
