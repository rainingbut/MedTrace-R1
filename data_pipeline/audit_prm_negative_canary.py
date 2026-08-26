"""Audit the isolated 24-case PRM negative canary without calling a model."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

import yaml

from data_pipeline.build_strict_pilot_view import _source_hashes
from data_pipeline.cot_api import validate_validator_result
from data_pipeline.prm_negative_canary_config import (
    validate_prm_negative_canary_config,
)
from data_pipeline.prm_negative_policy import verification_disposition
from data_pipeline.run_cot_pilot_real import _load_jsonl, _write_json


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cot/prm_negative_canary_v1.yaml")
    return parser.parse_args()


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def audit_canary(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("PRM negative canary config must be an object")
    validate_prm_negative_canary_config(config)
    run_dir = _repo_path(str(config["source_run_dir"]))
    output_dir = run_dir / str(config["output_subdir"])
    candidates = _load_jsonl(output_dir / "candidates.jsonl")
    events = _load_jsonl(output_dir / "validator_events.jsonl")
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (output_dir / "private_manifest.json").read_text(encoding="utf-8")
    )
    candidate_by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    event_by_id = {event["candidate_id"]: event for event in events}
    duplicate_candidates = len(candidate_by_id) != len(candidates)
    duplicate_events = len(event_by_id) != len(events)

    dispositions: Counter[str] = Counter()
    by_origin: dict[str, Counter[str]] = defaultdict(Counter)
    by_benchmark: dict[str, Counter[str]] = defaultdict(Counter)
    first_errors: Counter[str] = Counter()
    strict_contract_valid = 0
    negative_origins: set[str] = set()
    negative_benchmarks: set[str] = set()
    controlled_negative = 0
    controlled_exact_first_error = 0
    for candidate_id, candidate in candidate_by_id.items():
        event = event_by_id.get(candidate_id)
        if event is None or event.get("status") != "complete":
            disposition = "unavailable"
        else:
            result = event.get("result")
            try:
                validate_validator_result(
                    result, len(candidate["trajectory"]["steps"])
                )
            except (KeyError, TypeError, ValueError):
                disposition = "invalid_contract"
            else:
                strict_contract_valid += 1
                disposition = verification_disposition(result)
                first = result.get("first_error_step")
                first_errors["none" if first is None else str(first)] += 1
        dispositions[disposition] += 1
        origin = candidate["origin"]
        benchmark = candidate["source"]["dataset"]
        by_origin[origin][disposition] += 1
        by_benchmark[benchmark][disposition] += 1
        if disposition == "strict_process_negative":
            negative_origins.add(origin)
            negative_benchmarks.add(benchmark)
            if origin == "controlled_single_error":
                controlled_negative += 1
                if (
                    event["result"]["first_error_step"]
                    == candidate["intended_error_step"]
                ):
                    controlled_exact_first_error += 1

    source_hashes = _source_hashes(run_dir)
    source_unchanged = source_hashes == manifest.get("source_artifact_sha256")
    event_cost = round(
        sum(float(event.get("usage", {}).get("cost_cny", 0)) for event in events),
        8,
    )
    integrity_checks = {
        "metadata_complete": metadata.get("status") == "complete",
        "candidate_total_24": len(candidates) == 24,
        "validator_event_total_24": len(events) == 24,
        "candidate_ids_unique": not duplicate_candidates,
        "validator_candidate_ids_unique": not duplicate_events,
        "validator_keys_match_candidates": set(event_by_id) == set(candidate_by_id),
        "source_artifacts_unchanged": source_unchanged,
        "metadata_cost_matches_events": abs(
            event_cost - float(metadata.get("spent_cny_equivalent", -1))
        ) < 1e-8,
    }
    strict_negatives = dispositions["strict_process_negative"]
    machine_checks = {
        "transport_and_contract_24_of_24": strict_contract_valid == 24,
        "minimum_eight_strict_negatives": strict_negatives >= 8,
        "negative_examples_cover_both_benchmarks": negative_benchmarks
        == {"medqa", "medmcqa"},
        "negative_examples_cover_two_origins": len(negative_origins) >= 2,
    }
    controlled_match_rate = (
        controlled_exact_first_error / controlled_negative
        if controlled_negative else None
    )
    return {
        "schema_version": "medtrace.prm-negative-canary-audit.v1",
        "contains_private_text_or_ids": False,
        "counts": {
            "candidates": len(candidates),
            "validator_events": len(events),
            "strict_contract_valid": strict_contract_valid,
            "strict_process_negatives": strict_negatives,
        },
        "dispositions": dict(sorted(dispositions.items())),
        "by_origin": {
            key: dict(sorted(value.items())) for key, value in sorted(by_origin.items())
        },
        "by_benchmark": {
            key: dict(sorted(value.items()))
            for key, value in sorted(by_benchmark.items())
        },
        "first_error_step": dict(sorted(first_errors.items())),
        "controlled": {
            "strict_negatives": controlled_negative,
            "exact_intended_first_error": controlled_exact_first_error,
            "exact_match_rate": controlled_match_rate,
        },
        "cost": {
            "events_cny_equivalent": event_cost,
            "hard_cap_cny_equivalent": config["budget"][
                "api_hard_cap_cny_equivalent"
            ],
        },
        "integrity": {
            "checks": integrity_checks,
            "passed": all(integrity_checks.values()),
            "failed_checks": sorted(
                key for key, value in integrity_checks.items() if not value
            ),
        },
        "machine_quality": {
            "checks": machine_checks,
            "passed": all(machine_checks.values()),
            "failed_checks": sorted(
                key for key, value in machine_checks.items() if not value
            ),
        },
        "human_review": {
            "required": True,
            "trajectory_label_accuracy_target": config["quality_gates"][
                "human_trajectory_label_accuracy"
            ],
            "exact_first_error_accuracy_target": config["quality_gates"][
                "human_exact_first_error_accuracy"
            ],
            "auto_approved": False,
        },
        "decision": {
            "training_merge_authorized": False,
            "full_scale_generation_authorized": False,
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    counts = report["counts"]
    return "\n".join(
        [
            "# PRM negative canary audit",
            "",
            "Aggregate-only report; no private question, ID, or trajectory text.",
            "",
            f"- Candidates / validator events: {counts['candidates']} / "
            f"{counts['validator_events']}",
            f"- Strict contract valid: {counts['strict_contract_valid']}",
            f"- Strict process negatives: {counts['strict_process_negatives']}",
            f"- Dispositions: `{json.dumps(report['dispositions'], sort_keys=True)}`",
            f"- By origin: `{json.dumps(report['by_origin'], sort_keys=True)}`",
            f"- By benchmark: `{json.dumps(report['by_benchmark'], sort_keys=True)}`",
            f"- Controlled target match: `"
            f"{json.dumps(report['controlled'], sort_keys=True)}`",
            f"- Cost: CNY {report['cost']['events_cny_equivalent']:.8f}",
            "",
            "## Gates",
            "",
            f"- Integrity passed: {str(report['integrity']['passed']).lower()}",
            f"- Machine quality passed: "
            f"{str(report['machine_quality']['passed']).lower()}",
            f"- Failed machine checks: `"
            f"{json.dumps(report['machine_quality']['failed_checks'])}`",
            "- Human review required: true",
            "- Training merge authorized: false",
            "- Full-scale generation authorized: false",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    config_path = _repo_path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("PRM negative canary config must be an object")
    validate_prm_negative_canary_config(config)
    report = audit_canary(config_path)
    output_dir = _repo_path(str(config["source_run_dir"])) / str(
        config["output_subdir"]
    )
    _write_json(output_dir / "quality_audit.json", report)
    markdown = render_markdown(report)
    (output_dir / "quality_audit.md").write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    if not report["integrity"]["passed"]:
        raise RuntimeError("PRM negative canary source/run integrity failed")


if __name__ == "__main__":
    main()
