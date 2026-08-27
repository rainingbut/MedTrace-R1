"""Audit the canonical PRM canary after isolated HTTP 429 recovery."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

import yaml

from data_pipeline.cot_api import validate_validator_result
from data_pipeline.prm_negative_policy import verification_disposition
from data_pipeline.prm_negative_recovery_config import (
    validate_prm_negative_recovery_config,
)
from data_pipeline.prm_negative_recovery_state import (
    canonical_event_map,
    load_source_state,
    recovery_attempts_by_candidate,
    sha256_file,
    source_canary_hashes,
)
from data_pipeline.run_cot_pilot_real import _load_jsonl, _write_json


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/cot/prm_negative_validator_recovery_v1.yaml",
    )
    return parser.parse_args()


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def audit_recovery(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("PRM negative recovery config must be an object")
    validate_prm_negative_recovery_config(config)
    source = load_source_state(config, REPO_ROOT)
    output_dir = REPO_ROOT / str(config["source_run_dir"]) / str(
        config["output_subdir"]
    )
    attempts = _load_jsonl(output_dir / "recovery_attempts.jsonl")
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (output_dir / "private_manifest.json").read_text(encoding="utf-8")
    )
    grouped = recovery_attempts_by_candidate(attempts)
    selected = set(source["selected_ids"])
    max_attempts = int(config["throttle"]["max_total_attempts_per_candidate"])
    attempt_structure_valid = True
    contract_valid_attempts = 0
    recovered_ids: set[str] = set()
    for candidate_id, values in grouped.items():
        if candidate_id not in selected or len(values) > max_attempts:
            attempt_structure_valid = False
            continue
        numbers = [int(event["recovery_attempt"]) for event in values]
        if numbers != list(range(1, len(values) + 1)):
            attempt_structure_valid = False
        complete_seen = False
        candidate = source["candidate_by_id"][candidate_id]
        for event in values:
            expected_fields = (
                event.get("source_status")
                == source["source_event_by_id"][candidate_id]["status"]
                and event.get("origin") == candidate["origin"]
                and event.get("benchmark") == candidate["source"]["dataset"]
            )
            if not expected_fields or complete_seen:
                attempt_structure_valid = False
            if event.get("status") == "complete":
                try:
                    validate_validator_result(
                        event.get("result"), len(candidate["trajectory"]["steps"])
                    )
                except (KeyError, TypeError, ValueError):
                    attempt_structure_valid = False
                else:
                    contract_valid_attempts += 1
                    recovered_ids.add(candidate_id)
                    complete_seen = True
            elif event.get("result") is not None:
                attempt_structure_valid = False

    canonical, provenance = canonical_event_map(source, attempts)
    dispositions: Counter[str] = Counter()
    by_origin: dict[str, Counter[str]] = defaultdict(Counter)
    by_benchmark: dict[str, Counter[str]] = defaultdict(Counter)
    canonical_contract_valid = 0
    negative_origins: set[str] = set()
    negative_benchmarks: set[str] = set()
    controlled_negative = 0
    controlled_exact = 0
    for candidate_id, candidate in source["candidate_by_id"].items():
        event = canonical[candidate_id]
        if event.get("status") != "complete":
            disposition = "unavailable"
        else:
            try:
                validate_validator_result(
                    event.get("result"), len(candidate["trajectory"]["steps"])
                )
            except (KeyError, TypeError, ValueError):
                disposition = "invalid_contract"
            else:
                canonical_contract_valid += 1
                disposition = verification_disposition(event["result"])
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
                    controlled_exact += 1

    recovery_cost = round(
        sum(float(event.get("usage", {}).get("cost_cny", 0)) for event in attempts),
        8,
    )
    source_cost = round(
        sum(
            float(event.get("usage", {}).get("cost_cny", 0))
            for event in source["source_events"]
        ),
        8,
    )
    manifest_expected = {
        "schema_version": "medtrace.prm-negative-validator-recovery-manifest.v1",
        "config_sha256": sha256_file(config_path),
        "source_canary_sha256": source["source_canary_hashes"],
        "selected_candidate_ids": source["selected_ids"],
    }
    terminal = sum(
        bool(values) and (
            any(event.get("status") == "complete" for event in values)
            or len(values) >= max_attempts
            or "http_429" not in set(values[-1].get("error_categories") or [])
        )
        for candidate_id in source["selected_ids"]
        for values in [grouped.get(candidate_id, [])]
    )
    integrity_checks = {
        "source_canary_unchanged": (
            source_canary_hashes(source["canary_dir"])
            == source["source_canary_hashes"]
        ),
        "private_manifest_exact": manifest == manifest_expected,
        "recovery_metadata_complete": metadata.get("status") == "complete",
        "selected_exactly_three": len(source["selected_ids"]) == 3,
        "attempt_structure_valid": attempt_structure_valid,
        "metadata_attempt_count_matches": (
            int(metadata.get("request_attempts", -1)) == len(attempts)
        ),
        "metadata_terminal_count_matches": (
            int(metadata.get("terminal_events", -1)) == terminal == 3
        ),
        "metadata_recovered_count_matches": (
            int(metadata.get("recovered_complete", -1)) == len(recovered_ids)
        ),
        "metadata_cost_matches": abs(
            float(metadata.get("spent_cny_equivalent", -1)) - recovery_cost
        ) < 1e-8,
        "recovery_within_hard_cap": recovery_cost
        <= float(config["budget"]["api_hard_cap_cny_equivalent"]),
    }
    strict_negatives = dispositions["strict_process_negative"]
    machine_checks = {
        "canonical_transport_and_contract_24_of_24": (
            canonical_contract_valid == 24
        ),
        "minimum_eight_strict_negatives": strict_negatives >= 8,
        "negative_examples_cover_both_benchmarks": negative_benchmarks
        == {"medqa", "medmcqa"},
        "negative_examples_cover_two_origins": len(negative_origins) >= 2,
    }
    return {
        "schema_version": "medtrace.prm-negative-validator-recovery-audit.v1",
        "contains_private_text_or_ids": False,
        "source": {
            "strict_contract_valid": 21,
            "unavailable": 3,
        },
        "recovery": {
            "request_attempts": len(attempts),
            "contract_valid_attempts": contract_valid_attempts,
            "recovered_complete": len(recovered_ids),
            "still_unavailable": 3 - len(recovered_ids),
            "provenance": dict(sorted(Counter(provenance.values()).items())),
        },
        "canonical": {
            "candidates": 24,
            "strict_contract_valid": canonical_contract_valid,
            "strict_process_negatives": strict_negatives,
            "dispositions": dict(sorted(dispositions.items())),
            "by_origin": {
                key: dict(sorted(value.items()))
                for key, value in sorted(by_origin.items())
            },
            "by_benchmark": {
                key: dict(sorted(value.items()))
                for key, value in sorted(by_benchmark.items())
            },
            "controlled": {
                "strict_negatives": controlled_negative,
                "exact_intended_first_error": controlled_exact,
                "exact_match_rate": (
                    controlled_exact / controlled_negative
                    if controlled_negative else None
                ),
            },
        },
        "cost": {
            "source_canary_cny_equivalent": source_cost,
            "recovery_cny_equivalent": recovery_cost,
            "combined_cny_equivalent": round(source_cost + recovery_cost, 8),
            "recovery_hard_cap_cny_equivalent": config["budget"][
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
        },
        "decision": {
            "training_merge_authorized": False,
            "full_scale_generation_authorized": False,
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    recovery = report["recovery"]
    canonical = report["canonical"]
    return "\n".join(
        [
            "# PRM negative validator recovery audit",
            "",
            "Aggregate-only report; no private question, ID, or trajectory text.",
            "",
            f"- Recovery attempts / recovered: "
            f"{recovery['request_attempts']} / {recovery['recovered_complete']}",
            f"- Still unavailable: {recovery['still_unavailable']}",
            f"- Canonical strict contract valid: "
            f"{canonical['strict_contract_valid']} / 24",
            f"- Canonical strict process negatives: "
            f"{canonical['strict_process_negatives']}",
            f"- Canonical dispositions: `"
            f"{json.dumps(canonical['dispositions'], sort_keys=True)}`",
            f"- By origin: `{json.dumps(canonical['by_origin'], sort_keys=True)}`",
            f"- By benchmark: "
            f"`{json.dumps(canonical['by_benchmark'], sort_keys=True)}`",
            f"- Controlled target match: "
            f"`{json.dumps(canonical['controlled'], sort_keys=True)}`",
            f"- Recovery / combined cost: CNY "
            f"{report['cost']['recovery_cny_equivalent']:.8f} / "
            f"{report['cost']['combined_cny_equivalent']:.8f}",
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
    report = audit_recovery(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = REPO_ROOT / str(config["source_run_dir"]) / str(
        config["output_subdir"]
    )
    _write_json(output_dir / "quality_audit.json", report)
    markdown = render_markdown(report)
    (output_dir / "quality_audit.md").write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    if not report["integrity"]["passed"]:
        raise RuntimeError("PRM validator recovery integrity checks failed")


if __name__ == "__main__":
    main()
