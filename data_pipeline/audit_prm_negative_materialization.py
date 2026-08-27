"""Audit isolated PRM negative materialization and label balance."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from data_pipeline.materialize_prm_negative_candidates import (
    _public_stats,
    build_materialization,
)
from data_pipeline.prm_negative_human_review_state import load_annotation_jsonl
from data_pipeline.prm_negative_materialization_config import (
    validate_prm_negative_materialization_config,
)
from data_pipeline.prm_negative_materialization_state import (
    load_materialization_config,
    load_materialization_context,
    source_binding,
    validate_prm_record_set,
)
from data_pipeline.prm_negative_recovery_state import sha256_file
from data_pipeline.run_cot_pilot_real import _write_json


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/cot/prm_negative_materialization_v1.yaml"
    )
    parser.add_argument("--audit-materialized-11", action="store_true")
    return parser.parse_args()


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def audit_materialization(
    config_path: Path, *, repo_root: Path = REPO_ROOT
) -> dict[str, Any]:
    config = load_materialization_config(config_path)
    context = load_materialization_context(config, repo_root)
    expected_summary, expected_candidates, expected_enriched = build_materialization(
        config_path, repo_root=repo_root
    )
    files = config["output_files"]
    output_dir = context["output_dir"]
    manifest_path = output_dir / str(files["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError("PRM materialization is not complete")
    if manifest.get("source_binding") != source_binding(context):
        raise RuntimeError("PRM materialization source binding changed")
    if manifest.get("summary") != expected_summary:
        raise RuntimeError("PRM materialization summary changed")
    candidate_path = output_dir / str(files["candidate_prefix_records"])
    enriched_path = output_dir / str(files["enriched_process_train"])
    expected_hashes = {
        files["candidate_prefix_records"]: sha256_file(candidate_path),
        files["enriched_process_train"]: sha256_file(enriched_path),
    }
    if manifest.get("output_sha256") != expected_hashes:
        raise RuntimeError("PRM materialization output hashes changed")
    candidate_records = load_annotation_jsonl(candidate_path)
    enriched = load_annotation_jsonl(enriched_path)
    if candidate_records != expected_candidates or enriched != expected_enriched:
        raise RuntimeError("materialized PRM records differ from recomputation")
    candidate_stats = validate_prm_record_set(candidate_records)
    enriched_stats = validate_prm_record_set(enriched)
    candidate_labels = Counter(record["label"] for record in candidate_records)
    enriched_labels = Counter(record["label"] for record in enriched)
    new_negative = candidate_labels[0]
    enriched_negative = enriched_labels[0]
    gates = config["quality_gates"]
    origins = expected_summary["candidate_prefixes"]["origins"]
    benchmarks = expected_summary["candidate_prefixes"]["benchmarks"]
    checks = {
        "exactly_11_approved_negative_trajectories_materialized": (
            candidate_stats["trajectories"]
            == int(gates["exact_approved_trajectories"])
        ),
        "at_least_11_new_negative_prefix_records": (
            new_negative >= int(gates["minimum_negative_prefix_records"])
        ),
        "at_least_11_negative_trajectories": (
            candidate_stats["negative_trajectories"]
            >= int(gates["minimum_negative_trajectories"])
        ),
        "both_benchmarks_present": set(benchmarks) == {"medqa", "medmcqa"},
        "at_least_two_origins_present": len(origins) >= int(
            gates["minimum_negative_origins"]
        ),
        "zero_duplicate_candidate_records": (
            candidate_stats["duplicate_records"] == 0
        ),
        "zero_duplicate_enriched_records": enriched_stats["duplicate_records"] == 0,
        "zero_full_trajectory_overlap_with_strict": (
            expected_summary["full_trajectory_overlap_with_strict"] == 0
        ),
        "source_artifacts_unchanged": (
            manifest["source_binding"] == source_binding(context)
        ),
        "strict_integer_labels": all(
            type(record["label"]) is int for record in enriched
        ),
    }
    passed = all(checks.values())
    return {
        "schema_version": "medtrace.prm-negative-materialization-audit.v1",
        "contains_private_text_or_ids": False,
        "source_strict": {
            "records": len(context["strict_records"]),
            "labels": context["strict_stats"]["labels"],
            "negative_trajectories": context["strict_stats"][
                "negative_trajectories"
            ],
        },
        "materialized_candidates": _public_stats(candidate_stats) | {
            "origins": origins,
            "benchmarks": benchmarks,
        },
        "enriched_derivative": _public_stats(enriched_stats),
        "negative_prefix_increase": {
            "before": context["strict_stats"]["labels"]["0"],
            "added": new_negative,
            "after": enriched_negative,
        },
        "quality_gate": {
            "checks": checks,
            "passed": passed,
            "failed_checks": sorted(key for key, value in checks.items() if not value),
        },
        "decision": {
            "derivative_prm_records_created": True,
            "training_use_authorized": False,
            "sft_changed": False,
            "full_scale_generation_authorized": False,
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    source = report["source_strict"]
    candidate = report["materialized_candidates"]
    enriched = report["enriched_derivative"]
    increase = report["negative_prefix_increase"]
    gate = report["quality_gate"]
    return "\n".join(
        [
            "# PRM negative materialization audit",
            "",
            "Aggregate-only report; no question, ID, or trajectory text.",
            "",
            f"- Strict source records / labels: {source['records']} / "
            f"`{json.dumps(source['labels'], sort_keys=True)}`",
            f"- Materialized trajectories / prefix records: "
            f"{candidate['trajectories']} / {candidate['records']}",
            f"- Materialized labels: "
            f"`{json.dumps(candidate['labels'], sort_keys=True)}`",
            f"- Materialized origins: "
            f"`{json.dumps(candidate['origins'], sort_keys=True)}`",
            f"- Materialized benchmarks: "
            f"`{json.dumps(candidate['benchmarks'], sort_keys=True)}`",
            f"- Enriched derivative records / labels: {enriched['records']} / "
            f"`{json.dumps(enriched['labels'], sort_keys=True)}`",
            f"- Negative prefixes before / added / after: "
            f"{increase['before']} / {increase['added']} / {increase['after']}",
            "",
            "## Gates",
            "",
            f"- Materialization quality passed: {str(gate['passed']).lower()}",
            f"- Failed checks: `{json.dumps(gate['failed_checks'])}`",
            "- SFT changed: false",
            "- Training use authorized: false",
            "- Full-scale generation authorized: false",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    config_path = _repo_path(args.config)
    report = audit_materialization(config_path, repo_root=REPO_ROOT)
    print(render_markdown(report), end="")
    if not args.audit_materialized_11:
        print("Preview only; no aggregate materialization audit was written.")
        return
    config = load_materialization_config(config_path)
    runtime = dict(config)
    runtime["execution_enabled"] = True
    validate_prm_negative_materialization_config(runtime, execute=True)
    context = load_materialization_context(config, REPO_ROOT)
    files = config["output_files"]
    _write_json(context["output_dir"] / str(files["aggregate_audit_json"]), report)
    markdown = render_markdown(report)
    (context["output_dir"] / str(files["aggregate_audit_markdown"])).write_text(
        markdown, encoding="utf-8"
    )
    print("Wrote aggregate-only PRM materialization audit.")
    print("Derivative PRM records remain unauthorized for training use.")


if __name__ == "__main__":
    main()
