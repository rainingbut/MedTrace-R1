"""Audit locked PRM adjudications without rewriting raw blind-review metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from data_pipeline.prm_negative_human_adjudication_config import (
    validate_prm_negative_human_adjudication_config,
)
from data_pipeline.prm_negative_human_adjudication_state import (
    load_adjudication_config,
    load_adjudication_context,
    validate_adjudication_lock,
    validate_adjudications,
)
from data_pipeline.prm_negative_human_review_state import load_annotation_jsonl
from data_pipeline.prm_negative_policy import verification_disposition
from data_pipeline.run_cot_pilot_real import _write_json, _write_jsonl


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/cot/prm_negative_human_adjudication_v1.yaml",
    )
    parser.add_argument("--score-locked-adjudication-3", action="store_true")
    return parser.parse_args()


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def score_adjudication(
    config_path: Path, *, repo_root: Path = REPO_ROOT
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = load_adjudication_config(config_path)
    context = load_adjudication_context(config, repo_root)
    files = config["private_files"]
    adjudication_path = context["canary_dir"] / str(files["adjudication"])
    records = load_annotation_jsonl(adjudication_path)
    metadata, adjudications = validate_adjudications(
        records, context, require_complete=True
    )
    lock_path = context["canary_dir"] / str(files["adjudication_lock"])
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    validate_adjudication_lock(lock, adjudication_path, context, metadata)

    adjudication_by_case = {
        record["case_number"]: record for record in adjudications
    }
    review = context["review_context"]
    first_total = 0
    first_correct = 0
    decisions = {"human": 0, "validator": 0}
    candidates_out: list[dict[str, Any]] = []
    for case_number, (candidate, annotation) in enumerate(
        zip(review["source"]["candidates"], context["annotations"], strict=True),
        start=1,
    ):
        if annotation["human_error_type"] != "process":
            continue
        first_total += 1
        candidate_id = str(candidate["candidate_id"])
        result = review["canonical"][candidate_id]["result"]
        adjudication = adjudication_by_case.get(case_number)
        final_first = annotation["human_first_error_step"]
        if adjudication is not None:
            final_first = adjudication["adjudicated_first_error_step"]
            decisions[adjudication["decision_source"]] += 1
        exact = final_first == result["first_error_step"]
        first_correct += int(exact)
        if exact and verification_disposition(result) == "strict_process_negative":
            candidates_out.append(
                {
                    "schema_version": (
                        "medtrace.prm-negative-human-adjudicated-candidate.v1"
                    ),
                    "case_number": case_number,
                    "candidate_id": candidate_id,
                    "source": candidate["source"],
                    "origin": candidate["origin"],
                    "canonical_first_error_step": result["first_error_step"],
                    "final_human_first_error_step": final_first,
                    "adjudicated": adjudication is not None,
                    "validator_provenance": review["provenance"][candidate_id],
                    "disposition": "strict_process_negative",
                    "status": "candidate_only_not_merged",
                }
            )
    accuracy = first_correct / first_total if first_total else None
    scoring = config["scoring"]
    checks = {
        "raw_blind_review_preserved_at_9_of_12": (
            context["raw_audit"]["scores"]["exact_first_error"]["correct"] == 9
            and context["raw_audit"]["scores"]["exact_first_error"]["total"] == 12
        ),
        "all_3_disagreements_adjudicated_and_locked": len(adjudications) == 3,
        "adjudicated_exact_first_error_accuracy_at_least_80_percent": (
            accuracy is not None
            and accuracy
            >= float(scoring["minimum_adjudicated_exact_first_error_accuracy"])
        ),
        "conservative_negative_candidates_at_least_8": (
            len(candidates_out)
            >= int(scoring["minimum_conservative_negative_candidates"])
        ),
    }
    passed = all(checks.values())
    report = {
        "schema_version": "medtrace.prm-negative-human-adjudication-audit.v1",
        "contains_private_text_or_ids": False,
        "original_blind_review": {
            "trajectory_label_correct": 19,
            "trajectory_label_total": 19,
            "exact_first_error_correct": 9,
            "exact_first_error_total": 12,
            "exact_first_error_accuracy": 0.75,
            "quality_gate_passed": False,
            "preserved": True,
        },
        "adjudication": {
            "disagreements": len(adjudications),
            "decision_sources": decisions,
            "exact_first_error_correct": first_correct,
            "exact_first_error_total": first_total,
            "exact_first_error_accuracy": accuracy,
        },
        "candidate_negative_list": {
            "records": len(candidates_out),
            "contains_training_records": False,
            "status": "candidate_only_not_merged",
        },
        "quality_gate": {
            "checks": checks,
            "passed": passed,
            "failed_checks": sorted(k for k, v in checks.items() if not v),
        },
        "decision": {
            "candidate_list_authorized": passed,
            "training_merge_authorized": False,
            "full_scale_generation_authorized": False,
        },
    }
    return report, candidates_out


def render_markdown(report: dict[str, Any]) -> str:
    raw = report["original_blind_review"]
    adjudication = report["adjudication"]
    candidates = report["candidate_negative_list"]
    gate = report["quality_gate"]
    return "\n".join(
        [
            "# PRM negative human-adjudication audit",
            "",
            "Aggregate-only report; raw blind metrics remain immutable.",
            "",
            f"- Raw blind trajectory accuracy: {raw['trajectory_label_correct']}/"
            f"{raw['trajectory_label_total']} = 1.0",
            f"- Raw blind exact-first-error accuracy: "
            f"{raw['exact_first_error_correct']}/{raw['exact_first_error_total']} = "
            f"{raw['exact_first_error_accuracy']}",
            "- Raw blind quality passed: false",
            f"- Adjudicated disagreements: {adjudication['disagreements']}",
            f"- Adjudication decisions: "
            f"`{json.dumps(adjudication['decision_sources'], sort_keys=True)}`",
            f"- Adjudicated exact-first-error accuracy: "
            f"{adjudication['exact_first_error_correct']}/"
            f"{adjudication['exact_first_error_total']} = "
            f"{adjudication['exact_first_error_accuracy']}",
            f"- Conservative negative candidates: {candidates['records']}",
            "",
            "## Gates",
            "",
            f"- Adjudication quality passed: {str(gate['passed']).lower()}",
            f"- Failed checks: `{json.dumps(gate['failed_checks'])}`",
            "- Candidate list contains training records: false",
            "- Training merge authorized: false",
            "- Full-scale generation authorized: false",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    config_path = _repo_path(args.config)
    report, candidates = score_adjudication(config_path, repo_root=REPO_ROOT)
    print(render_markdown(report), end="")
    if not args.score_locked_adjudication_3:
        print("Preview only; no adjudication audit or candidate file was written.")
        return
    config = load_adjudication_config(config_path)
    runtime = dict(config)
    runtime["write_enabled"] = True
    validate_prm_negative_human_adjudication_config(runtime, write=True)
    context = load_adjudication_context(config, REPO_ROOT)
    files = config["private_files"]
    _write_json(context["canary_dir"] / str(files["aggregate_audit_json"]), report)
    markdown = render_markdown(report)
    (context["canary_dir"] / str(files["aggregate_audit_markdown"])).write_text(
        markdown, encoding="utf-8"
    )
    if report["quality_gate"]["passed"]:
        target = context["canary_dir"] / str(files["approved_negative_candidates"])
        if target.exists():
            existing = load_annotation_jsonl(target)
            if existing != candidates:
                raise RuntimeError("existing adjudicated candidate list changed")
        else:
            _write_jsonl(target, candidates)
    print("Wrote aggregate adjudication audit.")
    print("No SFT or PRM training record was written or merged.")


if __name__ == "__main__":
    main()
