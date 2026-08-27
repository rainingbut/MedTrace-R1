"""Score locked private PRM human review and emit aggregate-only gates."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from data_pipeline.prm_negative_human_review_config import (
    validate_prm_negative_human_review_config,
)
from data_pipeline.prm_negative_human_review_state import (
    load_annotation_jsonl,
    load_human_review_config,
    load_review_context,
    validate_annotation_lock,
    validate_annotations,
)
from data_pipeline.prm_negative_policy import verification_disposition
from data_pipeline.run_cot_pilot_real import _write_json, _write_jsonl


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/cot/prm_negative_human_review_v1.yaml"
    )
    parser.add_argument("--score-locked-review-24", action="store_true")
    return parser.parse_args()


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def score_review(
    config_path: Path, *, repo_root: Path = REPO_ROOT
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = load_human_review_config(config_path)
    context = load_review_context(config, repo_root)
    files = config["private_files"]
    annotation_path = context["canary_dir"] / str(files["annotations"])
    lock_path = context["canary_dir"] / str(files["annotation_lock"])
    records = load_annotation_jsonl(annotation_path)
    metadata, annotations = validate_annotations(
        records, context, require_complete=True
    )
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    validate_annotation_lock(lock, annotation_path, context, metadata)

    problem_statuses: Counter[str] = Counter()
    validator_dispositions: Counter[str] = Counter()
    label_matrix: Counter[str] = Counter()
    trajectory_total = 0
    trajectory_correct = 0
    first_error_total = 0
    first_error_correct = 0
    candidates_out: list[dict[str, Any]] = []
    for case_number, (candidate, annotation) in enumerate(
        zip(context["source"]["candidates"], annotations, strict=True), start=1
    ):
        candidate_id = str(candidate["candidate_id"])
        result = context["canonical"][candidate_id]["result"]
        disposition = verification_disposition(result)
        validator_dispositions[disposition] += 1
        human_status = str(annotation["human_problem_status"])
        problem_statuses[human_status] += 1
        if human_status != "ok":
            continue
        human_label = int(annotation["human_trajectory_label"])
        validator_label = int(result["trajectory_label"])
        trajectory_total += 1
        trajectory_correct += int(human_label == validator_label)
        label_matrix[f"human:{human_label}|validator:{validator_label}"] += 1
        exact_first = False
        if human_label == 0:
            first_error_total += 1
            exact_first = (
                result.get("first_error_step")
                == annotation["human_first_error_step"]
            )
            first_error_correct += int(exact_first)
        if (
            human_label == 0
            and disposition == "strict_process_negative"
            and exact_first
        ):
            candidates_out.append(
                {
                    "schema_version": (
                        "medtrace.prm-negative-human-approved-candidate.v1"
                    ),
                    "case_number": case_number,
                    "candidate_id": candidate_id,
                    "source": candidate["source"],
                    "origin": candidate["origin"],
                    "canonical_first_error_step": result["first_error_step"],
                    "human_first_error_step": annotation["human_first_error_step"],
                    "validator_provenance": context["provenance"][candidate_id],
                    "disposition": disposition,
                    "status": "candidate_only_not_merged",
                }
            )
    trajectory_accuracy = (
        trajectory_correct / trajectory_total if trajectory_total else None
    )
    first_error_accuracy = (
        first_error_correct / first_error_total if first_error_total else None
    )
    scoring = config["scoring"]
    checks = {
        "all_24_annotations_locked_and_contract_valid": len(annotations) == 24,
        "blind_attestation_true": metadata["blinded_to_validator_outputs"] is True,
        "trajectory_label_accuracy_at_least_90_percent": (
            trajectory_accuracy is not None
            and trajectory_accuracy
            >= float(scoring["minimum_trajectory_label_accuracy"])
        ),
        "exact_first_error_accuracy_at_least_80_percent": (
            first_error_accuracy is not None
            and first_error_accuracy
            >= float(scoring["minimum_exact_first_error_accuracy"])
        ),
    }
    report = {
        "schema_version": "medtrace.prm-negative-human-review-audit.v1",
        "contains_private_text_or_ids": False,
        "annotations": {
            "total": len(annotations),
            "problem_status": dict(sorted(problem_statuses.items())),
            "human_problem_ok": trajectory_total,
            "human_negative_on_ok_problems": first_error_total,
        },
        "validator": {
            "canonical_dispositions": dict(sorted(validator_dispositions.items())),
            "label_confusion": dict(sorted(label_matrix.items())),
        },
        "scores": {
            "trajectory_label": {
                "correct": trajectory_correct,
                "total": trajectory_total,
                "accuracy": trajectory_accuracy,
                "minimum": scoring["minimum_trajectory_label_accuracy"],
            },
            "exact_first_error": {
                "correct": first_error_correct,
                "total": first_error_total,
                "accuracy": first_error_accuracy,
                "minimum": scoring["minimum_exact_first_error_accuracy"],
            },
        },
        "candidate_negative_list": {
            "records": len(candidates_out),
            "policy": "human_ok_negative_and_validator_strict_negative_exact_first",
            "contains_training_records": False,
        },
        "quality_gate": {
            "checks": checks,
            "passed": all(checks.values()),
            "failed_checks": sorted(key for key, value in checks.items() if not value),
        },
        "decision": {
            "candidate_list_authorized": all(checks.values()),
            "training_merge_authorized": False,
            "full_scale_generation_authorized": False,
        },
    }
    return report, candidates_out


def render_markdown(report: dict[str, Any]) -> str:
    annotations = report["annotations"]
    scores = report["scores"]
    return "\n".join(
        [
            "# PRM negative human-review audit",
            "",
            "Aggregate-only report; no private question, ID, trajectory, or notes.",
            "",
            f"- Annotations: {annotations['total']}",
            f"- Human problem status: "
            f"`{json.dumps(annotations['problem_status'], sort_keys=True)}`",
            f"- Human problem-ok cases: {annotations['human_problem_ok']}",
            f"- Human negatives on ok problems: "
            f"{annotations['human_negative_on_ok_problems']}",
            f"- Trajectory-label accuracy: "
            f"{scores['trajectory_label']['correct']}/"
            f"{scores['trajectory_label']['total']} = "
            f"{scores['trajectory_label']['accuracy']}",
            f"- Exact-first-error accuracy: "
            f"{scores['exact_first_error']['correct']}/"
            f"{scores['exact_first_error']['total']} = "
            f"{scores['exact_first_error']['accuracy']}",
            f"- Conservative negative candidates: "
            f"{report['candidate_negative_list']['records']}",
            "",
            "## Gates",
            "",
            f"- Human quality passed: "
            f"{str(report['quality_gate']['passed']).lower()}",
            f"- Failed checks: `"
            f"{json.dumps(report['quality_gate']['failed_checks'])}`",
            "- Candidate list contains training records: false",
            "- Training merge authorized: false",
            "- Full-scale generation authorized: false",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    config_path = _repo_path(args.config)
    report, candidates = score_review(config_path, repo_root=REPO_ROOT)
    print(render_markdown(report), end="")
    if not args.score_locked_review_24:
        print("Preview only; no audit or candidate file was written.")
        return
    config = load_human_review_config(config_path)
    runtime = dict(config)
    runtime["write_enabled"] = True
    validate_prm_negative_human_review_config(runtime, write=True)
    context = load_review_context(config, REPO_ROOT)
    files = config["private_files"]
    _write_json(
        context["canary_dir"] / str(files["aggregate_audit_json"]), report
    )
    markdown = render_markdown(report)
    (context["canary_dir"] / str(files["aggregate_audit_markdown"])).write_text(
        markdown, encoding="utf-8"
    )
    if report["quality_gate"]["passed"]:
        target = context["canary_dir"] / str(
            files["approved_negative_candidates"]
        )
        if target.exists():
            existing = [
                json.loads(line)
                for line in target.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if existing != candidates:
                raise RuntimeError("existing approved candidate list changed")
        else:
            _write_jsonl(target, candidates)
    print("Wrote aggregate human-review audit.")
    print("No SFT or PRM training record was written or merged.")


if __name__ == "__main__":
    main()
