"""Render private blind-review and answer-key files for the PRM negative canary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from data_pipeline.prm_negative_canary_config import (
    validate_prm_negative_canary_config,
)
from data_pipeline.prm_negative_policy import verification_disposition
from data_pipeline.run_cot_pilot_real import _load_jsonl


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cot/prm_negative_canary_v1.yaml")
    return parser.parse_args()


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def build_review(config_path: Path) -> tuple[str, str]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("PRM negative canary config must be an object")
    validate_prm_negative_canary_config(config)
    run_dir = _repo_path(str(config["source_run_dir"]))
    output_dir = run_dir / str(config["output_subdir"])
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("status") != "complete":
        raise RuntimeError("PRM negative canary run is not complete")
    candidates = _load_jsonl(output_dir / "candidates.jsonl")
    events = {
        event["candidate_id"]: event
        for event in _load_jsonl(output_dir / "validator_events.jsonl")
    }
    records: dict[str, dict[str, Any]] = {}
    for event in _load_jsonl(run_dir / "teacher_events.jsonl"):
        record = event["record"]
        records.setdefault(str(record["content_sha256"]), record)
    if len(candidates) != 24 or set(events) != {
        candidate["candidate_id"] for candidate in candidates
    }:
        raise RuntimeError("PRM negative canary review inputs are incomplete")

    blind = [
        "# PRM negative canary: blind human review",
        "",
        "PRIVATE: contains licensed questions and derived trajectories. Do not commit.",
        "Validator verdict, candidate origin, and intended mutation are hidden.",
        "",
    ]
    key = [
        "# PRM negative canary: validator key",
        "",
        "PRIVATE: open only after completing the blind review. Do not commit.",
        "",
    ]
    for case_number, candidate in enumerate(candidates, start=1):
        record = records[candidate["source"]["content_sha256"]]
        event = events[candidate["candidate_id"]]
        result = event.get("result")
        blind.extend(
            [
                f"## Case {case_number:02d}",
                "",
                f"Benchmark: {candidate['source']['dataset']}",
                "",
                str(record["question"]),
                "",
                *[
                    f"- {label}. {text}"
                    for label, text in record["choices"].items()
                ],
                "",
                f"Gold answer: {record['answer']}",
                f"Candidate answer: {candidate['trajectory']['predicted_answer']}",
                "",
                *[
                    f"{index}. {text}"
                    for index, text in enumerate(candidate["trajectory"]["steps"])
                ],
                "",
                "Human trajectory label (0/1):",
                "Human first error step (integer/null):",
                "Human problem status (ok/ambiguous/bad_gold):",
                "Notes:",
                "",
            ]
        )
        key.extend(
            [
                f"## Case {case_number:02d}",
                "",
                f"Origin: {candidate['origin']}",
                f"Intended error step: {candidate['intended_error_step']}",
                f"Validator status: {event['status']}",
                f"Validator disposition: {verification_disposition(result)}",
                f"Validator trajectory label: "
                f"{result.get('trajectory_label') if isinstance(result, dict) else None}",
                f"Validator first error step: "
                f"{result.get('first_error_step') if isinstance(result, dict) else None}",
                f"Validator problem status: "
                f"{result.get('problem_status') if isinstance(result, dict) else None}",
                "",
            ]
        )
        if isinstance(result, dict):
            for step in result.get("steps") or []:
                key.append(
                    f"- Step {step['index']}: {step['local_verdict']}; "
                    f"prefix={step['prefix_label']}; "
                    f"reason={step['concise_reason']}"
                )
            key.append("")
    return "\n".join(blind), "\n".join(key)


def main() -> None:
    args = parse_args()
    config_path = _repo_path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("PRM negative canary config must be an object")
    validate_prm_negative_canary_config(config)
    blind, key = build_review(config_path)
    output_dir = _repo_path(str(config["source_run_dir"])) / str(
        config["output_subdir"]
    )
    (output_dir / "human_review_blind.md").write_text(blind, encoding="utf-8")
    (output_dir / "human_review_key.md").write_text(key, encoding="utf-8")
    print(f"Wrote private review files under: {output_dir}")


if __name__ == "__main__":
    main()
