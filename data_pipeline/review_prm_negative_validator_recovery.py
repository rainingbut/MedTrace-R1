"""Render the private canonical validator key after PRM 429 recovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from data_pipeline.cot_api import validate_validator_result
from data_pipeline.prm_negative_human_review_state import (
    load_annotation_jsonl,
    load_human_review_config,
    load_review_context,
    validate_annotation_lock,
    validate_annotations,
)
from data_pipeline.prm_negative_policy import verification_disposition
from data_pipeline.prm_negative_recovery_config import (
    validate_prm_negative_recovery_config,
)
from data_pipeline.prm_negative_recovery_state import (
    canonical_event_map,
    load_source_state,
)
from data_pipeline.run_cot_pilot_real import _load_jsonl


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/cot/prm_negative_validator_recovery_v1.yaml",
    )
    parser.add_argument(
        "--human-review-config",
        default="configs/cot/prm_negative_human_review_v2.yaml",
    )
    return parser.parse_args()


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def build_recovered_key(config_path: Path) -> str:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("PRM negative recovery config must be an object")
    validate_prm_negative_recovery_config(config)
    source = load_source_state(config, REPO_ROOT)
    output_dir = REPO_ROOT / str(config["source_run_dir"]) / str(
        config["output_subdir"]
    )
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("status") != "complete":
        raise RuntimeError("PRM validator recovery run is not terminal")
    attempts = _load_jsonl(output_dir / "recovery_attempts.jsonl")
    canonical, provenance = canonical_event_map(source, attempts)
    for candidate_id, candidate in source["candidate_by_id"].items():
        event = canonical[candidate_id]
        if event.get("status") == "complete":
            validate_validator_result(
                event.get("result"), len(candidate["trajectory"]["steps"])
            )
    key = [
        "# PRM negative canary: canonical validator key after recovery",
        "",
        "PRIVATE: open only after completing human_review_blind.md. Do not commit.",
        "The existing blind-review file is intentionally not overwritten.",
        "",
    ]
    for case_number, candidate in enumerate(source["candidates"], start=1):
        candidate_id = str(candidate["candidate_id"])
        event = canonical[candidate_id]
        result = event.get("result")
        key.extend(
            [
                f"## Case {case_number:02d}",
                "",
                f"Origin: {candidate['origin']}",
                f"Intended error step: {candidate['intended_error_step']}",
                f"Validator provenance: {provenance[candidate_id]}",
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
    return "\n".join(key)


def main() -> None:
    args = parse_args()
    config_path = _repo_path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    human_config_path = _repo_path(args.human_review_config)
    human_config = load_human_review_config(human_config_path)
    if _repo_path(str(human_config["source_recovery_config"])) != config_path:
        raise RuntimeError("human-review lock targets a different recovery config")
    context = load_review_context(human_config, REPO_ROOT)
    files = human_config["private_files"]
    annotation_path = context["canary_dir"] / str(files["annotations"])
    records = load_annotation_jsonl(annotation_path)
    metadata, _ = validate_annotations(records, context, require_complete=True)
    lock_path = context["canary_dir"] / str(files["annotation_lock"])
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    validate_annotation_lock(lock, annotation_path, context, metadata)
    key = build_recovered_key(config_path)
    output_dir = REPO_ROOT / str(config["source_run_dir"]) / str(
        config["source_canary_subdir"]
    )
    target = output_dir / str(files["recovered_key"])
    target.write_text(key, encoding="utf-8")
    print(f"Wrote private recovered review key: {target}")
    print("The existing human_review_blind.md was not modified.")


if __name__ == "__main__":
    main()
