"""Validate and hash-lock completed private PRM human annotations."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from data_pipeline.prm_negative_human_review_config import (
    validate_prm_negative_human_review_config,
)
from data_pipeline.prm_negative_human_review_state import (
    expected_lock,
    load_annotation_jsonl,
    load_human_review_config,
    load_review_context,
    validate_annotation_lock,
    validate_annotations,
)
from data_pipeline.run_cot_pilot_real import _write_json


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/cot/prm_negative_human_review_v2.yaml"
    )
    parser.add_argument("--lock-completed-review-24", action="store_true")
    return parser.parse_args()


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def main() -> None:
    args = parse_args()
    config_path = _repo_path(args.config)
    config = load_human_review_config(config_path)
    context = load_review_context(config, REPO_ROOT)
    files = config["private_files"]
    annotation_path = context["canary_dir"] / str(files["annotations"])
    records = load_annotation_jsonl(annotation_path)
    metadata, _ = validate_annotations(records, context, require_complete=True)
    preview = {
        "schema_version": "medtrace.prm-negative-human-review-lock-preview.v2",
        "contains_private_text_or_ids": False,
        "annotations_contract_valid": True,
        "cases": 24,
        "blind_attestation": True,
        "model_or_api_calls": 0,
    }
    print(json.dumps(preview, indent=2, sort_keys=True))
    if not args.lock_completed_review_24:
        print("Preview only; no lock file was written.")
        return
    runtime = dict(config)
    runtime["write_enabled"] = True
    validate_prm_negative_human_review_config(runtime, write=True)
    lock_path = context["canary_dir"] / str(files["annotation_lock"])
    if lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        validate_annotation_lock(lock, annotation_path, context, metadata)
        print(f"Annotation lock already exists and remains valid: {lock_path}")
        return
    lock = expected_lock(
        annotation_path,
        context,
        metadata,
        locked_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    _write_json(lock_path, lock)
    print(f"Locked completed private human annotations: {lock_path}")


if __name__ == "__main__":
    main()
