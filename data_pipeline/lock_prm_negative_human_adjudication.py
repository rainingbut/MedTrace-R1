"""Validate and hash-lock completed private PRM human adjudications."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from data_pipeline.prm_negative_human_adjudication_config import (
    validate_prm_negative_human_adjudication_config,
)
from data_pipeline.prm_negative_human_adjudication_state import (
    expected_adjudication_lock,
    load_adjudication_config,
    load_adjudication_context,
    validate_adjudication_lock,
    validate_adjudications,
)
from data_pipeline.prm_negative_human_review_state import load_annotation_jsonl
from data_pipeline.run_cot_pilot_real import _write_json


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/cot/prm_negative_human_adjudication_v1.yaml",
    )
    parser.add_argument("--lock-completed-adjudication-3", action="store_true")
    return parser.parse_args()


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def main() -> None:
    args = parse_args()
    config = load_adjudication_config(_repo_path(args.config))
    context = load_adjudication_context(config, REPO_ROOT)
    files = config["private_files"]
    adjudication_path = context["canary_dir"] / str(files["adjudication"])
    records = load_annotation_jsonl(adjudication_path)
    metadata, _ = validate_adjudications(records, context, require_complete=True)
    preview = {
        "schema_version": (
            "medtrace.prm-negative-human-adjudication-lock-preview.v1"
        ),
        "contains_private_text_or_ids": False,
        "adjudications_contract_valid": True,
        "disagreements": len(context["disagreements"]),
        "original_blind_review_preserved": True,
        "model_or_api_calls": 0,
    }
    print(json.dumps(preview, indent=2, sort_keys=True))
    if not args.lock_completed_adjudication_3:
        print("Preview only; no adjudication lock was written.")
        return
    runtime = dict(config)
    runtime["write_enabled"] = True
    validate_prm_negative_human_adjudication_config(runtime, write=True)
    lock_path = context["canary_dir"] / str(files["adjudication_lock"])
    if lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        validate_adjudication_lock(
            lock, adjudication_path, context, metadata
        )
        print(f"Adjudication lock already exists and remains valid: {lock_path}")
        return
    lock = expected_adjudication_lock(
        adjudication_path,
        context,
        metadata,
        locked_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    _write_json(lock_path, lock)
    print(f"Locked completed private human adjudication: {lock_path}")


if __name__ == "__main__":
    main()
