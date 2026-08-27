"""Create the approved private structured PRM blind-review template once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from data_pipeline.prm_negative_human_review_config import (
    validate_prm_negative_human_review_config,
)
from data_pipeline.prm_negative_human_review_state import (
    annotation_template,
    load_human_review_config,
    load_review_context,
    validate_annotations,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/cot/prm_negative_human_review_v1.yaml"
    )
    parser.add_argument("--prepare-template-24", action="store_true")
    return parser.parse_args()


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    config_path = _repo_path(args.config)
    config = load_human_review_config(config_path)
    context = load_review_context(config, REPO_ROOT)
    files = config["private_files"]
    blind_path = context["canary_dir"] / str(files["blind_review"])
    if not blind_path.is_file():
        raise RuntimeError("private blind-review Markdown is missing")
    if blind_path.read_text(encoding="utf-8").count("## Case ") != 24:
        raise RuntimeError("private blind-review Markdown case count changed")
    target = context["canary_dir"] / str(files["annotations"])
    preview = {
        "schema_version": "medtrace.prm-negative-human-review-preview.v1",
        "contains_private_text_or_ids": False,
        "cases": 24,
        "template_exists": target.exists(),
        "model_or_api_calls": 0,
    }
    print(json.dumps(preview, indent=2, sort_keys=True))
    if not args.prepare_template_24:
        print("Preview only; no file was written.")
        return
    runtime = dict(config)
    runtime["write_enabled"] = True
    validate_prm_negative_human_review_config(runtime, write=True)
    if target.exists():
        records = annotation_template(24)
        existing = [
            json.loads(line)
            for line in target.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        validate_annotations(existing, context, require_complete=False)
        if existing == records:
            print(f"Template already exists unchanged: {target}")
            return
        raise RuntimeError("human annotation file already exists; refusing to overwrite")
    records = annotation_template(24)
    _write_jsonl(target, records)
    print(f"Wrote private structured annotation template: {target}")


if __name__ == "__main__":
    main()
