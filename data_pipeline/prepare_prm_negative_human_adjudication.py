"""Create the private post-lock PRM disagreement-adjudication template."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from data_pipeline.prm_negative_human_adjudication_config import (
    validate_prm_negative_human_adjudication_config,
)
from data_pipeline.prm_negative_human_adjudication_state import (
    adjudication_template,
    load_adjudication_config,
    load_adjudication_context,
    validate_adjudications,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/cot/prm_negative_human_adjudication_v1.yaml",
    )
    parser.add_argument("--prepare-template-3", action="store_true")
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
    config = load_adjudication_config(_repo_path(args.config))
    context = load_adjudication_context(config, REPO_ROOT)
    target = context["canary_dir"] / str(config["private_files"]["adjudication"])
    template = adjudication_template(context["disagreements"])
    preview = {
        "schema_version": "medtrace.prm-negative-human-adjudication-preview.v1",
        "contains_private_text_or_ids": False,
        "source_blind_review_preserved": True,
        "disagreements": len(context["disagreements"]),
        "template_exists": target.exists(),
        "model_or_api_calls": 0,
    }
    print(json.dumps(preview, indent=2, sort_keys=True))
    if not args.prepare_template_3:
        print("Preview only; no adjudication file was written.")
        return
    runtime = dict(config)
    runtime["write_enabled"] = True
    validate_prm_negative_human_adjudication_config(runtime, write=True)
    if target.exists():
        existing = [
            json.loads(line)
            for line in target.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        validate_adjudications(existing, context, require_complete=False)
        if existing == template:
            print(f"Adjudication template already exists unchanged: {target}")
            return
        raise RuntimeError("adjudication file already exists; refusing to overwrite")
    _write_jsonl(target, template)
    print(f"Wrote private adjudication template: {target}")


if __name__ == "__main__":
    main()
