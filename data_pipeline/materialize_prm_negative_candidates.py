"""Materialize 11 approved negative trajectories into isolated PRM prefixes."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from data_pipeline.prm_negative_materialization_config import (
    validate_prm_negative_materialization_config,
)
from data_pipeline.prm_negative_materialization_state import (
    build_candidate_prefix_records,
    load_materialization_config,
    load_materialization_context,
    source_binding,
    validate_prm_record_set,
)
from data_pipeline.prm_negative_recovery_state import sha256_file
from data_pipeline.run_cot_pilot_real import _write_json, _write_jsonl


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/cot/prm_negative_materialization_v1.yaml"
    )
    parser.add_argument("--materialize-approved-11", action="store_true")
    return parser.parse_args()


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _public_stats(stats: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in stats.items() if key != "fingerprints"}


def build_materialization(
    config_path: Path, *, repo_root: Path = REPO_ROOT
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    config = load_materialization_config(config_path)
    context = load_materialization_context(config, repo_root)
    review = context["adjudication_context"]["review_context"]
    candidate_records, candidate_stats = build_candidate_prefix_records(
        context["approved"],
        review["source"]["candidate_by_id"],
        review["canonical"],
    )
    strict_fingerprints = set(context["strict_stats"]["fingerprints"].values())
    candidate_fingerprints = set(candidate_stats["fingerprints"].values())
    full_trajectory_overlap = len(strict_fingerprints & candidate_fingerprints)
    if full_trajectory_overlap:
        raise RuntimeError("approved candidates overlap the strict source")
    enriched = [*context["strict_records"], *candidate_records]
    enriched_stats = validate_prm_record_set(enriched)
    if enriched_stats["duplicate_records"]:
        raise RuntimeError("enriched PRM records contain duplicate keys")
    expected_labels = Counter(record["label"] for record in enriched)
    if enriched_stats["labels"] != {
        str(key): expected_labels[key] for key in sorted(expected_labels)
    }:
        raise RuntimeError("enriched PRM label accounting changed")
    summary = {
        "schema_version": "medtrace.prm-negative-materialization-preview.v1",
        "contains_private_text_or_ids": False,
        "model_or_api_calls": 0,
        "gpu_inference": False,
        "strict_source": _public_stats(context["strict_stats"]),
        "candidate_prefixes": _public_stats(candidate_stats),
        "enriched": _public_stats(enriched_stats),
        "full_trajectory_overlap_with_strict": full_trajectory_overlap,
        "source_artifacts_unchanged": True,
        "sft_records_written": 0,
        "training_use_authorized": False,
    }
    return summary, candidate_records, enriched


def _existing_output_matches(
    output_dir: Path,
    files: dict[str, str],
    candidate_records: list[dict[str, Any]],
    enriched: list[dict[str, Any]],
    context: dict[str, Any],
    summary: dict[str, Any],
) -> bool:
    manifest_path = output_dir / files["manifest"]
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidate_path = output_dir / files["candidate_prefix_records"]
    enriched_path = output_dir / files["enriched_process_train"]
    if not candidate_path.is_file() or not enriched_path.is_file():
        raise RuntimeError("materialization manifest has missing output files")
    from data_pipeline.prm_negative_human_review_state import load_annotation_jsonl

    expected_hashes = {
        files["candidate_prefix_records"]: sha256_file(candidate_path),
        files["enriched_process_train"]: sha256_file(enriched_path),
    }
    return (
        manifest.get("status") == "complete"
        and manifest.get("source_binding") == source_binding(context)
        and manifest.get("summary") == summary
        and manifest.get("output_sha256") == expected_hashes
        and load_annotation_jsonl(candidate_path) == candidate_records
        and load_annotation_jsonl(enriched_path) == enriched
    )


def main() -> None:
    args = parse_args()
    config_path = _repo_path(args.config)
    summary, candidate_records, enriched = build_materialization(
        config_path, repo_root=REPO_ROOT
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if not args.materialize_approved_11:
        print("Preview only; no PRM record was written.")
        return
    config = load_materialization_config(config_path)
    runtime = dict(config)
    runtime["execution_enabled"] = True
    validate_prm_negative_materialization_config(runtime, execute=True)
    context = load_materialization_context(config, REPO_ROOT)
    output_dir = context["output_dir"]
    files = config["output_files"]
    if _existing_output_matches(
        output_dir, files, candidate_records, enriched, context, summary
    ):
        print(f"Materialized PRM outputs already exist unchanged: {output_dir}")
        return
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("partial or changed materialization output exists")
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / str(files["candidate_prefix_records"])
    enriched_path = output_dir / str(files["enriched_process_train"])
    _write_jsonl(candidate_path, candidate_records)
    _write_jsonl(enriched_path, enriched)
    if source_binding(context) != source_binding(
        load_materialization_context(config, REPO_ROOT)
    ):
        raise RuntimeError("source artifacts changed during materialization")
    manifest = {
        "schema_version": "medtrace.prm-negative-materialization-manifest.v1",
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contains_private_text_or_ids": False,
        "model_or_api_calls": 0,
        "gpu_inference": False,
        "source_binding": source_binding(context),
        "summary": summary,
        "output_sha256": {
            files["candidate_prefix_records"]: sha256_file(candidate_path),
            files["enriched_process_train"]: sha256_file(enriched_path),
        },
        "training_use_authorized": False,
        "full_scale_generation_authorized": False,
    }
    _write_json(output_dir / str(files["manifest"]), manifest)
    print(f"Materialized isolated PRM candidate prefixes: {output_dir}")
    print("No SFT record was written and training use remains unauthorized.")


if __name__ == "__main__":
    main()
