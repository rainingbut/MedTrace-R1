import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from data_pipeline.audit_prm_negative_validator_recovery import audit_recovery
from data_pipeline.audit_prm_negative_human_review import score_review
from data_pipeline.prm_negative_human_review_config import (
    validate_prm_negative_human_review_config,
)
from data_pipeline.prm_negative_human_review_state import (
    annotation_template,
    expected_lock,
    load_review_context,
    validate_annotation_lock,
    validate_annotations,
)
from data_pipeline.prm_negative_recovery_config import (
    validate_prm_negative_recovery_config,
)
from data_pipeline.prm_negative_recovery_state import (
    canonical_event_map,
    load_source_state,
    sha256_file,
    source_canary_hashes,
)
from data_pipeline.review_prm_negative_validator_recovery import (
    build_recovered_key,
)
from data_pipeline.run_prm_negative_validator_recovery import (
    _validate_existing_attempts,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/cot/prm_negative_validator_recovery_v1.yaml"
HUMAN_CONFIG_PATH = ROOT / "configs/cot/prm_negative_human_review_v2.yaml"


def validator_result(label: int, first: int | None = None) -> dict:
    return {
        "trajectory_label": label,
        "first_error_step": first,
        "answer_consistent": True,
        "problem_status": "ok",
        "steps": [
            {
                "index": index,
                "local_verdict": "incorrect" if index == first else "correct",
                "prefix_label": int(first is None or index < first),
                "error_codes": ["medical_fact_error"] if index == first else [],
                "concise_reason": "private reason",
            }
            for index in range(3)
        ],
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def create_fixture(root: Path) -> tuple[dict, dict]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    recovery_config_target = (
        root / "configs/cot/prm_negative_validator_recovery_v1.yaml"
    )
    recovery_config_target.parent.mkdir(parents=True, exist_ok=True)
    recovery_config_target.write_text(
        CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8"
    )
    run_dir = root / config["source_run_dir"]
    canary_dir = run_dir / config["source_canary_subdir"]
    recovery_dir = run_dir / config["output_subdir"]
    canary_dir.mkdir(parents=True)
    candidates = []
    source_events = []
    teachers = []
    origins = (
        ["controlled_single_error"] * 8
        + ["existing_teacher_answer_mismatch"] * 8
        + ["local_student"] * 8
    )
    source_cost = 0.0
    for index, origin in enumerate(origins):
        candidate_id = f"candidate-{index:02d}"
        source_hash = f"{index:064x}"
        benchmark = "medqa" if index % 2 == 0 else "medmcqa"
        candidates.append(
            {
                "candidate_id": candidate_id,
                "origin": origin,
                "source": {
                    "dataset": benchmark,
                    "content_sha256": source_hash,
                },
                "trajectory": {
                    "steps": ["private one", "private two", "private three"],
                    "predicted_answer": "A",
                },
                "intended_error_step": 1 if origin == "controlled_single_error" else None,
            }
        )
        teachers.append(
            {
                "record": {
                    "content_sha256": source_hash,
                    "question": "private question",
                    "choices": {"A": "private a", "B": "private b"},
                    "answer": "A",
                }
            }
        )
        if index < 3:
            event_cost = 0.32
            source_events.append(
                {
                    "candidate_id": candidate_id,
                    "status": "api_or_parse_error",
                    "result": None,
                    "usage": {"cost_cny": event_cost},
                    "attempt_diagnostics": [
                        {
                            "status": "failed",
                            "error_category": "http_429",
                            "content_present": False,
                        }
                    ],
                }
            )
        else:
            event_cost = 0.01
            result = validator_result(0, 1) if index < 12 else validator_result(1)
            source_events.append(
                {
                    "candidate_id": candidate_id,
                    "status": "complete",
                    "result": result,
                    "usage": {"cost_cny": event_cost},
                }
            )
        source_cost += event_cost
    write_jsonl(run_dir / "teacher_events.jsonl", teachers)
    write_jsonl(canary_dir / "candidates.jsonl", candidates)
    write_jsonl(canary_dir / "validator_events.jsonl", source_events)
    (canary_dir / "metadata.json").write_text(
        json.dumps({"status": "complete", "spent_cny_equivalent": source_cost}),
        encoding="utf-8",
    )
    (canary_dir / "private_manifest.json").write_text(
        json.dumps({"source_artifact_sha256": {"pilot": "hash"}}),
        encoding="utf-8",
    )
    source_state = load_source_state(config, root)
    attempts = []
    for candidate_id in source_state["selected_ids"]:
        candidate = source_state["candidate_by_id"][candidate_id]
        attempts.append(
            {
                "schema_version": (
                    "medtrace.prm-negative-validator-recovery-attempt.v1"
                ),
                "candidate_id": candidate_id,
                "recovery_attempt": 1,
                "origin": candidate["origin"],
                "benchmark": candidate["source"]["dataset"],
                "source_status": "api_or_parse_error",
                "status": "complete",
                "result": validator_result(0, 1),
                "usage": {"cost_cny": 0.01},
                "error_categories": [],
                "attempt_diagnostics": [],
            }
        )
    write_jsonl(recovery_dir / "recovery_attempts.jsonl", attempts)
    source_hashes = source_canary_hashes(canary_dir)
    (recovery_dir / "private_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": (
                    "medtrace.prm-negative-validator-recovery-manifest.v1"
                ),
                "config_sha256": sha256_file(CONFIG_PATH),
                "source_canary_sha256": source_hashes,
                "selected_candidate_ids": source_state["selected_ids"],
            }
        ),
        encoding="utf-8",
    )
    (recovery_dir / "metadata.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "request_attempts": 3,
                "terminal_events": 3,
                "recovered_complete": 3,
                "spent_cny_equivalent": 0.03,
            }
        ),
        encoding="utf-8",
    )
    return config, source_state


def create_human_annotations(root: Path) -> tuple[dict, dict, Path, dict]:
    human_config = yaml.safe_load(HUMAN_CONFIG_PATH.read_text(encoding="utf-8"))
    context = load_review_context(human_config, root)
    records = annotation_template(24)
    records[0].update(
        {
            "reviewer_role": "independent medical reviewer",
            "blinded_to_validator_outputs": True,
            "review_completed_at_utc": "2026-08-27T12:00:00+08:00",
        }
    )
    for candidate, annotation in zip(
        context["source"]["candidates"], records[1:], strict=True
    ):
        result = context["canonical"][candidate["candidate_id"]]["result"]
        annotation["human_problem_status"] = "ok"
        annotation["human_trajectory_label"] = result["trajectory_label"]
        annotation["human_error_type"] = (
            "process" if result["trajectory_label"] == 0 else None
        )
        annotation["human_first_error_step"] = result["first_error_step"]
    annotation_path = context["canary_dir"] / human_config["private_files"][
        "annotations"
    ]
    write_jsonl(annotation_path, records)
    metadata, _ = validate_annotations(records, context, require_complete=True)
    lock = expected_lock(
        annotation_path,
        context,
        metadata,
        locked_at_utc="2026-08-27T12:01:00+08:00",
    )
    lock_path = context["canary_dir"] / human_config["private_files"][
        "annotation_lock"
    ]
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    return human_config, context, annotation_path, lock


class PrmNegativeValidatorRecoveryTests(unittest.TestCase):
    def test_config_is_frozen_and_execution_needs_runtime_copy(self):
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        validate_prm_negative_recovery_config(config)
        runtime = dict(config)
        runtime["execution_enabled"] = True
        validate_prm_negative_recovery_config(runtime, execute=True)
        self.assertEqual(config["selection"]["total"], 3)
        self.assertEqual(config["budget"]["api_hard_cap_cny_equivalent"], 2)

    def test_source_selection_requires_exact_http_429_signature(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch(
                "data_pipeline.prm_negative_recovery_state._source_hashes",
                return_value={"pilot": "hash"},
            ):
                config, state = create_fixture(root)
                self.assertEqual(len(state["selected_ids"]), 3)
                path = state["canary_dir"] / "validator_events.jsonl"
                events = [json.loads(line) for line in path.read_text().splitlines()]
                events[0]["attempt_diagnostics"][0]["error_category"] = "timeout"
                write_jsonl(path, events)
                with self.assertRaisesRegex(RuntimeError, "signature"):
                    load_source_state(config, root)

    def test_attempt_validation_rejects_more_than_two_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch(
                "data_pipeline.prm_negative_recovery_state._source_hashes",
                return_value={"pilot": "hash"},
            ):
                _, state = create_fixture(root)
            candidate_id = state["selected_ids"][0]
            grouped = {
                candidate_id: [
                    {"recovery_attempt": number, "status": "api_or_parse_error"}
                    for number in (1, 2, 3)
                ]
            }
            with self.assertRaisesRegex(RuntimeError, "attempt cap"):
                _validate_existing_attempts(grouped, state, 2)

    def test_canonical_audit_passes_after_three_recoveries_without_private_text(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch(
                "data_pipeline.prm_negative_recovery_state._source_hashes",
                return_value={"pilot": "hash"},
            ), patch(
                "data_pipeline.audit_prm_negative_validator_recovery.REPO_ROOT",
                root,
            ):
                _, state = create_fixture(root)
                attempts = _load_attempts(
                    root / "results/cot/pilot_v1_real/"
                    "prm_negative_enrichment_v1/canary_v1/"
                    "validator_recovery_v1/recovery_attempts.jsonl"
                )
                canonical, provenance = canonical_event_map(state, attempts)
                self.assertEqual(
                    CounterLike(provenance.values())["validator_recovery_v1"], 3
                )
                self.assertEqual(len(canonical), 24)
                report = audit_recovery(CONFIG_PATH)
            serialized = json.dumps(report)
            self.assertTrue(report["integrity"]["passed"])
            self.assertTrue(report["machine_quality"]["passed"])
            self.assertEqual(report["canonical"]["strict_contract_valid"], 24)
            self.assertNotIn("private question", serialized)
            self.assertFalse(report["decision"]["training_merge_authorized"])

    def test_recovered_key_uses_canonical_results_without_overwriting_blind(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch(
                "data_pipeline.prm_negative_recovery_state._source_hashes",
                return_value={"pilot": "hash"},
            ), patch(
                "data_pipeline.review_prm_negative_validator_recovery.REPO_ROOT",
                root,
            ):
                create_fixture(root)
                key = build_recovered_key(CONFIG_PATH)
            self.assertIn("validator_recovery_v1", key)
            self.assertIn("strict_process_negative", key)
            self.assertIn("open only after completing", key)

    def test_human_review_config_and_annotation_contract_are_frozen(self):
        config = yaml.safe_load(HUMAN_CONFIG_PATH.read_text(encoding="utf-8"))
        validate_prm_negative_human_review_config(config)
        runtime = dict(config)
        runtime["write_enabled"] = True
        validate_prm_negative_human_review_config(runtime, write=True)
        records = annotation_template(24)
        records[0].update(
            {
                "reviewer_role": "medical reviewer",
                "blinded_to_validator_outputs": True,
                "review_completed_at_utc": "2026-08-27T12:00:00+08:00",
            }
        )
        records[1].update(
            {
                "human_problem_status": "ambiguous",
                "human_trajectory_label": 0,
                "human_error_type": "process",
                "human_first_error_step": 1,
            }
        )
        context = {
            "source": {
                "candidates": [
                    {"trajectory": {"steps": ["a", "b", "c"]}}
                    for _ in range(24)
                ]
            }
        }
        with self.assertRaisesRegex(ValueError, "non-ok"):
            validate_annotations(records, context, require_complete=True)

    def test_annotation_lock_detects_any_post_lock_edit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch(
                "data_pipeline.prm_negative_recovery_state._source_hashes",
                return_value={"pilot": "hash"},
            ):
                create_fixture(root)
                _, context, annotation_path, lock = create_human_annotations(root)
                records = _load_attempts(annotation_path)
                metadata, _ = validate_annotations(
                    records, context, require_complete=True
                )
                validate_annotation_lock(lock, annotation_path, context, metadata)
                records[1]["notes"] = "post-lock edit"
                write_jsonl(annotation_path, records)
                with self.assertRaisesRegex(RuntimeError, "lock"):
                    validate_annotation_lock(lock, annotation_path, context, metadata)

    def test_answer_only_negative_requires_null_first_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch(
                "data_pipeline.prm_negative_recovery_state._source_hashes",
                return_value={"pilot": "hash"},
            ):
                create_fixture(root)
                _, context, annotation_path, _ = create_human_annotations(root)
                records = _load_attempts(annotation_path)
                records[1]["human_error_type"] = "answer_only"
                records[1]["human_first_error_step"] = None
                validate_annotations(records, context, require_complete=True)
                records[1]["human_first_error_step"] = 1
                with self.assertRaisesRegex(ValueError, "answer-only"):
                    validate_annotations(records, context, require_complete=True)

    def test_human_review_scoring_passes_and_emits_only_candidate_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch(
                "data_pipeline.prm_negative_recovery_state._source_hashes",
                return_value={"pilot": "hash"},
            ):
                create_fixture(root)
                create_human_annotations(root)
                report, candidates = score_review(
                    HUMAN_CONFIG_PATH, repo_root=root
                )
            serialized = json.dumps(report)
            self.assertTrue(report["quality_gate"]["passed"])
            self.assertEqual(report["scores"]["trajectory_label"]["accuracy"], 1.0)
            self.assertEqual(report["scores"]["exact_first_error"]["accuracy"], 1.0)
            self.assertEqual(len(candidates), 12)
            self.assertNotIn("private question", serialized)
            self.assertTrue(
                all("trajectory" not in candidate for candidate in candidates)
            )
            self.assertFalse(report["decision"]["training_merge_authorized"])

    def test_human_review_gate_fails_below_trajectory_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch(
                "data_pipeline.prm_negative_recovery_state._source_hashes",
                return_value={"pilot": "hash"},
            ):
                create_fixture(root)
                _, context, annotation_path, _ = create_human_annotations(root)
                records = _load_attempts(annotation_path)
                for record in records[1:5]:
                    record["human_trajectory_label"] = 1
                    record["human_error_type"] = None
                    record["human_first_error_step"] = None
                write_jsonl(annotation_path, records)
                metadata, _ = validate_annotations(
                    records, context, require_complete=True
                )
                lock = expected_lock(
                    annotation_path, context, metadata,
                    locked_at_utc="2026-08-27T12:02:00+08:00",
                )
                lock_path = context["canary_dir"] / (
                    "human_review_annotations_v2.lock.json"
                )
                lock_path.write_text(json.dumps(lock), encoding="utf-8")
                report, _ = score_review(HUMAN_CONFIG_PATH, repo_root=root)
            self.assertFalse(report["quality_gate"]["passed"])
            self.assertIn(
                "trajectory_label_accuracy_at_least_90_percent",
                report["quality_gate"]["failed_checks"],
            )


def _load_attempts(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def CounterLike(values) -> dict:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


if __name__ == "__main__":
    unittest.main()
