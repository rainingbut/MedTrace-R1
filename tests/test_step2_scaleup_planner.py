import copy
import json
from pathlib import Path
import tempfile
import unittest

from data_pipeline.step2_scaleup_planner import (
    build_plan,
    collect_observations,
    load_config,
    render_markdown,
    validate_config,
    wilson_lower_bound,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/cot/step2_scaleup_plan_v1.yaml"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def make_private_fixture(root: Path, config: dict) -> Path:
    run_dir = root / "pilot_v1_real"
    files = config["source_files"]

    teachers = []
    for index in range(160):
        benchmark = "medqa" if index < 80 else "medmcqa"
        teachers.append(
            {
                "record_id": f"private-{index}",
                "candidate_index": index % 4,
                "record": {
                    "benchmark": benchmark,
                    "split": "train",
                    "question": "SECRET MEDICAL QUESTION",
                },
                "attempts": 1,
                "usage": {"input_tokens": 100, "output_tokens": 200, "cost_cny": 0.01},
            }
        )
    screeners = [
        {"record_id": f"private-{index}", "usage": {"cost_cny": 0}}
        for index in range(120)
    ]
    validators = [
        {
            "record_id": f"private-{index}",
            "attempts": 1,
            "usage": {"input_tokens": 300, "output_tokens": 400, "cost_cny": 0.02},
        }
        for index in range(120)
    ]
    strict_sft = [
        {
            "trajectory_id": f"secret-{index}",
            "source": {
                "dataset": "medqa" if index < 54 else "medmcqa",
                "split": "train",
            },
            "trajectory": {"steps": ["SECRET TRAJECTORY"]},
        }
        for index in range(107)
    ]
    pilot_audit = {
        "contains_private_text": False,
        "invariants_passed": True,
        "counts": {
            "questions": 40,
            "rule_passed": 120,
            "screener_events": 120,
            "validator_events": 120,
            "canonical_trajectories": 100,
        },
        "cost": {"total_cny": 4.0},
    }

    origin_order = [
        "existing_teacher_answer_mismatch",
        "local_student",
        "controlled_single_error",
    ]
    candidates = [
        {
            "candidate_id": f"private-candidate-{origin}-{index}",
            "origin": origin,
            "trajectory": {"steps": ["SECRET NEGATIVE"]},
        }
        for origin in origin_order
        for index in range(8)
    ]
    canary_validators = [
        {
            "candidate_id": candidate["candidate_id"],
            "usage": {"input_tokens": 500, "output_tokens": 600, "cost_cny": 0.05},
        }
        for candidate in candidates
    ]
    recovery_attempts = [
        {
            "candidate_id": f"private-recovery-{index}",
            "usage": {"input_tokens": 500, "output_tokens": 600, "cost_cny": 0.04},
        }
        for index in range(3)
    ]
    student_events = [
        {"attempt_id": f"private-student-{index}", "candidate": {"secret": True}}
        for index in range(8)
    ]
    controlled_events = [
        {"attempt_id": f"private-controlled-{index}", "candidate": {"secret": True}}
        for index in range(8)
    ]
    recovery_audit = {
        "integrity": {"passed": True},
        "machine_quality": {"passed": True},
        "canonical": {
            "strict_contract_valid": 24,
            "strict_process_negatives": 12,
            "controlled": {
                "exact_intended_first_error": 7,
                "strict_negatives": 8,
                "exact_match_rate": 0.875,
            },
        },
        "cost": {"combined_cny_equivalent": 1.32},
    }
    materialization_audit = {
        "quality_gate": {"passed": True},
        "materialized_candidates": {
            "negative_trajectories": 11,
            "labels": {"0": 43, "1": 32},
            "origins": {
                "controlled_single_error": 7,
                "existing_teacher_answer_mismatch": 2,
                "local_student": 2,
            },
        },
        "enriched_derivative": {
            "negative_trajectories": 12,
            "labels": {"0": 47, "1": 749},
        },
    }

    write_json(run_dir / files["pilot_audit"], pilot_audit)
    write_jsonl(run_dir / files["teacher_events"], teachers)
    write_jsonl(run_dir / files["screener_events"], screeners)
    write_jsonl(run_dir / files["validator_events"], validators)
    write_jsonl(run_dir / files["strict_sft"], strict_sft)
    write_json(run_dir / files["canary_audit"], {"integrity": {"passed": True}})
    write_jsonl(run_dir / files["canary_candidates"], candidates)
    write_jsonl(run_dir / files["canary_validator_events"], canary_validators)
    write_jsonl(run_dir / files["student_generation_events"], student_events)
    write_jsonl(run_dir / files["controlled_generation_events"], controlled_events)
    write_json(run_dir / files["recovery_audit"], recovery_audit)
    write_jsonl(run_dir / files["recovery_attempts"], recovery_attempts)
    write_json(run_dir / files["adjudication_audit"], {"quality_gate": {"passed": True}})
    write_json(run_dir / files["materialization_audit"], materialization_audit)
    return run_dir


class Step2ScaleupPlannerTests(unittest.TestCase):
    def test_committed_config_is_fail_closed(self):
        config = load_config(CONFIG_PATH)
        self.assertFalse(config["write_enabled"])
        self.assertFalse(config["authorization"]["model_or_api_calls_authorized"])
        self.assertFalse(config["authorization"]["gpu_inference_authorized"])
        self.assertFalse(config["authorization"]["paid_canary_authorized"])
        with self.assertRaisesRegex(ValueError, "enabled runtime copy"):
            validate_config(config, write=True)
        runtime = copy.deepcopy(config)
        runtime["write_enabled"] = True
        validate_config(runtime, write=True)

    def test_config_rejects_paid_or_network_authority(self):
        config = load_config(CONFIG_PATH)
        changed = copy.deepcopy(config)
        changed["authorization"]["model_or_api_calls_authorized"] = True
        with self.assertRaisesRegex(ValueError, "authorization boundary"):
            validate_config(changed)

    def test_wilson_lower_bound_is_conservative(self):
        controlled = wilson_lower_bound(7, 8, 1.281551565545)
        student = wilson_lower_bound(2, 8, 1.281551565545)
        self.assertLess(controlled, 7 / 8)
        self.assertLess(student, 2 / 8)
        self.assertGreater(controlled, student)

    def test_real_log_plan_is_aggregate_only(self):
        config = load_config(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as directory:
            run_dir = make_private_fixture(Path(directory), config)
            runtime = copy.deepcopy(config)
            runtime["source_run_dir"] = str(run_dir)
            observations = collect_observations(runtime, repo_root=Path(directory))
            report = build_plan(runtime, observations)
            markdown = render_markdown(report)

        serialized = json.dumps(report)
        self.assertEqual(report["zero_call_preflight"]["status"], "passed")
        self.assertEqual(report["zero_call_preflight"]["network_or_model_calls_made"], 0)
        self.assertEqual(report["recommendation"]["new_questions"], 2500)
        self.assertEqual(report["recommendation"]["target_total_negative_trajectories"], 500)
        self.assertEqual(len(report["scenarios"]), 4)
        ratio = report["recommendation"]["projected_prm_positive_to_negative_ratio"]
        self.assertGreaterEqual(ratio, 5.0)
        self.assertLess(ratio, 5.01)
        self.assertEqual(
            report["recommendation"]["candidate_validation_quota_by_origin"],
            {"controlled_single_error": 591, "local_student": 901},
        )
        self.assertEqual(
            report["recommendation"]["prm_selected_new_positive_prefix_quota"],
            7605,
        )
        self.assertEqual(
            report["recommendation"]["prm_negative_trajectory_split"],
            {"test": 50, "train": 400, "validation": 50},
        )
        self.assertEqual(
            report["recommendation"]["sft_expected_trajectory_split"],
            {"train": 6354, "validation": 334},
        )
        self.assertGreater(
            report["recommendation"]["candidate_validation_quota_by_origin"]["local_student"],
            0,
        )
        self.assertNotIn("SECRET", serialized)
        self.assertNotIn("private-", serialized)
        self.assertNotIn("SECRET", markdown)
        self.assertIn("planning only", markdown)

    def test_missing_real_logs_fail_closed(self):
        config = load_config(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as directory:
            runtime = copy.deepcopy(config)
            runtime["source_run_dir"] = str(Path(directory) / "missing")
            with self.assertRaisesRegex(FileNotFoundError, "required private"):
                collect_observations(runtime, repo_root=Path(directory))


if __name__ == "__main__":
    unittest.main()
