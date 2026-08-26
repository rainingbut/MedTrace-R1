import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from data_pipeline.audit_prm_negative_canary import audit_canary
from data_pipeline.build_strict_pilot_view import strict_records
from data_pipeline.cot_prompts import (
    build_controlled_mutation_prompt,
    build_prm_student_prompt,
)
from data_pipeline.prm_negative_canary_config import (
    validate_prm_negative_canary_config,
)
from data_pipeline.review_prm_negative_canary import build_review
from data_pipeline.run_prm_negative_canary import (
    _target_index,
    _validate_replacement,
    _validate_candidate_mix,
    build_candidate,
    select_existing_natural,
)


ROOT = Path(__file__).resolve().parents[1]


def validator_result(label: int, first: int | None = None) -> dict:
    verdicts = ["correct", "correct", "correct"]
    if first is not None:
        verdicts[first] = "incorrect"
    return {
        "trajectory_label": label,
        "first_error_step": first,
        "answer_consistent": True,
        "problem_status": "ok",
        "steps": [
            {
                "index": index,
                "local_verdict": verdict,
                "prefix_label": int(first is None or index < first),
                "error_codes": ["medical_fact_error"]
                if verdict == "incorrect" else [],
                "concise_reason": "private",
            }
            for index, verdict in enumerate(verdicts)
        ],
    }


def canonical(identifier: str, result: dict, disposition: str) -> dict:
    return {
        "trajectory_id": f"sha256:{identifier * 64}"[:71],
        "source": {
            "dataset": "medqa",
            "source_id": f"private-{identifier}",
            "content_sha256": identifier * 64,
            "split": "train",
        },
        "problem": {
            "question": "private question",
            "choices": {"A": "a", "B": "b"},
            "gold_answer": "A",
        },
        "trajectory": {
            "steps": [
                {"index": index, "text": f"private step {index}"}
                for index in range(3)
            ],
            "predicted_answer": "A",
        },
        "verification": result,
        "disposition": disposition,
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


class PrmNegativeCanaryTests(unittest.TestCase):
    def test_committed_config_is_frozen_and_execution_needs_runtime_copy(self):
        config = yaml.safe_load(
            (ROOT / "configs/cot/prm_negative_canary_v1.yaml").read_text(
                encoding="utf-8"
            )
        )
        validate_prm_negative_canary_config(config)
        runtime = dict(config)
        runtime["execution_enabled"] = True
        validate_prm_negative_canary_config(runtime, execute=True)
        self.assertEqual(config["budget"]["api_hard_cap_cny_equivalent"], 20)

    def test_strict_view_excludes_boolean_prefix_contract(self):
        positive = validator_result(1)
        negative = validator_result(0, 1)
        invalid = validator_result(1)
        for step in invalid["steps"]:
            step["prefix_label"] = True
        records = [
            canonical("a", positive, "sft_accept"),
            canonical("b", negative, "prm_only"),
            canonical("c", invalid, "sft_accept"),
        ]

        strict, sft, prm, excluded = strict_records(records)

        self.assertEqual(len(strict), 2)
        self.assertEqual(len(sft), 1)
        self.assertEqual(len(prm), 6)
        self.assertEqual(excluded, {"validator_contract_invalid": 1})
        self.assertTrue(all(type(record["label"]) is int for record in prm))

    def test_natural_selection_prefers_distinct_questions(self):
        events = []
        for index, content in enumerate(("a", "a", "b", "c", "d")):
            events.append(
                {
                    "record_id": f"private-{index}",
                    "candidate_index": index,
                    "status": "complete",
                    "record": {
                        "benchmark": "medqa",
                        "source_id": f"source-{index}",
                        "content_sha256": content * 64,
                        "split": "train",
                        "choices": {"A": "a", "B": "b"},
                    },
                    "rule_check": {
                        "passed": False,
                        "steps": ["one", "two", "three"],
                        "predicted_answer": "B",
                        "failure_codes": ["gold_answer_mismatch"],
                    },
                }
            )
        selected = select_existing_natural(events, 20260826, {"medqa": 4})
        identities = {event["record"]["content_sha256"] for event in selected}
        self.assertEqual(len(selected), 4)
        self.assertEqual(len(identities), 4)

    def test_candidate_has_no_label_and_controlled_target_is_bounded(self):
        record = {
            "benchmark": "medqa",
            "source_id": "private",
            "content_sha256": "a" * 64,
            "split": "train",
        }
        candidate = build_candidate(
            record=record,
            origin="controlled_single_error",
            steps=["one", "changed", "three"],
            predicted_answer="A",
            generator_role="controlled_mutator",
            model_id="model",
            model_revision="revision",
            prompt_version="mutator_v1",
            run_git_commit="b" * 40,
            parent_trajectory_id="sha256:" + "c" * 64,
            intended_error_step=1,
        )
        self.assertNotIn("label", candidate)
        self.assertEqual(candidate["disposition"], "requires_independent_validation")
        self.assertEqual(_target_index("early", 5), 0)
        self.assertEqual(_target_index("middle", 5), 2)
        self.assertEqual(_target_index("late", 5), 4)

    def test_prompts_keep_student_gold_blind_and_mutation_structural(self):
        student = build_prm_student_prompt("private", {"A": "one", "B": "two"})
        self.assertNotIn("gold_answer", student)
        mutation = build_controlled_mutation_prompt(
            "private", {"A": "one", "B": "two"},
            ["alpha", "beta", "gamma"], "A", 1,
        )
        self.assertIn("replacement for step 1", mutation)
        self.assertIn("single key replacement_step", mutation)
        self.assertEqual(
            _validate_replacement({"replacement_step": "  wrong claim  "}),
            {"replacement_step": "wrong claim"},
        )

    def test_candidate_mix_requires_exact_origin_and_benchmark_quotas(self):
        config = yaml.safe_load(
            (ROOT / "configs/cot/prm_negative_canary_v1.yaml").read_text(
                encoding="utf-8"
            )
        )
        candidates = []
        for origin in config["selection"]["by_origin"]:
            for benchmark in ("medqa", "medmcqa"):
                for index in range(4):
                    candidates.append(
                        {
                            "candidate_id": f"{origin}:{benchmark}:{index}",
                            "origin": origin,
                            "source": {"dataset": benchmark},
                        }
                    )
        _validate_candidate_mix(candidates, config)
        candidates.pop()
        with self.assertRaisesRegex(RuntimeError, "total"):
            _validate_candidate_mix(candidates, config)

    def test_aggregate_canary_audit_passes_machine_gates_without_private_text(self):
        config_path = ROOT / "configs/cot/prm_negative_canary_v1.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = (
                root / config["source_run_dir"] / config["output_subdir"]
            )
            output.mkdir(parents=True)
            candidates = []
            events = []
            origins = list(config["selection"]["by_origin"])
            cost = 0.0
            for origin_index, origin in enumerate(origins):
                for benchmark in ("medqa", "medmcqa"):
                    for index in range(4):
                        candidate_id = f"candidate-{origin_index}-{benchmark}-{index}"
                        controlled = origin == "controlled_single_error"
                        candidate = {
                            "candidate_id": candidate_id,
                            "origin": origin,
                            "source": {"dataset": benchmark},
                            "trajectory": {"steps": ["a", "b", "c"]},
                            "intended_error_step": 1 if controlled else None,
                        }
                        negative = origin != "local_student"
                        result = validator_result(0, 1) if negative else validator_result(1)
                        event_cost = 0.01
                        cost += event_cost
                        candidates.append(candidate)
                        events.append(
                            {
                                "candidate_id": candidate_id,
                                "status": "complete",
                                "result": result,
                                "usage": {"cost_cny": event_cost},
                                "private_text": "must not leak",
                            }
                        )
            write_jsonl(output / "candidates.jsonl", candidates)
            write_jsonl(output / "validator_events.jsonl", events)
            (output / "metadata.json").write_text(
                json.dumps({"status": "complete", "spent_cny_equivalent": cost}),
                encoding="utf-8",
            )
            (output / "private_manifest.json").write_text(
                json.dumps({"source_artifact_sha256": {"source": "hash"}}),
                encoding="utf-8",
            )
            with (
                patch("data_pipeline.audit_prm_negative_canary.REPO_ROOT", root),
                patch(
                    "data_pipeline.audit_prm_negative_canary._source_hashes",
                    return_value={"source": "hash"},
                ),
            ):
                report = audit_canary(config_path)
            serialized = json.dumps(report)
            self.assertTrue(report["integrity"]["passed"])
            self.assertTrue(report["machine_quality"]["passed"])
            self.assertEqual(report["counts"]["strict_process_negatives"], 16)
            self.assertNotIn("must not leak", serialized)
            self.assertFalse(report["human_review"]["auto_approved"])

    def test_private_review_blinds_origin_and_validator_until_key(self):
        config_path = ROOT / "configs/cot/prm_negative_canary_v1.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / config["source_run_dir"]
            output = run_dir / config["output_subdir"]
            output.mkdir(parents=True)
            source_hash = "a" * 64
            record = {
                "content_sha256": source_hash,
                "question": "private question",
                "choices": {"A": "private a", "B": "private b"},
                "answer": "A",
            }
            write_jsonl(run_dir / "teacher_events.jsonl", [{"record": record}])
            candidates = []
            events = []
            for index in range(24):
                candidate_id = f"candidate-{index}"
                candidates.append(
                    {
                        "candidate_id": candidate_id,
                        "origin": "controlled_single_error",
                        "source": {"dataset": "medqa", "content_sha256": source_hash},
                        "trajectory": {
                            "steps": ["first", "private changed", "third"],
                            "predicted_answer": "A",
                        },
                        "intended_error_step": 1,
                    }
                )
                events.append(
                    {
                        "candidate_id": candidate_id,
                        "status": "complete",
                        "result": validator_result(0, 1),
                    }
                )
            write_jsonl(output / "candidates.jsonl", candidates)
            write_jsonl(output / "validator_events.jsonl", events)
            (output / "metadata.json").write_text(
                json.dumps({"status": "complete"}), encoding="utf-8"
            )
            with patch("data_pipeline.review_prm_negative_canary.REPO_ROOT", root):
                blind, key = build_review(config_path)
            self.assertIn("private question", blind)
            self.assertNotIn("controlled_single_error", blind)
            self.assertNotIn("Validator disposition", blind)
            self.assertIn("controlled_single_error", key)
            self.assertIn("strict_process_negative", key)


if __name__ == "__main__":
    unittest.main()
