import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from data_pipeline.audit_prm_negative_opportunities import (
    _quality_status,
    audit_opportunities,
    render_markdown,
)
from data_pipeline.prm_negative_policy import (
    structurally_usable_wrong_answer,
    validate_prm_negative_policy,
    verification_disposition,
)


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


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
                "concise_reason": "private reason",
            }
            for index, verdict in enumerate(verdicts)
        ],
    }


class PrmNegativePolicyTests(unittest.TestCase):
    def test_committed_policy_and_candidate_contract_are_safe(self):
        root = Path(__file__).resolve().parents[1]
        config = yaml.safe_load(
            (root / "configs/cot/prm_negative_enrichment_v1.yaml").read_text(
                encoding="utf-8"
            )
        )
        validate_prm_negative_policy(config)
        schema = json.loads(
            (root / "schemas/prm_negative_candidate_v1.schema.json").read_text(
                encoding="utf-8"
            )
        )
        source = schema["properties"]["source"]["properties"]
        self.assertEqual(source["split"]["const"], "train")
        self.assertNotIn("label", schema["properties"])
        self.assertEqual(
            schema["properties"]["disposition"]["const"],
            "requires_independent_validation",
        )

    def test_wrong_answer_is_only_an_opportunity(self):
        event = {
            "status": "complete",
            "record": {"choices": {"A": "private", "B": "private"}},
            "rule_check": {
                "passed": False,
                "steps": ["one", "two", "three"],
                "predicted_answer": "B",
                "failure_codes": ["gold_answer_mismatch"],
            },
        }
        self.assertTrue(structurally_usable_wrong_answer(event))
        event["rule_check"]["failure_codes"].append("duplicate_step")
        self.assertFalse(structurally_usable_wrong_answer(event))

    def test_verification_policy_is_conservative(self):
        self.assertEqual(
            verification_disposition(validator_result(0, 1)),
            "strict_process_negative",
        )
        uncertain = validator_result(0, 1)
        uncertain["steps"][1]["local_verdict"] = "uncertain"
        self.assertEqual(
            verification_disposition(uncertain), "human_review_uncertain"
        )
        answer_only = validator_result(0, None)
        self.assertEqual(
            verification_disposition(answer_only),
            "answer_only_or_inconsistent_negative",
        )


class PrmNegativeOpportunityAuditTests(unittest.TestCase):
    def test_label_quality_failure_is_not_source_integrity_failure(self):
        canonical = {
            "strict_contract_valid": 108,
            "dispositions": {
                "invalid_contract": 1,
                "strict_positive": 107,
                "strict_process_negative": 1,
            },
        }
        sft = {
            "strict_contract_valid": 107,
            "dispositions": {"invalid_contract": 1, "strict_positive": 107},
        }
        prm = [{"label": 1}, {"label": 0}, {"label": True}]

        quality = _quality_status(canonical, sft, prm)

        self.assertFalse(quality["strict_training_artifacts_passed"])
        self.assertEqual(
            quality["failed_checks"],
            [
                "canonical_contracts_strict",
                "prm_labels_are_strict_binary_integers",
                "sft_contracts_strict",
            ],
        )
        self.assertTrue(quality["requires_normalization_or_revalidation"])

    def test_aggregate_audit_finds_candidates_without_leaking_private_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "results/cot/pilot_v1_real"
            run_dir.mkdir(parents=True)
            config = {
                "schema_version": "medtrace.cot.prm-negative-enrichment.v1",
                "execution_enabled": False,
                "source_run_dir": "results/cot/pilot_v1_real",
                "output_subdir": "prm_negative_enrichment_v1",
                "output_stem": "prm_negative_opportunity_audit",
                "original_artifacts_immutable": True,
                "source_identity": {
                    "questions": 2,
                    "teacher_events": 4,
                    "rule_passed": 3,
                    "canonical_trajectories": 2,
                    "prm_records": 6,
                },
                "natural_candidate_policy": {
                    "answer_mismatch_only": "requires_independent_validation",
                    "screener_reject": "requires_independent_validation",
                    "validator_unavailable": "recovery_only_not_negative_evidence",
                    "malformed_or_truncated": "exclude",
                },
                "label_policy": {
                    "label_semantics": "prefix_correctness",
                    "positive_requires": {
                        "trajectory_label": 1,
                        "problem_status": "ok",
                        "first_error_step": None,
                    },
                    "uncertain_first_error": "human_review",
                    "ambiguous_or_bad_gold": "human_review",
                    "wrong_answer_without_reasoning_error": "exclude_from_prm_negative",
                    "negative_requires": {
                        "trajectory_label": 0,
                        "problem_status": "ok",
                        "first_error_step": "integer",
                        "first_error_local_verdict": "incorrect",
                    },
                    "prefixes_before_first_error": 1,
                    "prefixes_from_first_error": 0,
                },
                "future_canary": {
                    "requires_separate_user_approval": False,
                    "approval_status": "approved_2026-08-26",
                    "target_total": 24,
                    "target_by_benchmark": {"medqa": 12, "medmcqa": 12},
                    "preferred_origin_targets": {
                        "existing_natural": 8,
                        "local_student": 8,
                        "controlled_single_error": 8,
                    },
                    "transport_and_contract_success_rate": 1.0,
                    "minimum_distinct_strict_negative_trajectories": 8,
                    "minimum_human_trajectory_label_accuracy": 0.90,
                    "minimum_human_exact_first_error_accuracy": 0.80,
                },
            }
            config_path = root / "policy.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            validate_prm_negative_policy(config)

            def teacher(
                private_id: str,
                index: int,
                benchmark: str,
                passed: bool,
                failures: list[str],
            ) -> dict:
                return {
                    "record_id": private_id,
                    "candidate_index": index,
                    "status": "complete",
                    "record": {
                        "benchmark": benchmark,
                        "choices": {"A": "secret-a", "B": "secret-b"},
                    },
                    "raw_response": "secret trajectory text",
                    "rule_check": {
                        "passed": passed,
                        "steps": ["secret one", "secret two", "secret three"],
                        "predicted_answer": "A" if passed else "B",
                        "failure_codes": failures,
                    },
                }

            teachers = [
                teacher("private-a", 0, "medqa", False, ["gold_answer_mismatch"]),
                teacher("private-a", 1, "medqa", True, []),
                teacher("private-b", 0, "medmcqa", True, []),
                teacher("private-b", 1, "medmcqa", True, []),
            ]
            screeners = [
                {
                    "record_id": "private-a", "candidate_index": 1,
                    "result": {"verdict": "reject"},
                },
                {
                    "record_id": "private-b", "candidate_index": 0,
                    "result": {"verdict": "pass"},
                },
                {
                    "record_id": "private-b", "candidate_index": 1,
                    "result": {"verdict": "review"},
                },
            ]
            positive = validator_result(1)
            negative = validator_result(0, 1)
            validators = [
                {
                    "record_id": "private-b", "candidate_index": 0,
                    "status": "complete", "result": positive,
                },
                {
                    "record_id": "private-b", "candidate_index": 1,
                    "status": "complete", "result": negative,
                },
            ]
            canonical = [
                {
                    "trajectory_id": "positive",
                    "trajectory": {"steps": [{"text": "secret"}] * 3},
                    "verification": positive,
                },
                {
                    "trajectory_id": "negative",
                    "trajectory": {"steps": [{"text": "secret"}] * 3},
                    "verification": negative,
                },
            ]
            sft = [canonical[0]]
            prm = [
                {"trajectory_id": "positive", "step_index": index, "label": 1}
                for index in range(3)
            ] + [
                {"trajectory_id": "negative", "step_index": index,
                 "label": 1 if index == 0 else 0}
                for index in range(3)
            ]
            metadata = {
                "status": "complete", "questions": 2,
                "teacher_events": 4, "screener_events": 3,
                "validator_events": 2, "canonical_trajectories": 2,
                "sft_records": 1, "prm_records": 6,
            }
            (run_dir / "metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            for name, records in (
                ("teacher_events.jsonl", teachers),
                ("screener_events.jsonl", screeners),
                ("validator_events.jsonl", validators),
                ("canonical_trajectories.jsonl", canonical),
                ("sft_verified.jsonl", sft),
                ("process_train.jsonl", prm),
            ):
                write_jsonl(run_dir / name, records)

            with patch(
                "data_pipeline.audit_prm_negative_opportunities.REPO_ROOT", root
            ):
                report = audit_opportunities(config_path)
            serialized = json.dumps(report)
            markdown = render_markdown(report)

            self.assertTrue(report["integrity"]["passed"])
            self.assertTrue(report["quality"]["strict_training_artifacts_passed"])
            self.assertEqual(report["natural_opportunities"]["total"], 2)
            self.assertEqual(
                report["natural_opportunities"]["answer_mismatch_only"], 1
            )
            self.assertEqual(report["natural_opportunities"]["screener_reject"], 1)
            self.assertEqual(
                report["canonical"]["dispositions"],
                {"strict_positive": 1, "strict_process_negative": 1},
            )
            self.assertEqual(report["prm"]["negative_prefix_records"], 2)
            self.assertEqual(
                report["prm"]["distinct_trajectories_with_negative_prefix"], 1
            )
            self.assertEqual(report["prm"]["non_integer_label_records"], 0)
            self.assertNotIn("private-a", serialized)
            self.assertNotIn("secret trajectory", serialized)
            self.assertNotIn("private-a", markdown)


if __name__ == "__main__":
    unittest.main()
