import json
from pathlib import Path
import tempfile
import unittest

from data_pipeline.audit_cot_pilot import (
    _deep_coverage,
    _strict_binary_label_counts,
    _tokens,
    _validator_cost_breakdown,
    _validator_diagnostics,
    audit_run,
    render_markdown,
)


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


class PilotAuditTests(unittest.TestCase):
    def test_duplicate_tokens_support_cjk_text(self):
        self.assertIn("医学", _tokens("医学推理"))
        self.assertIn("推理", _tokens("医学推理"))

    def test_strict_binary_labels_do_not_merge_boolean_and_integer(self):
        self.assertEqual(
            _strict_binary_label_counts([0, 1, True, False, None, "1"]),
            {
                "bool:false": 1,
                "bool:true": 1,
                "int:0": 1,
                "int:1": 1,
                "invalid_type:str": 1,
                "none": 1,
            },
        )

    def test_validator_diagnostics_preserve_only_aggregate_error_sequences(self):
        screeners = [
            {"record_id": "private-a", "candidate_index": 0, "result": {"verdict": "pass"}},
            {"record_id": "private-b", "candidate_index": 0, "result": {"verdict": "review"}},
        ]
        validators = [
            {
                "record_id": "private-a", "candidate_index": 0,
                "status": "complete", "attempts": 2,
                "errors": ["ValueError: validator JSON keys differ"],
                "result": {"trajectory_label": 1},
                "usage": {"cost_cny": 0.12, "failed_attempt_reserve_cny": 0.04},
            },
            {
                "record_id": "private-b", "candidate_index": 0,
                "status": "api_or_parse_error", "attempts": 2,
                "errors": [
                    "ValueError: invalid trajectory_label",
                    "IncompleteRead: connection closed",
                ],
                "result": None,
                "usage": {"cost_cny": 0.09, "failed_attempt_reserve_cny": 0.09},
            },
        ]

        diagnostics = _validator_diagnostics(screeners, validators)
        costs = _validator_cost_breakdown(validators)
        serialized = json.dumps({"diagnostics": diagnostics, "costs": costs})

        self.assertEqual(
            diagnostics["final_failure_error_sequences"],
            {"response_contract -> IncompleteRead": 1},
        )
        self.assertEqual(
            diagnostics["retry_outcomes"],
            {"attempts=2 -> api_or_parse_error": 1, "attempts=2 -> complete": 1},
        )
        self.assertEqual(
            diagnostics["outcome_by_screener_verdict"],
            {"pass": {"label_1": 1}, "review": {"api_or_parse_error": 1}},
        )
        self.assertEqual(costs["completed_response_charged_cny"], 0.08)
        self.assertEqual(costs["failed_attempt_charged_cny"], 0.13)
        self.assertNotIn("private-a", serialized)
        self.assertNotIn("private-b", serialized)

    def test_deep_coverage_stratifies_benchmarks_and_question_outcomes(self):
        def teacher(key: str, benchmark: str, passed: bool) -> dict:
            return {
                "record_id": key, "candidate_index": 0,
                "record": {"benchmark": benchmark, "content_sha256": key},
                "rule_check": {"passed": passed},
            }

        teachers = [
            teacher("hash-a", "medqa", False),
            teacher("hash-b", "medqa", True),
            teacher("hash-c", "medmcqa", True),
            teacher("hash-d", "medmcqa", True),
        ]
        screeners = [
            {"record_id": key, "candidate_index": 0}
            for key in ("hash-b", "hash-c", "hash-d")
        ]
        validators = [
            {"record_id": "hash-b", "candidate_index": 0, "status": "api_or_parse_error"},
            {"record_id": "hash-c", "candidate_index": 0, "status": "complete"},
            {"record_id": "hash-d", "candidate_index": 0, "status": "complete"},
        ]
        canonical = [
            {"source": {"dataset": "medmcqa", "content_sha256": "hash-c"}},
            {"source": {"dataset": "medmcqa", "content_sha256": "hash-d"}},
        ]
        sft = [{"source": {"dataset": "medmcqa", "content_sha256": "hash-d"}}]

        by_benchmark, outcomes = _deep_coverage(
            teachers, screeners, validators, canonical, sft
        )

        self.assertEqual(
            outcomes,
            {
                "canonical_without_sft": 1,
                "has_sft": 1,
                "no_rule_pass": 1,
                "no_validator_complete": 1,
            },
        )
        self.assertEqual(by_benchmark["medqa"]["questions_with_sft"], 0)
        self.assertEqual(by_benchmark["medmcqa"]["questions_with_sft"], 1)

    def test_aggregates_without_leaking_private_text(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            metadata = {
                "status": "complete",
                "questions": 1,
                "teacher_events": 2,
                "screener_events": 1,
                "validator_events": 1,
                "canonical_trajectories": 1,
                "sft_records": 1,
                "prm_records": 2,
                "spent_cny_equivalent": 0.3,
                "budget_stop_limit_cny": 9,
                "config_sha256": "c" * 64,
                "questions_sha256": "q" * 64,
                "preflight": {"screener_runtime": {"git_commit": "g" * 40}},
            }
            (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            teachers = [
                {
                    "record_id": "private-id",
                    "candidate_index": 0,
                    "status": "complete",
                    "attempts": 1,
                    "raw_response": "Secret question and answer",
                    "record": {"benchmark": "medqa", "content_sha256": "h" * 64},
                    "usage": {"cost_cny": 0.1},
                    "rule_check": {"passed": True, "steps": ["alpha", "beta"], "failure_codes": []},
                },
                {
                    "record_id": "private-id",
                    "candidate_index": 1,
                    "status": "complete",
                    "attempts": 1,
                    "raw_response": "More private text",
                    "record": {"benchmark": "medqa", "content_sha256": "h" * 64},
                    "usage": {"cost_cny": 0.1},
                    "rule_check": {"passed": False, "steps": [], "failure_codes": ["gold_answer_mismatch"]},
                },
            ]
            screeners = [{
                "record_id": "private-id", "candidate_index": 0,
                "status": "complete", "attempts": 1, "usage": {"cost_cny": 0},
                "result": {"verdict": "pass", "error_codes": []},
            }]
            result = {
                "trajectory_label": 1, "first_error_step": None,
                "problem_status": "ok",
                "steps": [
                    {"prefix_label": True, "error_codes": []},
                    {"prefix_label": 1, "error_codes": []},
                ],
            }
            validators = [{
                "record_id": "private-id", "candidate_index": 0,
                "status": "complete", "attempts": 1, "errors": [], "result": result,
                "usage": {
                    "cost_cny": 0.1, "cost_source": "provider_reported_plus_failed_attempt_reserve",
                    "failed_attempt_reserve_cny": 0.02,
                    "routed_provider": "Together",
                },
            }]
            canonical = [{
                "trajectory_id": "sha256:" + "a" * 64,
                "source": {"dataset": "medqa", "source_id": "private-id", "content_sha256": "h" * 64},
            }]
            sft = list(canonical)
            prm = [
                {"trajectory_id": canonical[0]["trajectory_id"], "label": True, "error_codes": []},
                {"trajectory_id": canonical[0]["trajectory_id"], "label": 1, "error_codes": []},
            ]
            for name, records in (
                ("teacher_events.jsonl", teachers),
                ("screener_events.jsonl", screeners),
                ("validator_events.jsonl", validators),
                ("canonical_trajectories.jsonl", canonical),
                ("sft_verified.jsonl", sft),
                ("process_train.jsonl", prm),
            ):
                write_jsonl(run_dir / name, records)

            report = audit_run(run_dir)
            serialized = json.dumps(report)
            markdown = render_markdown(report)

            self.assertTrue(report["invariants_passed"])
            self.assertEqual(report["coverage"]["questions_with_sft"], 1)
            self.assertEqual(report["teacher"]["rule_failure_codes"], {"gold_answer_mismatch": 1})
            self.assertEqual(report["validator"]["routed_provider"], {"Together": 1})
            self.assertEqual(
                report["prm"]["strict_label_counts"],
                {"bool:true": 1, "int:1": 1},
            )
            self.assertEqual(report["prm"]["records_with_non_integer_binary_label"], 1)
            self.assertFalse(report["prm"]["strict_label_quality_passed"])
            self.assertEqual(report["coverage"]["by_benchmark"]["medqa"]["questions_with_sft"], 1)
            self.assertNotIn("Secret question", serialized)
            self.assertNotIn("private-id", serialized)
            self.assertNotIn("private-id", markdown)


if __name__ == "__main__":
    unittest.main()
