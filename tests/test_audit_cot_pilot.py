import json
from pathlib import Path
import tempfile
import unittest

from data_pipeline.audit_cot_pilot import _tokens, audit_run, render_markdown


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


class PilotAuditTests(unittest.TestCase):
    def test_duplicate_tokens_support_cjk_text(self):
        self.assertIn("医学", _tokens("医学推理"))
        self.assertIn("推理", _tokens("医学推理"))

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
                    "record": {"content_sha256": "h" * 64},
                    "usage": {"cost_cny": 0.1},
                    "rule_check": {"passed": True, "steps": ["alpha", "beta"], "failure_codes": []},
                },
                {
                    "record_id": "private-id",
                    "candidate_index": 1,
                    "status": "complete",
                    "attempts": 1,
                    "raw_response": "More private text",
                    "record": {"content_sha256": "h" * 64},
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
                    {"prefix_label": 1, "error_codes": []},
                    {"prefix_label": 1, "error_codes": []},
                ],
            }
            validators = [{
                "record_id": "private-id", "candidate_index": 0,
                "status": "complete", "attempts": 1, "errors": [], "result": result,
                "usage": {
                    "cost_cny": 0.1, "cost_source": "provider_reported",
                    "routed_provider": "Together",
                },
            }]
            canonical = [{
                "trajectory_id": "sha256:" + "a" * 64,
                "source": {"source_id": "private-id", "content_sha256": "h" * 64},
            }]
            sft = list(canonical)
            prm = [
                {"trajectory_id": canonical[0]["trajectory_id"], "label": 1, "error_codes": []},
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
            self.assertNotIn("Secret question", serialized)
            self.assertNotIn("private-id", serialized)
            self.assertNotIn("private-id", markdown)


if __name__ == "__main__":
    unittest.main()
