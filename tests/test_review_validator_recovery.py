import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from data_pipeline.review_validator_recovery import build_review


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(value) + "\n" for value in values), encoding="utf-8"
    )


class ValidatorRecoveryReviewTests(unittest.TestCase):
    def test_review_shows_semantics_but_omits_identifiers_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "pilot"
            output_dir = run_dir / "recovery"
            output_dir.mkdir(parents=True)
            config = {
                "source_run_dir": str(run_dir),
                "output_subdir": "recovery",
                "canary": {"total": 1},
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            private_id = "PRIVATE-RECORD-ID"
            key = [private_id, 2]
            write_json(output_dir / "private_manifest.json", {"selected_keys": [key]})
            write_json(output_dir / "canary_metadata.json", {"status": "complete"})
            teacher = {
                "record_id": private_id,
                "candidate_index": 2,
                "record": {
                    "benchmark": "medqa",
                    "question": "VISIBLE QUESTION",
                    "choices": {"A": "VISIBLE OPTION A", "B": "VISIBLE OPTION B"},
                    "answer": "A",
                },
                "rule_check": {
                    "steps": ["VISIBLE STEP"], "predicted_answer": "A"
                },
            }
            screener = {
                "record_id": private_id,
                "candidate_index": 2,
                "result": {
                    "verdict": "review", "suspected_first_error_step": 0,
                    "error_codes": ["medical_fact_error"],
                    "concise_reason": "VISIBLE SCREENER REASON",
                },
            }
            result = {
                "trajectory_label": 1,
                "first_error_step": None,
                "answer_consistent": True,
                "problem_status": "ok",
                "steps": [{
                    "index": 0, "local_verdict": "correct", "prefix_label": 1,
                    "error_codes": [], "concise_reason": "VISIBLE VALIDATOR REASON",
                }],
            }
            recovery = {
                "record_id": private_id,
                "candidate_index": 2,
                "status": "complete",
                "result": result,
                "request_id": "PRIVATE-REQUEST-ID",
                "response_content_sha256": "PRIVATE-HASH",
                "usage": {"cost_cny": 9.9, "routed_provider": "PRIVATE-PROVIDER"},
            }
            write_jsonl(run_dir / "teacher_events.jsonl", [teacher])
            write_jsonl(run_dir / "screener_events.jsonl", [screener])
            write_jsonl(output_dir / "canary_events.jsonl", [recovery])

            with patch(
                "data_pipeline.review_validator_recovery.validate_recovery_config"
            ):
                rendered = build_review(config_path)

            for visible in (
                "VISIBLE QUESTION", "VISIBLE OPTION A", "VISIBLE STEP",
                "VISIBLE SCREENER REASON", "VISIBLE VALIDATOR REASON",
            ):
                self.assertIn(visible, rendered)
            for private in (
                private_id, "PRIVATE-REQUEST-ID", "PRIVATE-HASH",
                "PRIVATE-PROVIDER", "9.9",
            ):
                self.assertNotIn(private, rendered)
            self.assertIn("CASE 1/1", rendered)
            self.assertIn("HUMAN DECISION FOR CASE 1", rendered)

    def test_case_filter_is_one_based_and_validated(self):
        with self.assertRaisesRegex(ValueError, "between 1 and 2"):
            # Fail before any event lookup by using an invalid explicit case.
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                run_dir = root / "pilot"
                output_dir = run_dir / "recovery"
                output_dir.mkdir(parents=True)
                config_path = root / "config.yaml"
                config_path.write_text(
                    yaml.safe_dump({
                        "source_run_dir": str(run_dir), "output_subdir": "recovery",
                        "canary": {"total": 2},
                    }),
                    encoding="utf-8",
                )
                write_json(output_dir / "private_manifest.json", {
                    "selected_keys": [["a", 0], ["b", 0]]
                })
                write_json(output_dir / "canary_metadata.json", {"status": "complete"})
                with patch(
                    "data_pipeline.review_validator_recovery.validate_recovery_config"
                ):
                    build_review(config_path, [0])


if __name__ == "__main__":
    unittest.main()
