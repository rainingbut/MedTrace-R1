import unittest

from data_pipeline.cot_api import (
    parse_json_object,
    validate_screener_result,
    validate_validator_result,
)


class JsonJudgeContractTests(unittest.TestCase):
    def test_parses_plain_and_fenced_json(self):
        self.assertEqual(parse_json_object('{"ok": true}'), {"ok": True})
        self.assertEqual(parse_json_object('```json\n{"ok": true}\n```'), {"ok": True})

    def test_validates_screener_index(self):
        value = {
            "verdict": "review",
            "suspected_first_error_step": 1,
            "error_codes": ["medical_fact_error"],
            "concise_reason": "check it",
        }
        self.assertEqual(validate_screener_result(value, 3), value)
        value["suspected_first_error_step"] = 3
        with self.assertRaisesRegex(ValueError, "first-error"):
            validate_screener_result(value, 3)

    def test_validator_enforces_prefix_semantics(self):
        value = {
            "trajectory_label": 0,
            "first_error_step": 1,
            "answer_consistent": True,
            "problem_status": "ok",
            "steps": [
                {"index": 0, "local_verdict": "correct", "prefix_label": 1, "error_codes": [], "concise_reason": "ok"},
                {"index": 1, "local_verdict": "incorrect", "prefix_label": 0, "error_codes": ["medical_fact_error"], "concise_reason": "bad"},
                {"index": 2, "local_verdict": "correct", "prefix_label": 0, "error_codes": [], "concise_reason": "locally ok"}
            ]
        }
        self.assertEqual(validate_validator_result(value, 3), value)
        value["steps"][2]["prefix_label"] = 1
        with self.assertRaisesRegex(ValueError, "remain zero"):
            validate_validator_result(value, 3)


if __name__ == "__main__":
    unittest.main()
