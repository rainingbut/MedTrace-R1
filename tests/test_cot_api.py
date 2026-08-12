import unittest
from unittest.mock import MagicMock, patch

from data_pipeline.cot_api import (
    post_chat_completion,
    parse_json_object,
    validate_screener_result,
    validate_validator_result,
)


class JsonJudgeContractTests(unittest.TestCase):
    def test_reads_openrouter_usage_cost_and_provider(self):
        response = MagicMock()
        response.__enter__.return_value = response
        response.__exit__.return_value = False
        response.headers = {}
        response.read.return_value = (
            b'{"id":"r1","provider":"Together","choices":[{"message":'
            b'{"content":"{}"},"finish_reason":"stop"}],"usage":'
            b'{"prompt_tokens":10,"completion_tokens":3,"cost":0.0012}}'
        )
        with patch("data_pipeline.cot_api.urllib_request.urlopen", return_value=response):
            result = post_chat_completion(
                base_url="https://openrouter.invalid/api/v1",
                api_key="secret",
                model="deepseek/deepseek-v4-pro",
                system_prompt="system",
                user_prompt="user",
                temperature=0,
                max_tokens=10,
                timeout_seconds=1,
            )
        self.assertEqual(result.billed_cost_usd, 0.0012)
        self.assertEqual(result.routed_provider, "Together")

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
