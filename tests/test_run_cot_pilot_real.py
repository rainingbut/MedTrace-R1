import unittest
from unittest.mock import patch

from data_pipeline.cot_api import ChatResult
from data_pipeline.cot_budget import BudgetLedger, request_cost_cny
from data_pipeline.run_cot_pilot_real import _call_with_budget


class RealPilotBudgetTests(unittest.TestCase):
    def test_parse_failure_records_actual_provider_cost_once(self):
        pricing = {
            "api_hard_cap_cny_equivalent": 10,
            "stop_before_limit_fraction": 0.9,
            "usd_to_cny": 7.2,
            "teacher_cny_per_million_input_tokens": 2.5,
            "teacher_cny_per_million_output_tokens": 10,
            "validator_usd_per_million_input_tokens": 0.435,
            "validator_usd_per_million_output_tokens": 0.87,
        }
        ledger = BudgetLedger(10, 0.9)
        response = ChatResult(
            content="not-json",
            reasoning_content=None,
            request_id="request-1",
            finish_reason="stop",
            input_tokens=100,
            output_tokens=50,
        )
        config = {
            "base_url": "https://example.invalid",
            "model_id": "validator",
            "temperature": 0,
            "max_output_tokens": 512,
            "max_retries": 0,
            "timeout_seconds": 1,
            "reasoning_effort": "high",
            "require_zero_data_retention": True,
            "allow_provider_fallbacks": True,
        }

        with patch(
            "data_pipeline.run_cot_pilot_real.post_chat_completion",
            return_value=response,
        ) as post:
            result = _call_with_budget(
                role="validator",
                config=config,
                system_prompt="system",
                user_prompt="user",
                ledger=ledger,
                pricing=pricing,
                api_key="secret",
                response_format_json=True,
            )

        expected = request_cost_cny("validator", 100, 50, pricing)
        self.assertEqual(result["status"], "api_or_parse_error")
        self.assertAlmostEqual(ledger.spent_cny, expected)
        self.assertAlmostEqual(result["usage"]["cost_cny"], expected)
        extras = post.call_args.kwargs["extra_body"]
        self.assertEqual(extras["reasoning"], {"effort": "high"})
        self.assertEqual(
            extras["provider"],
            {
                "zdr": True,
                "data_collection": "deny",
                "allow_fallbacks": True,
                "require_parameters": True,
            },
        )


if __name__ == "__main__":
    unittest.main()
