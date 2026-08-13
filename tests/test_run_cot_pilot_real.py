import unittest
from unittest.mock import patch

from data_pipeline.cot_api import ChatResult
from data_pipeline.cot_budget import BudgetLedger, request_cost_cny
from data_pipeline.run_cot_pilot_real import _call_with_budget


class RealPilotBudgetTests(unittest.TestCase):
    def test_success_attempt_also_records_diagnostics(self):
        pricing = {
            "api_hard_cap_cny_equivalent": 10,
            "stop_before_limit_fraction": 0.9,
            "usd_to_cny": 7.2,
            "teacher_cny_per_million_input_tokens": 2.5,
            "teacher_cny_per_million_output_tokens": 10,
            "validator_usd_per_million_input_tokens": 2.10,
            "validator_usd_per_million_output_tokens": 4.40,
        }
        response = ChatResult(
            content='{"ok": true}', reasoning_content=None, request_id="ok-id",
            finish_reason="stop", input_tokens=10, output_tokens=5,
            billed_cost_usd=0.001, routed_provider="Together", reasoning_tokens=2,
        )
        config = {
            "base_url": "https://example.invalid", "model_id": "validator",
            "temperature": 0, "max_output_tokens": 128, "max_retries": 0,
            "timeout_seconds": 1, "reasoning_effort": "high",
            "require_zero_data_retention": True, "allow_provider_fallbacks": True,
        }
        with patch(
            "data_pipeline.run_cot_pilot_real.post_chat_completion",
            return_value=response,
        ) as post:
            call = _call_with_budget(
                role="validator", config=config, system_prompt="system",
                user_prompt="user", ledger=BudgetLedger(10, 0.9), pricing=pricing,
                api_key="secret", response_format_json=True,
                response_format={"type": "json_schema", "json_schema": {"strict": True}},
            )
        diagnostic = call["attempt_diagnostics"][0]
        self.assertEqual(call["status"], "complete")
        self.assertEqual(diagnostic["status"], "complete")
        self.assertEqual(diagnostic["request_id"], "ok-id")
        self.assertIsNone(diagnostic["error_detail"])
        self.assertEqual(
            post.call_args.kwargs["response_format"],
            {"type": "json_schema", "json_schema": {"strict": True}},
        )

    def test_parse_failure_records_actual_provider_cost_once(self):
        pricing = {
            "api_hard_cap_cny_equivalent": 10,
            "stop_before_limit_fraction": 0.9,
            "usd_to_cny": 7.2,
            "teacher_cny_per_million_input_tokens": 2.5,
            "teacher_cny_per_million_output_tokens": 10,
            "validator_usd_per_million_input_tokens": 2.10,
            "validator_usd_per_million_output_tokens": 4.40,
        }
        ledger = BudgetLedger(10, 0.9)
        response = ChatResult(
            content="not-json",
            reasoning_content=None,
            request_id="request-1",
            finish_reason="stop",
            input_tokens=100,
            output_tokens=50,
            billed_cost_usd=0.001,
            routed_provider="Together",
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

        expected = 0.001 * pricing["usd_to_cny"]
        self.assertEqual(result["status"], "api_or_parse_error")
        self.assertAlmostEqual(ledger.spent_cny, expected)
        self.assertAlmostEqual(result["usage"]["cost_cny"], expected)
        self.assertEqual(result["attempt_diagnostics"][0]["status"], "failed")
        extras = post.call_args.kwargs["extra_body"]
        self.assertEqual(extras["reasoning"], {"effort": "high"})
        self.assertEqual(
            extras["provider"],
            {
                "zdr": True,
                "data_collection": "deny",
                "allow_fallbacks": True,
                "require_parameters": True,
                "sort": "price",
                "max_price": {"prompt": 2.10, "completion": 4.40},
            },
        )

    def test_missing_openrouter_cost_fails_closed_with_reserved_cost(self):
        pricing = {
            "api_hard_cap_cny_equivalent": 10,
            "stop_before_limit_fraction": 0.9,
            "usd_to_cny": 7.2,
            "teacher_cny_per_million_input_tokens": 2.5,
            "teacher_cny_per_million_output_tokens": 10,
            "validator_usd_per_million_input_tokens": 2.10,
            "validator_usd_per_million_output_tokens": 4.40,
        }
        ledger = BudgetLedger(10, 0.9)
        response = ChatResult(
            content='{"ok": true}',
            reasoning_content=None,
            request_id="request-2",
            finish_reason="stop",
            input_tokens=100,
            output_tokens=50,
            billed_cost_usd=None,
            routed_provider="unknown",
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
        ):
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

        self.assertEqual(result["status"], "api_or_parse_error")
        self.assertIn("missing usage.cost", result["errors"][0])
        self.assertGreater(ledger.spent_cny, 0)
        self.assertEqual(result["usage"]["cost_source"], "failed_attempt_reserve")

    def test_empty_validator_content_preserves_attempt_diagnostics(self):
        pricing = {
            "api_hard_cap_cny_equivalent": 10,
            "stop_before_limit_fraction": 0.9,
            "usd_to_cny": 7.2,
            "teacher_cny_per_million_input_tokens": 2.5,
            "teacher_cny_per_million_output_tokens": 10,
            "validator_usd_per_million_input_tokens": 2.10,
            "validator_usd_per_million_output_tokens": 4.40,
        }
        result = ChatResult(
            content=None,
            reasoning_content="private reasoning",
            request_id="request-empty",
            finish_reason="length",
            input_tokens=120,
            output_tokens=8192,
            billed_cost_usd=0.02,
            routed_provider="Together",
            reasoning_tokens=8180,
        )
        config = {
            "base_url": "https://example.invalid", "model_id": "validator",
            "temperature": 0, "max_output_tokens": 8192, "max_retries": 0,
            "timeout_seconds": 1, "reasoning_effort": "high",
            "require_zero_data_retention": True, "allow_provider_fallbacks": True,
        }
        with patch(
            "data_pipeline.run_cot_pilot_real.post_chat_completion",
            return_value=result,
        ):
            call = _call_with_budget(
                role="validator", config=config, system_prompt="system",
                user_prompt="user", ledger=BudgetLedger(100, 0.9),
                pricing=pricing, api_key="secret", response_format_json=True,
                response_format={"type": "json_schema"},
            )
        self.assertEqual(call["status"], "api_or_parse_error")
        diagnostic = call["attempt_diagnostics"][0]
        self.assertEqual(diagnostic["error_detail"], "response_no_text_content")
        self.assertEqual(diagnostic["finish_reason"], "length")
        self.assertEqual(diagnostic["routed_provider"], "Together")
        self.assertEqual(diagnostic["reasoning_tokens"], 8180)
        self.assertFalse(diagnostic["content_present"])
        self.assertEqual(diagnostic["status"], "failed")
        self.assertNotIn("private reasoning", str(diagnostic))


if __name__ == "__main__":
    unittest.main()
