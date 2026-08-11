import unittest

from data_pipeline.cot_budget import BudgetExceeded, BudgetLedger, request_cost_cny
from data_pipeline.cot_rules import check_teacher_response


class CotRuleTests(unittest.TestCase):
    def test_accepts_strict_xml_trajectory(self):
        raw = (
            "<step>First claim.</step><step>Second claim.</step>"
            "<step>Third claim.</step><answer>B</answer>"
        )
        result = check_teacher_response(raw, "ABCD", "B")
        self.assertTrue(result.passed)
        self.assertEqual(result.predicted_answer, "B")

    def test_rejects_duplicate_step(self):
        raw = (
            "<step>Same.</step><step>Same.</step>"
            "<step>Third.</step><answer>B</answer>"
        )
        result = check_teacher_response(raw, "ABCD", "B")
        self.assertIn("duplicate_step", result.failure_codes)

    def test_rejects_gold_mismatch_and_missing_answer(self):
        wrong = (
            "<step>One.</step><step>Two.</step><step>Three.</step><answer>A</answer>"
        )
        self.assertIn(
            "gold_answer_mismatch",
            check_teacher_response(wrong, "ABCD", "B").failure_codes,
        )
        missing = "<step>One.</step><step>Two.</step><step>Three.</step>"
        self.assertIn(
            "missing_answer_tag",
            check_teacher_response(missing, "ABCD", "B").failure_codes,
        )


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.pricing = {
            "teacher_cny_per_million_input_tokens": 2.5,
            "teacher_cny_per_million_output_tokens": 10,
            "validator_usd_per_million_input_tokens": 0.435,
            "validator_usd_per_million_output_tokens": 0.87,
            "usd_to_cny": 7.2,
        }

    def test_teacher_and_validator_costs(self):
        self.assertAlmostEqual(
            request_cost_cny("teacher", 1_000_000, 1_000_000, self.pricing),
            12.5,
        )
        self.assertAlmostEqual(
            request_cost_cny("validator", 1_000_000, 1_000_000, self.pricing),
            (0.435 + 0.87) * 7.2,
        )

    def test_budget_stops_before_hard_cap(self):
        ledger = BudgetLedger(hard_cap_cny=10, stop_fraction=0.9, spent_cny=8.5)
        ledger.assert_can_spend(0.5)
        with self.assertRaises(BudgetExceeded):
            ledger.assert_can_spend(0.500001)


if __name__ == "__main__":
    unittest.main()
