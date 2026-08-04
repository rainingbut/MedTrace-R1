import unittest

from evaluation.answer_extractor import extract_answer


CHOICES = ("A", "B", "C", "D")


class AnswerExtractorTests(unittest.TestCase):
    def test_exact_final_line_is_strictly_valid(self):
        result = extract_answer("Reasoning here.\nFinal Answer: B", CHOICES)
        self.assertEqual(result.answer, "B")
        self.assertEqual(result.parse_status, "parsed_strict")
        self.assertTrue(result.format_valid)

    def test_explicit_english_answer_is_parsed_but_not_format_valid(self):
        result = extract_answer("Therefore, the answer is C.", CHOICES)
        self.assertEqual(result.answer, "C")
        self.assertEqual(result.parse_status, "parsed_explicit")
        self.assertFalse(result.format_valid)

    def test_explicit_chinese_answer_is_parsed(self):
        result = extract_answer("综合判断，最终答案为 D。", CHOICES)
        self.assertEqual(result.answer, "D")
        self.assertFalse(result.format_valid)

    def test_final_marker_takes_priority_over_reasoning_mention(self):
        result = extract_answer(
            "At first the answer is A, but that is wrong.\nFinal Answer: C", CHOICES
        )
        self.assertEqual(result.answer, "C")
        self.assertTrue(result.format_valid)

    def test_conflicting_final_markers_are_ambiguous(self):
        result = extract_answer("Final Answer: A\nFinal Answer: B", CHOICES)
        self.assertIsNone(result.answer)
        self.assertEqual(result.parse_status, "ambiguous")
        self.assertFalse(result.format_valid)

    def test_repeated_marker_is_not_format_valid(self):
        result = extract_answer("Final Answer: A\nFinal Answer: A", CHOICES)
        self.assertEqual(result.answer, "A")
        self.assertEqual(result.parse_status, "parsed_final_marker_repeated")
        self.assertFalse(result.format_valid)

    def test_invalid_option_is_rejected(self):
        result = extract_answer("Final Answer: E", CHOICES)
        self.assertIsNone(result.answer)
        self.assertEqual(result.parse_status, "invalid_choice")

    def test_option_text_is_never_used_for_fuzzy_matching(self):
        result = extract_answer("I believe the diagnosis is Alpha.", CHOICES)
        self.assertIsNone(result.answer)
        self.assertEqual(result.parse_status, "missing")

    def test_empty_response_is_rejected(self):
        result = extract_answer("  \n", CHOICES)
        self.assertIsNone(result.answer)
        self.assertEqual(result.parse_status, "empty")


if __name__ == "__main__":
    unittest.main()
