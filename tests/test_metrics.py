import unittest

from evaluation.metrics import score_records


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            {
                "id": "one",
                "benchmark": "medqa",
                "answer": "A",
                "extracted_answer": "A",
                "format_valid": True,
                "completion_tokens": 10,
                "latency_seconds": 1.0,
                "error": None,
            },
            {
                "id": "two",
                "benchmark": "medqa",
                "answer": "B",
                "extracted_answer": "A",
                "format_valid": True,
                "completion_tokens": 20,
                "latency_seconds": 3.0,
                "error": None,
            },
            {
                "id": "three",
                "benchmark": "medmcqa",
                "answer": "C",
                "extracted_answer": None,
                "format_valid": False,
                "completion_tokens": None,
                "latency_seconds": 2.0,
                "error": "timeout",
            },
        ]

    def test_scores_once_against_the_fixed_prediction(self):
        metrics = score_records(self.records)
        overall = metrics["overall"]
        self.assertEqual(overall["total"], 3)
        self.assertEqual(overall["correct"], 1)
        self.assertEqual(overall["accuracy"], 0.333333)
        self.assertEqual(overall["parse_rate"], 0.666667)
        self.assertEqual(overall["format_rate"], 0.666667)
        self.assertEqual(overall["errors"], 1)
        self.assertEqual(overall["truncation_rate"], 0.0)
        self.assertEqual(overall["average_completion_tokens"], 15.0)
        self.assertEqual(overall["average_latency_seconds"], 2.0)

    def test_reports_benchmarks_separately(self):
        metrics = score_records(self.records)
        self.assertEqual(metrics["by_benchmark"]["medqa"]["accuracy"], 0.5)
        self.assertEqual(metrics["by_benchmark"]["medmcqa"]["accuracy"], 0.0)

    def test_duplicate_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate prediction id"):
            score_records([self.records[0], self.records[0]])

    def test_missing_required_fields_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing fields"):
            score_records([{"id": "broken"}])


if __name__ == "__main__":
    unittest.main()
