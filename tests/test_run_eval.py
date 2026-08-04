import unittest

from evaluation.run_eval import _select_records


class RecordSelectionTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            {"id": "m1", "benchmark": "medqa"},
            {"id": "m2", "benchmark": "medqa"},
            {"id": "c1", "benchmark": "medmcqa"},
            {"id": "c2", "benchmark": "medmcqa"},
        ]

    def test_limit_per_benchmark_is_stratified(self):
        selected = _select_records(self.records, None, 1)
        self.assertEqual([record["id"] for record in selected], ["m1", "c1"])

    def test_global_limit_preserves_order(self):
        selected = _select_records(self.records, 2, None)
        self.assertEqual([record["id"] for record in selected], ["m1", "m2"])

    def test_limits_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "only one"):
            _select_records(self.records, 1, 1)

    def test_non_positive_limit_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            _select_records(self.records, None, 0)


if __name__ == "__main__":
    unittest.main()
