import unittest

from scripts.estimate_full_run import estimate_run


class EstimateFullRunTests(unittest.TestCase):
    def setUp(self):
        self.metadata = {
            "dry_run": False,
            "completed_records": 200,
            "elapsed_seconds": 360.0,
        }
        self.metrics = {
            "overall": {
                "accuracy": 0.6,
                "parse_rate": 0.99,
                "format_rate": 0.98,
                "truncation_rate": 0.0,
            }
        }
        self.manifest = {"combined_records": 5456}

    def test_estimates_runtime_and_cost(self):
        report = estimate_run(
            self.metadata, self.metrics, self.manifest, hourly_rate=2.0, safety_factor=1.25
        )
        self.assertEqual(report["projected_compute_hours"], 2.728)
        self.assertEqual(report["projected_hours_with_safety_factor"], 3.41)
        self.assertEqual(report["projected_compute_cost"], 5.46)
        self.assertEqual(report["recommended_booking_hours"], 4)

    def test_rejects_dry_run(self):
        self.metadata["dry_run"] = True
        with self.assertRaisesRegex(ValueError, "real model pilot"):
            estimate_run(self.metadata, self.metrics, self.manifest, None, 1.25)

    def test_rejects_unsafe_factor(self):
        with self.assertRaisesRegex(ValueError, "at least 1"):
            estimate_run(self.metadata, self.metrics, self.manifest, None, 0.5)


if __name__ == "__main__":
    unittest.main()
