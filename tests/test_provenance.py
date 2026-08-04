import unittest

from data_pipeline.verify_benchmark_provenance import (
    _compare_dataset,
    _normalise_medmcqa,
    _normalise_medqa,
)


class ProvenanceMappingTests(unittest.TestCase):
    def test_medqa_mapping(self):
        row = {
            "sent1": "Question?",
            "sent2": "",
            "ending0": "One",
            "ending1": "Two",
            "ending2": "Three",
            "ending3": "Four",
            "label": 2,
        }
        self.assertEqual(
            _normalise_medqa(row),
            {
                "question": "Question?",
                "choices": {"A": "One", "B": "Two", "C": "Three", "D": "Four"},
                "answer": "C",
            },
        )

    def test_medmcqa_mapping(self):
        row = {
            "question": "Question?",
            "opa": "One",
            "opb": "Two",
            "opc": "Three",
            "opd": "Four",
            "cop": 3,
        }
        self.assertEqual(_normalise_medmcqa(row)["answer"], "D")

    def test_comparison_detects_answer_mismatch(self):
        local = [
            {
                "question": "Question?",
                "options": {"A": "One", "B": "Two", "C": "Three", "D": "Four"},
                "answer_idx": "A",
            }
        ]
        official = [
            {
                "question": "Question?",
                "opa": "One",
                "opb": "Two",
                "opc": "Three",
                "opd": "Four",
                "cop": 1,
            }
        ]
        report = _compare_dataset(local, official, _normalise_medmcqa)
        self.assertEqual(report["status"], "mismatch")
        self.assertEqual(report["first_mismatches"][0]["fields"], ["answer"])


if __name__ == "__main__":
    unittest.main()
