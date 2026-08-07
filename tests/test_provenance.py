import hashlib
from pathlib import Path
import tempfile
import unittest

from data_pipeline.verify_benchmark_provenance import (
    SOURCES,
    _cached_file_matches,
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

    def test_cross_dataset_cache_is_rejected_by_file_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            cached_file = Path(directory) / "split.parquet"
            cached_file.write_bytes(b"medqa-page-returned-for-medmcqa")
            actual_sha = hashlib.sha256(cached_file.read_bytes()).hexdigest()
            self.assertTrue(_cached_file_matches(cached_file, actual_sha))
            self.assertFalse(_cached_file_matches(cached_file, "0" * 64))

    def test_sources_pin_distinct_files_and_hashes(self):
        self.assertEqual(len(SOURCES), 2)
        self.assertEqual(len({source["data_file"] for source in SOURCES}), 2)
        self.assertEqual(len({source["file_sha256"] for source in SOURCES}), 2)


if __name__ == "__main__":
    unittest.main()
