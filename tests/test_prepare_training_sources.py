import unittest

from data_pipeline.prepare_training_sources import (
    _normalise_medmcqa,
    _normalise_medqa,
    _source_url,
)


class TrainingSourceNormalisationTests(unittest.TestCase):
    def test_medqa_train_mapping(self):
        row = {
            "id": "train-00001",
            "sent1": "Question?",
            "sent2": "",
            "ending0": "One",
            "ending1": "Two",
            "ending2": "Three",
            "ending3": "Four",
            "label": 2,
        }
        record = _normalise_medqa(row)
        self.assertEqual(record["source_id"], "train-00001")
        self.assertEqual(record["answer"], "C")

    def test_medmcqa_train_mapping_does_not_copy_explanation(self):
        row = {
            "id": "source-id",
            "question": "Question?",
            "opa": "One",
            "opb": "Two",
            "opc": "Three",
            "opd": "Four",
            "cop": 1,
            "exp": "Reference explanation must not enter generation input.",
            "subject_name": "Medicine",
            "topic_name": "Topic",
            "choice_type": "single",
        }
        record = _normalise_medmcqa(row)
        self.assertEqual(record["answer"], "B")
        self.assertNotIn("exp", record)
        self.assertTrue(record["reference_explanation_present"])

    def test_download_url_contains_exact_revision(self):
        revision = "a" * 40
        url = _source_url("owner/dataset", revision, "data/train.parquet")
        self.assertIn(revision, url)
        self.assertTrue(url.endswith("/data/train.parquet"))


if __name__ == "__main__":
    unittest.main()
