import copy
import unittest

from data_pipeline.cot_isolation import (
    EvaluationIsolationIndex,
    content_sha256,
    validate_train_record,
)


REVISION = "1" * 40
FILE_SHA = "2" * 64


def make_record(
    *,
    record_id="train_1",
    split="train",
    question="Which treatment is preferred for this condition?",
    choices=None,
    answer="A",
):
    choices = choices or {"A": "Treatment one", "B": "Treatment two"}
    return {
        "id": record_id,
        "benchmark": "medqa",
        "split": split,
        "question": question,
        "choices": choices,
        "answer": answer,
        "source_revision": REVISION,
        "source_file_sha256": FILE_SHA,
        "content_sha256": content_sha256(question, choices),
    }


class TrainOnlyGateTests(unittest.TestCase):
    def test_accepts_valid_train_record(self):
        validate_train_record(make_record())

    def test_rejects_test_and_validation_splits(self):
        for split in ("test", "validation", "dev"):
            with self.subTest(split=split):
                with self.assertRaisesRegex(ValueError, "split='train'"):
                    validate_train_record(make_record(split=split))

    def test_rejects_unpinned_source(self):
        record = make_record()
        record["source_file_sha256"] = "REPLACE_AFTER_DOWNLOAD"
        with self.assertRaisesRegex(ValueError, "source_file_sha256"):
            validate_train_record(record)

    def test_rejects_tampered_content(self):
        record = make_record()
        record["question"] = "Changed after hashing"
        with self.assertRaisesRegex(ValueError, "content_sha256"):
            validate_train_record(record)


class EvaluationOverlapTests(unittest.TestCase):
    def setUp(self):
        self.evaluation = make_record(record_id="eval_1", split="test")
        self.index = EvaluationIsolationIndex([self.evaluation])

    def test_rejects_exact_evaluation_content(self):
        training = copy.deepcopy(self.evaluation)
        training.update(
            {
                "id": "train_copy",
                "split": "train",
                "source_revision": REVISION,
                "source_file_sha256": FILE_SHA,
            }
        )
        with self.assertRaisesRegex(ValueError, "exact_content"):
            validate_train_record(training, self.index)

    def test_rejects_normalized_question_with_changed_choices(self):
        training = make_record(
            question="WHICH treatment is preferred for this condition!!!",
            choices={"A": "Different one", "B": "Different two"},
        )
        with self.assertRaisesRegex(ValueError, "normalised_question"):
            validate_train_record(training, self.index)

    def test_rejects_high_similarity_question(self):
        evaluation = make_record(
            record_id="eval_long",
            split="test",
            question=(
                "A patient with fever cough chest pain and low oxygen saturation "
                "has a focal opacity on chest radiography which diagnosis is likely"
            ),
        )
        index = EvaluationIsolationIndex([evaluation])
        training = make_record(
            question=(
                "A patient with fever cough chest pain and low oxygen saturation "
                "has a focal opacity on chest radiography which diagnosis is most likely"
            )
        )
        with self.assertRaisesRegex(ValueError, "near_question"):
            validate_train_record(training, index)

    def test_does_not_flag_shared_generic_medical_words(self):
        evaluation = make_record(
            record_id="eval_generic",
            split="test",
            question="Which treatment is most appropriate for a patient with pneumonia?",
        )
        index = EvaluationIsolationIndex([evaluation])
        training = make_record(
            question="Which diagnostic imaging is most appropriate for a patient with fracture?"
        )
        validate_train_record(training, index)


if __name__ == "__main__":
    unittest.main()
