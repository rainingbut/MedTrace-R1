import json
from pathlib import Path
import unittest

import yaml

from data_pipeline.cot_config import validate_pilot_config
from data_pipeline.cot_errors import ALL_ERROR_CODES
from data_pipeline.cot_prompts import (
    build_screener_prompt,
    build_teacher_prompt,
    build_validator_prompt,
)


ROOT = Path(__file__).resolve().parents[1]


class PromptTests(unittest.TestCase):
    def setUp(self):
        self.question = "Which option is correct?"
        self.choices = {"A": "Alpha", "B": "Beta"}

    def test_teacher_is_gold_blind_and_requires_xml_contract(self):
        prompt = build_teacher_prompt(self.question, self.choices)
        self.assertNotIn("gold_answer", prompt)
        self.assertIn("<step>...</step>", prompt)
        self.assertIn("<answer>X</answer>", prompt)

    def test_screener_receives_gold_and_is_conservative(self):
        prompt = build_screener_prompt(
            self.question, self.choices, "B", ["A claim"], "B"
        )
        self.assertIn('"gold_answer": "B"', prompt)
        self.assertIn("pass|reject|review", prompt)

    def test_validator_defines_prefix_labels_and_hides_screener(self):
        prompt = build_validator_prompt(
            self.question, self.choices, "B", ["A claim"], "B"
        )
        self.assertIn("prefix_label", prompt)
        self.assertIn("not given any earlier screener verdict", prompt)
        self.assertIn("medical_fact_error", prompt)

    def test_failure_taxonomy_is_unique(self):
        self.assertEqual(len(ALL_ERROR_CODES), len(set(ALL_ERROR_CODES)))


class FrozenArtifactTests(unittest.TestCase):
    def test_pilot_config_is_safe_but_not_ready_for_real_run(self):
        with (ROOT / "configs/cot/pilot_v1.yaml").open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        validate_pilot_config(config)
        with self.assertRaisesRegex(ValueError, "explicitly enabled"):
            validate_pilot_config(config, real_run=True)

    def test_canonical_schema_is_valid_json_and_train_only(self):
        with (ROOT / "schemas/cot_trajectory_v1.schema.json").open(
            encoding="utf-8"
        ) as handle:
            schema = json.load(handle)
        self.assertEqual(
            schema["properties"]["source"]["properties"]["split"]["const"],
            "train",
        )
        self.assertEqual(
            schema["properties"]["generation"]["properties"]["candidate_index"]["maximum"],
            3,
        )

    def test_source_manifest_schema_is_train_only(self):
        with (ROOT / "schemas/training_source_manifest_v1.schema.json").open(
            encoding="utf-8"
        ) as handle:
            schema = json.load(handle)
        item = schema["properties"]["sources"]["items"]
        self.assertEqual(item["properties"]["split"]["const"], "train")


if __name__ == "__main__":
    unittest.main()
