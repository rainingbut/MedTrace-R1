import json
from pathlib import Path
import tempfile
import unittest

import yaml

from data_pipeline.cot_recovery_config import validate_recovery_config
from data_pipeline.run_validator_recovery import (
    _prepare_private_manifest,
    _preview,
    select_canary,
)


class ValidatorRecoveryTests(unittest.TestCase):
    def test_committed_recovery_config_is_safe_and_frozen(self):
        path = Path(__file__).resolve().parents[1] / "configs/cot/validator_recovery_v2.yaml"
        with path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        validate_recovery_config(config)
        self.assertFalse(config["execution_enabled"])
        self.assertEqual(config["validator"]["max_output_tokens"], 8192)
        self.assertEqual(config["budget"]["api_hard_cap_cny_equivalent"], 5)
        runtime = dict(config)
        runtime["execution_enabled"] = True
        validate_recovery_config(runtime, execute=True)

    def test_private_manifest_refuses_changed_resume_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "private_manifest.json"
            kwargs = {
                "config_sha256": "a" * 64,
                "source_hashes": {"metadata.json": "b" * 64},
                "selected": [("private-id", 0)],
            }
            _prepare_private_manifest(path, **kwargs)
            _prepare_private_manifest(path, **kwargs)
            value = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(value["selected_keys"], [["private-id", 0]])
            with self.assertRaisesRegex(RuntimeError, "manifest changed"):
                _prepare_private_manifest(
                    path,
                    config_sha256="c" * 64,
                    source_hashes=kwargs["source_hashes"],
                    selected=kwargs["selected"],
                )

    def test_canary_selection_is_deterministic_stratified_and_private(self):
        teachers = {}
        validators = {}
        details = [
            ("mqa-a", "medqa", "JSONDecodeError: Unterminated string"),
            ("mqa-b", "medqa", "ValueError: chat completion response has no text content"),
            ("mqa-c", "medqa", "JSONDecodeError: Expecting property name"),
            ("mqa-d", "medqa", "JSONDecodeError: Expecting value"),
            ("mqa-e", "medqa", "JSONDecodeError: Unterminated string"),
            ("mm-a", "medmcqa", "JSONDecodeError: Unterminated string"),
            ("mm-b", "medmcqa", "ValueError: chat completion response has no text content"),
            ("mm-c", "medmcqa", "JSONDecodeError: Expecting value"),
        ]
        for record_id, benchmark, error in details:
            key = (record_id, 0)
            teachers[key] = {
                "record_id": record_id, "candidate_index": 0,
                "record": {"benchmark": benchmark},
            }
            validators[key] = {
                "record_id": record_id, "candidate_index": 0,
                "status": "api_or_parse_error", "errors": [error],
            }
        selected = select_canary(
            teachers, validators, {"medqa": 4, "medmcqa": 2}
        )
        self.assertEqual(len(selected), 6)
        preview = _preview(selected, teachers, validators)
        self.assertEqual(preview["by_benchmark"], {"medmcqa": 2, "medqa": 4})
        self.assertFalse(preview["contains_private_text_or_ids"])
        self.assertNotIn("mqa-a", str(preview))
        self.assertEqual(
            selected,
            select_canary(teachers, validators, {"medqa": 4, "medmcqa": 2}),
        )


if __name__ == "__main__":
    unittest.main()
