from pathlib import Path
import unittest

from evaluation.run_eval import _sha256, _select_records, _validate_runtime_manifest


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


class RuntimeManifestTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "model_revision": "model-sha",
            "expected_vllm_version": "0.24.0",
            "expected_vllm_image": "vllm-image@sha256:digest",
            "expected_native_requirements_file": "requirements.txt",
        }

    def test_docker_runtime_requires_pinned_image(self):
        manifest = {
            "runtime_backend": "docker",
            "model_id": self.config["model"],
            "model_revision": "model-sha",
            "requested_image": "vllm-image@sha256:digest",
            "packages": {"vllm": "0.24.0"},
        }
        _validate_runtime_manifest(self.config, manifest)

    def test_native_runtime_requires_current_requirements_hash(self):
        requirements = Path("requirements.txt").resolve()
        manifest = {
            "runtime_backend": "native",
            "model_id": self.config["model"],
            "model_revision": "model-sha",
            "requested_image": None,
            "packages": {"vllm": "0.24.0"},
            "requirements": {"sha256": _sha256(requirements)},
        }
        _validate_runtime_manifest(self.config, manifest)

    def test_unknown_backend_is_rejected(self):
        manifest = {
            "runtime_backend": "mystery",
            "model_id": self.config["model"],
            "model_revision": "model-sha",
            "packages": {"vllm": "0.24.0"},
        }
        with self.assertRaisesRegex(ValueError, "unsupported runtime backend"):
            _validate_runtime_manifest(self.config, manifest)


if __name__ == "__main__":
    unittest.main()
