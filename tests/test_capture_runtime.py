import unittest

from scripts.capture_runtime import validate_server_command


class ServerCommandValidationTests(unittest.TestCase):
    def test_accepts_native_vllm_command(self):
        validate_server_command(
            [
                "vllm",
                "serve",
                "Qwen/Qwen2.5-7B-Instruct",
                "--revision",
                "model-sha",
            ],
            "Qwen/Qwen2.5-7B-Instruct",
            "model-sha",
        )

    def test_accepts_docker_entrypoint_arguments(self):
        validate_server_command(
            [
                "--model",
                "Qwen/Qwen2.5-7B-Instruct",
                "--revision",
                "model-sha",
            ],
            "Qwen/Qwen2.5-7B-Instruct",
            "model-sha",
        )

    def test_rejects_unexpected_revision(self):
        with self.assertRaisesRegex(RuntimeError, "unexpected model revision"):
            validate_server_command(
                [
                    "vllm",
                    "serve",
                    "Qwen/Qwen2.5-7B-Instruct",
                    "--revision",
                    "wrong-sha",
                ],
                "Qwen/Qwen2.5-7B-Instruct",
                "model-sha",
            )


if __name__ == "__main__":
    unittest.main()
