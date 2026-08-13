import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from data_pipeline.audit_validator_recovery import audit_recovery, render_markdown
from data_pipeline.run_validator_recovery import SOURCE_FILES


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(value) + "\n" for value in values), encoding="utf-8"
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ValidatorRecoveryAuditTests(unittest.TestCase):
    def test_audit_is_aggregate_strict_and_compares_old_unavailable_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "pilot"
            output_dir = run_dir / "recovery"
            output_dir.mkdir(parents=True)
            identity = {
                "config_sha256": "c" * 64,
                "questions_sha256": "q" * 64,
                "generation_git_commit": "g" * 40,
            }
            metadata = {
                **{key: identity[key] for key in ("config_sha256", "questions_sha256")},
                "preflight": {"screener_runtime": {"git_commit": identity["generation_git_commit"]}},
            }
            write_json(run_dir / "metadata.json", metadata)

            keys = [("private-a", 0), ("private-b", 0)]
            teachers = []
            screeners = []
            validators = []
            recovered = []
            for index, key in enumerate(keys):
                record_id, candidate_index = key
                benchmark = "medqa" if index == 0 else "medmcqa"
                teachers.append({
                    "record_id": record_id,
                    "candidate_index": candidate_index,
                    "record": {"benchmark": benchmark, "question": "SECRET"},
                    "rule_check": {"steps": ["PRIVATE TRAJECTORY"]},
                })
                screeners.append({
                    "record_id": record_id,
                    "candidate_index": candidate_index,
                    "result": {"verdict": "pass" if index == 0 else "review"},
                })
                old_error = (
                    "JSONDecodeError: Unterminated string at private content"
                    if index == 0
                    else "ValueError: chat completion response has no text content"
                )
                validators.append({
                    "record_id": record_id,
                    "candidate_index": candidate_index,
                    "status": "api_or_parse_error",
                    "errors": [old_error],
                })
                detail = (
                    "json_syntax_unterminated_string"
                    if index == 0 else "response_no_text_content"
                )
                result = {
                    "trajectory_label": 1 - index,
                    "first_error_step": None if index == 0 else 0,
                    "answer_consistent": index == 0,
                    "problem_status": "ok",
                    "steps": [{
                        "index": 0,
                        "local_verdict": "correct" if index == 0 else "incorrect",
                        "prefix_label": 1 - index,
                        "error_codes": [] if index == 0 else ["medical_fact_error"],
                        "concise_reason": "PRIVATE REASON",
                    }],
                }
                recovered.append({
                    "record_id": record_id,
                    "candidate_index": candidate_index,
                    "source_error_details": [detail],
                    "status": "complete",
                    "result": result,
                    "usage": {"cost_cny": 0.1},
                    "attempt_diagnostics": [{
                        "status": "complete",
                        "finish_reason": "stop",
                        "routed_provider": "Provider",
                        "content_present": True,
                        "reasoning_tokens": 10,
                    }],
                })

            write_jsonl(run_dir / "teacher_events.jsonl", teachers)
            write_jsonl(run_dir / "screener_events.jsonl", screeners)
            write_jsonl(run_dir / "validator_events.jsonl", validators)
            write_jsonl(output_dir / "canary_events.jsonl", recovered)
            for name in SOURCE_FILES:
                path = run_dir / name
                if not path.exists():
                    write_jsonl(path, [])
            source_hashes = {name: sha256(run_dir / name) for name in SOURCE_FILES}

            config = {
                "schema_version": "medtrace.cot.validator-recovery.v2",
                "execution_enabled": False,
                "source_run_dir": str(run_dir),
                "output_subdir": "recovery",
                "source_identity": {**identity, "expected_failed_validator_events": 2},
                "canary": {
                    "total": 2,
                    "by_benchmark": {"medqa": 1, "medmcqa": 1},
                    "selection_method": "benchmark_stratified_error_detail_round_robin",
                },
                "validator": {
                    "provider": "openrouter", "model_id": "deepseek/deepseek-v4-pro",
                    "provider_version": "OpenRouter-DeepSeek-V4-Pro",
                    "base_url": "https://openrouter.ai/api/v1",
                    "api_key_env": "OPENROUTER_API_KEY", "prompt_version": "validator_v2",
                    "reasoning_effort": "high", "require_zero_data_retention": True,
                    "allow_provider_fallbacks": True, "response_format": "json_schema_strict",
                    "temperature": 0, "max_output_tokens": 8192, "max_retries": 0,
                    "timeout_seconds": 600,
                },
                "budget": {
                    "api_hard_cap_cny_equivalent": 5,
                    "stop_before_limit_fraction": 0.9, "usd_to_cny": 7.2,
                    "validator_usd_per_million_input_tokens": 2.1,
                    "validator_usd_per_million_output_tokens": 4.4,
                },
            }
            config_path = root / "recovery.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            write_json(output_dir / "private_manifest.json", {
                "schema_version": "medtrace.validator-recovery-private-manifest.v2",
                "config_sha256": sha256(config_path),
                "source_artifact_sha256": source_hashes,
                "selected_keys": [list(key) for key in keys],
            })
            write_json(output_dir / "canary_metadata.json", {
                "status": "complete", "selected_events": 2, "completed_events": 2,
                "source_identity": identity, "source_artifacts_unchanged": True,
                "spent_cny_equivalent": 0.2,
            })

            # The production config is deliberately frozen to the real 6-event run;
            # this synthetic fixture exercises aggregation with two private keys.
            with patch(
                "data_pipeline.audit_validator_recovery.validate_recovery_config"
            ):
                report = audit_recovery(config_path)
            serialized = json.dumps(report)
            markdown = render_markdown(report)

            self.assertTrue(report["gates"]["integrity_passed"])
            self.assertTrue(report["gates"]["transport_and_contract_passed"])
            self.assertFalse(report["gates"]["semantic_auto_approval"])
            self.assertEqual(report["counts"]["old_validator_unavailable"], 2)
            self.assertEqual(
                report["recovery"]["strict_trajectory_label_counts"],
                {"int:0": 1, "int:1": 1},
            )
            self.assertEqual(report["diagnostics"]["reasoning_tokens"]["sum"], 20)
            for private_value in (
                "private-a", "private-b", "SECRET", "PRIVATE TRAJECTORY", "PRIVATE REASON"
            ):
                self.assertNotIn(private_value, serialized)
                self.assertNotIn(private_value, markdown)


if __name__ == "__main__":
    unittest.main()
