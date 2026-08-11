"""Run the resumable stage-2 pilot; currently real provider calls are locked."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import yaml

from data_pipeline.cot_budget import BudgetLedger
from data_pipeline.cot_prompts import build_teacher_prompt
from data_pipeline.cot_rules import check_teacher_response


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cot/pilot_v1.yaml")
    parser.add_argument("--pilot-manifest", default=None)
    parser.add_argument("--run-dir", default="results/cot/pilot_v1_dryrun")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
    return records


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _mock_teacher_response(record: dict[str, Any], candidate_index: int) -> str:
    gold = str(record["answer"])
    other = next(label for label in record["choices"] if label != gold)
    normal = [
        "Identify the clinically relevant findings in the question.",
        "Compare those findings with the mechanisms represented by the options.",
        "Select the option that best fits the combined evidence.",
    ]
    if candidate_index == 0:
        steps, answer = normal, gold
    elif candidate_index == 1:
        steps = [normal[0], "MOCK_ERROR: assert an intentionally false medical fact.", normal[2]]
        answer = gold
    elif candidate_index == 2:
        steps, answer = [normal[0], normal[0], normal[2]], gold
    else:
        steps, answer = normal, other
    return "".join(f"<step>{step}</step>" for step in steps) + f"<answer>{answer}</answer>"


def _mock_review(candidate_index: int, step_count: int) -> tuple[dict[str, Any], dict[str, Any]]:
    if candidate_index == 1:
        screening = {
            "status": "complete",
            "verdict": "review",
            "suspected_first_error_step": 1,
            "error_codes": ["medical_fact_error"],
        }
        verdicts = ["correct", "incorrect", "correct"]
        verification_steps = [
            {
                "index": index,
                "local_verdict": verdict,
                "prefix_label": 1 if index == 0 else 0,
                "error_codes": ["medical_fact_error"] if index == 1 else [],
                "concise_reason": "mock negative" if index == 1 else "mock correct",
            }
            for index, verdict in enumerate(verdicts[:step_count])
        ]
        verification = {
            "trajectory_label": 0,
            "first_error_step": 1,
            "answer_consistent": True,
            "problem_status": "ok",
            "steps": verification_steps,
        }
    else:
        screening = {
            "status": "complete",
            "verdict": "pass",
            "suspected_first_error_step": None,
            "error_codes": [],
        }
        verification = {
            "trajectory_label": 1,
            "first_error_step": None,
            "answer_consistent": True,
            "problem_status": "ok",
            "steps": [
                {
                    "index": index,
                    "local_verdict": "correct",
                    "prefix_label": 1,
                    "error_codes": [],
                    "concise_reason": "mock correct",
                }
                for index in range(step_count)
            ],
        }
    return screening, verification


def _event_to_canonical(event: dict[str, Any], config: dict[str, Any]) -> dict[str, Any] | None:
    rule = event["rule_check"]
    if not rule["passed"]:
        return None
    record = event["record"]
    screening, verification = _mock_review(
        int(event["candidate_index"]), len(rule["steps"])
    )
    screening.update(
        {
            "model_id": config["screener"]["model_id"],
            "model_revision": config["screener"]["model_revision"],
            "prompt_version": config["screener"]["prompt_version"],
        }
    )
    verification.update(
        {
            "status": "complete",
            "provider": config["validator"]["provider"],
            "model_id": config["validator"]["model_id"],
            "provider_version": config["validator"]["provider_version"],
            "prompt_version": config["validator"]["prompt_version"],
            "request_id": None,
            "usage": {"input_tokens": 0, "output_tokens": 0, "cost_cny": 0.0},
        }
    )
    raw_digest = hashlib.sha256(event["raw_response"].encode("utf-8")).hexdigest()
    trajectory_digest = hashlib.sha256(
        f"{record['id']}:{event['candidate_index']}:{raw_digest}".encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": "medtrace.cot.v1",
        "trajectory_id": f"sha256:{trajectory_digest}",
        "source": {
            "dataset": record["benchmark"],
            "config": "default",
            "split": record["split"],
            "source_revision": record["source_revision"],
            "source_file_sha256": record["source_file_sha256"],
            "source_id": record["source_id"],
            "source_index": record["source_index"],
            "content_sha256": record["content_sha256"],
        },
        "problem": {
            "question": record["question"],
            "choices": record["choices"],
            "gold_answer": record["answer"],
        },
        "generation": {
            "run_mode": "dry_run",
            "provider": config["teacher"]["provider"],
            "model_id": config["teacher"]["model_id"],
            "provider_version": config["teacher"]["model_id"],
            "model_revision": None,
            "prompt_version": config["teacher"]["prompt_version"],
            "candidate_index": event["candidate_index"],
            "decoding": {
                key: config["teacher"][key]
                for key in ("temperature", "top_p", "max_output_tokens")
            },
            "request_id": None,
            "usage": {"input_tokens": 0, "output_tokens": 0, "cost_cny": 0.0},
        },
        "trajectory": {
            "steps": [
                {"index": index, "text": text}
                for index, text in enumerate(rule["steps"])
            ],
            "predicted_answer": rule["predicted_answer"],
            "raw_response_sha256": raw_digest,
        },
        "rule_check": {"passed": True, "failure_codes": []},
        "screening": screening,
        "verification": verification,
        "disposition": (
            "sft_accept" if verification["trajectory_label"] == 1 else "prm_only"
        ),
    }


def main() -> None:
    args = parse_args()
    if not args.dry_run:
        raise RuntimeError(
            "real provider calls are locked in this stage; use --dry-run"
        )
    config_path = _repo_path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config.get("execution_enabled") is not False:
        raise ValueError("dry-run requires the committed execution lock")

    pilot_manifest_path = _repo_path(
        args.pilot_manifest or config["outputs"]["pilot_input_manifest"]
    )
    with pilot_manifest_path.open("r", encoding="utf-8") as handle:
        pilot_manifest = json.load(handle)
    questions_path = _repo_path(pilot_manifest["questions_file"])
    if _sha256(questions_path) != pilot_manifest["questions_sha256"]:
        raise ValueError("pilot questions hash differs from input manifest")
    records = _load_jsonl(questions_path)

    run_dir = _repo_path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = run_dir / "metadata.json"
    config_sha256 = _sha256(config_path)
    questions_sha256 = _sha256(questions_path)
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            previous_metadata = json.load(handle)
        if previous_metadata.get("dry_run") is not True:
            raise ValueError("cannot resume a non-dry run as dry-run")
        if previous_metadata.get("config_sha256") != config_sha256:
            raise ValueError("cannot resume: pilot config hash changed")
        if previous_metadata.get("questions_sha256") != questions_sha256:
            raise ValueError("cannot resume: pilot questions hash changed")
    events_path = run_dir / "generation_events.jsonl"
    existing = _load_jsonl(events_path)
    completed = {
        (str(event["record"]["id"]), int(event["candidate_index"]))
        for event in existing
    }
    if len(completed) != len(existing):
        raise ValueError("generation event log contains duplicate candidate keys")

    budget = config["budget"]
    ledger = BudgetLedger(
        hard_cap_cny=float(budget["api_hard_cap_cny_equivalent"]),
        stop_fraction=float(budget["stop_before_limit_fraction"]),
    )
    rule_config = config["rule_filter"]
    with events_path.open("a", encoding="utf-8", newline="\n") as handle:
        for record in records:
            for candidate_index in range(int(config["sampling"]["candidates_per_question"])):
                key = (str(record["id"]), candidate_index)
                if key in completed:
                    continue
                ledger.assert_can_spend(0.0)
                prompt = build_teacher_prompt(record["question"], record["choices"])
                raw_response = _mock_teacher_response(record, candidate_index)
                checked = check_teacher_response(
                    raw_response,
                    record["choices"],
                    record["answer"],
                    min_steps=int(rule_config["min_steps"]),
                    max_steps=int(rule_config["max_steps"]),
                    max_trajectory_characters=int(
                        rule_config["max_trajectory_characters"]
                    ),
                    max_step_characters=int(rule_config["max_step_characters"]),
                    finish_reason="mock",
                )
                ledger.record(0.0)
                event = {
                    "schema_version": "medtrace.cot-generation-event.v1",
                    "record": record,
                    "candidate_index": candidate_index,
                    "teacher_prompt": prompt,
                    "raw_response": raw_response,
                    "finish_reason": "mock",
                    "rule_check": checked.to_dict(),
                    "api_called": False,
                    "request_id": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0, "cost_cny": 0.0},
                }
                handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()

    events = _load_jsonl(events_path)
    expected_events = len(records) * int(config["sampling"]["candidates_per_question"])
    if len(events) != expected_events:
        raise ValueError(f"event count mismatch: {len(events)} != {expected_events}")
    if any(event["api_called"] for event in events):
        raise ValueError("dry-run event unexpectedly recorded an API call")

    canonical = [
        value
        for event in events
        if (value := _event_to_canonical(event, config)) is not None
    ]
    _write_jsonl(run_dir / "canonical_trajectories.jsonl", canonical)
    sft = [
        {
            "id": item["trajectory_id"],
            "messages": [
                {
                    "role": "user",
                    "content": build_teacher_prompt(
                        item["problem"]["question"], item["problem"]["choices"]
                    ),
                },
                {
                    "role": "assistant",
                    "content": "".join(
                        f"<step>{step['text']}</step>"
                        for step in item["trajectory"]["steps"]
                    )
                    + f"<answer>{item['trajectory']['predicted_answer']}</answer>",
                },
            ],
            "source": item["source"],
        }
        for item in canonical
        if item["disposition"] == "sft_accept"
    ]
    _write_jsonl(run_dir / "sft_verified.jsonl", sft)
    prm: list[dict[str, Any]] = []
    for item in canonical:
        for step in item["verification"]["steps"]:
            prm.append(
                {
                    "trajectory_id": item["trajectory_id"],
                    "step_index": step["index"],
                    "prefix": [
                        value["text"]
                        for value in item["trajectory"]["steps"][: step["index"] + 1]
                    ],
                    "label": step["prefix_label"],
                    "error_codes": step["error_codes"],
                    "source": item["source"],
                }
            )
    _write_jsonl(run_dir / "process_train.jsonl", prm)

    metadata = {
        "schema_version": "medtrace.cot-dryrun.v1",
        "status": "complete",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": True,
        "api_calls": 0,
        "spent_cny": ledger.spent_cny,
        "config_sha256": config_sha256,
        "pilot_manifest_sha256": _sha256(pilot_manifest_path),
        "questions_sha256": questions_sha256,
        "questions": len(records),
        "events": len(events),
        "rule_passed": sum(event["rule_check"]["passed"] for event in events),
        "rule_rejected": sum(not event["rule_check"]["passed"] for event in events),
        "canonical_trajectories": len(canonical),
        "sft_records": len(sft),
        "prm_records": len(prm),
        "prm_positive": sum(record["label"] == 1 for record in prm),
        "prm_negative": sum(record["label"] == 0 for record in prm),
    }
    _write_json(metadata_path, metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
