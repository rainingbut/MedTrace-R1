"""Minimal OpenAI-compatible HTTP client and strict judge response validation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from typing import Any
from urllib import request as urllib_request


@dataclass(frozen=True)
class ChatResult:
    content: str
    reasoning_content: str | None
    request_id: str | None
    finish_reason: str | None
    input_tokens: int | None
    output_tokens: int | None
    billed_cost_usd: float | None
    routed_provider: str | None


def require_api_key(environment_name: str, *, allow_empty: bool = False) -> str:
    value = os.environ.get(environment_name)
    if value and value.strip():
        return value.strip()
    if allow_empty:
        return "EMPTY"
    raise RuntimeError(f"required API key environment variable is missing: {environment_name}")


def post_chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int | None,
    timeout_seconds: float,
    top_p: float | None = None,
    response_format_json: bool = False,
    extra_body: dict[str, Any] | None = None,
) -> ChatResult:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "stream": False,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if top_p is not None:
        payload["top_p"] = top_p
    if response_format_json:
        payload["response_format"] = {"type": "json_object"}
    if extra_body:
        payload.update(extra_body)

    url = f"{base_url.rstrip('/')}/chat/completions"
    request = urllib_request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib_request.urlopen(request, timeout=timeout_seconds) as response:
        body = json.loads(response.read().decode("utf-8"))
        header_request_id = response.headers.get("x-request-id")
    choice = body["choices"][0]
    message = choice["message"]
    usage = body.get("usage") or {}
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("chat completion response has no text content")
    billed_cost = _optional_float(usage.get("cost"))
    if billed_cost is not None and (not math.isfinite(billed_cost) or billed_cost < 0):
        raise ValueError("chat completion usage.cost must be finite and non-negative")
    return ChatResult(
        content=content,
        reasoning_content=message.get("reasoning_content"),
        request_id=str(body.get("id") or header_request_id or "") or None,
        finish_reason=choice.get("finish_reason"),
        input_tokens=_optional_int(usage.get("prompt_tokens", usage.get("input_tokens"))),
        output_tokens=_optional_int(
            usage.get("completion_tokens", usage.get("output_tokens"))
        ),
        billed_cost_usd=billed_cost,
        routed_provider=str(body.get("provider") or "") or None,
    )


def get_models(base_url: str, api_key: str, timeout_seconds: float) -> set[str]:
    request = urllib_request.Request(
        f"{base_url.rstrip('/')}/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    with urllib_request.urlopen(request, timeout=timeout_seconds) as response:
        body = json.loads(response.read().decode("utf-8"))
    return {str(item["id"]) for item in body.get("data", [])}


def parse_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1])
            if text.lstrip().lower().startswith("json"):
                text = text.lstrip()[4:].lstrip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("judge response must be one JSON object")
    return value


def validate_screener_result(value: dict[str, Any], step_count: int) -> dict[str, Any]:
    required = {
        "verdict",
        "suspected_first_error_step",
        "error_codes",
        "concise_reason",
    }
    if set(value) != required:
        raise ValueError(f"screener JSON keys differ: {sorted(set(value) ^ required)}")
    if value["verdict"] not in {"pass", "reject", "review"}:
        raise ValueError("invalid screener verdict")
    first = value["suspected_first_error_step"]
    if first is not None and (not isinstance(first, int) or not 0 <= first < step_count):
        raise ValueError("invalid screener first-error index")
    if not isinstance(value["error_codes"], list) or not all(
        isinstance(code, str) for code in value["error_codes"]
    ):
        raise ValueError("invalid screener error_codes")
    if not isinstance(value["concise_reason"], str):
        raise ValueError("invalid screener concise_reason")
    return value


def validate_validator_result(value: dict[str, Any], step_count: int) -> dict[str, Any]:
    required = {
        "trajectory_label",
        "first_error_step",
        "answer_consistent",
        "problem_status",
        "steps",
    }
    if set(value) != required:
        raise ValueError(f"validator JSON keys differ: {sorted(set(value) ^ required)}")
    if value["trajectory_label"] not in {0, 1}:
        raise ValueError("invalid trajectory_label")
    if not isinstance(value["answer_consistent"], bool):
        raise ValueError("answer_consistent must be boolean")
    if value["problem_status"] not in {"ok", "ambiguous", "bad_gold"}:
        raise ValueError("invalid problem_status")
    steps = value["steps"]
    if not isinstance(steps, list) or len(steps) != step_count:
        raise ValueError("validator must return exactly one result per step")
    expected_prefix = 1
    first_bad: int | None = None
    for index, step in enumerate(steps):
        required_step = {
            "index",
            "local_verdict",
            "prefix_label",
            "error_codes",
            "concise_reason",
        }
        if not isinstance(step, dict) or set(step) != required_step:
            raise ValueError("validator step keys differ from contract")
        if step["index"] != index:
            raise ValueError("validator step indices are not consecutive")
        if step["local_verdict"] not in {"correct", "incorrect", "uncertain"}:
            raise ValueError("invalid local_verdict")
        if step["local_verdict"] != "correct" and first_bad is None:
            first_bad = index
            expected_prefix = 0
        if step["prefix_label"] != expected_prefix:
            raise ValueError("prefix labels do not become and remain zero after first error")
        if not isinstance(step["error_codes"], list) or not isinstance(
            step["concise_reason"], str
        ):
            raise ValueError("invalid validator step details")
    if value["first_error_step"] != first_bad:
        raise ValueError("first_error_step differs from step verdicts")
    expected_label = int(
        first_bad is None
        and value["answer_consistent"]
        and value["problem_status"] == "ok"
    )
    if value["trajectory_label"] != expected_label:
        raise ValueError("trajectory_label is inconsistent with validation details")
    return value


def _optional_int(value: object) -> int | None:
    return int(value) if value is not None else None


def _optional_float(value: object) -> float | None:
    return float(value) if value is not None else None
