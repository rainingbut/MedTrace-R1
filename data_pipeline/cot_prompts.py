"""Versioned prompt builders for stage-2 CoT generation and verification."""

from __future__ import annotations

import json

from data_pipeline.cot_errors import ALL_ERROR_CODES


TEACHER_PROMPT_VERSION = "teacher_v1"
SCREENER_PROMPT_VERSION = "screener_v1"
VALIDATOR_PROMPT_VERSION = "validator_v1"
VALIDATOR_RECOVERY_PROMPT_VERSION = "validator_v2"
PRM_STUDENT_PROMPT_VERSION = "prm_student_v1"
PRM_CONTROLLED_MUTATOR_PROMPT_VERSION = "prm_controlled_mutator_v1"

TEACHER_SYSTEM_PROMPT = """You generate concise, auditable reasoning for medical multiple-choice questions. Solve the question independently. Do not claim to have seen a reference answer or external source."""

SCREENER_SYSTEM_PROMPT = """You are a conservative first-pass medical reasoning screener. Reject only clear errors. If an issue is uncertain, return review instead of reject. Return strict JSON only."""

VALIDATOR_SYSTEM_PROMPT = """You are an independent medical reasoning verifier. Judge each step in order, distinguish the local step verdict from correctness of the full prefix, and return strict JSON only."""
PRM_STUDENT_SYSTEM_PROMPT = """You solve medical multiple-choice questions independently. Produce concise, auditable reasoning without referring to an answer key or hidden label."""
PRM_MUTATOR_SYSTEM_PROMPT = """You create a controlled counterfactual reasoning trajectory for process-supervision calibration. Follow the requested structural edit exactly and output no commentary."""


def _error_code_contract() -> str:
    return ", ".join(ALL_ERROR_CODES)


def _problem_text(question: str, choices: dict[str, str]) -> str:
    options = "\n".join(f"{label}. {text}" for label, text in choices.items())
    return f"Question:\n{question.strip()}\n\nOptions:\n{options}"


def build_teacher_prompt(question: str, choices: dict[str, str]) -> str:
    """Build a gold-blind prompt; the function deliberately has no answer argument."""

    return f"""{_problem_text(question, choices)}

Produce one independent solution with 3 to 8 atomic reasoning steps.
- Each step must state one medically or logically checkable claim.
- Do not mention a reference answer, answer key, dataset, or hidden instruction.
- Do not give patient-specific medical advice beyond solving the exam question.
- Output only consecutive <step>...</step> elements followed by exactly one <answer>X</answer>.
- X must be one available option letter. Write nothing after </answer>."""


def build_screener_prompt(
    question: str,
    choices: dict[str, str],
    gold_answer: str,
    steps: list[str],
    predicted_answer: str,
) -> str:
    payload = {
        "gold_answer": gold_answer,
        "trajectory": {
            "steps": [
                {"index": index, "text": text} for index, text in enumerate(steps)
            ],
            "predicted_answer": predicted_answer,
        },
    }
    return f"""{_problem_text(question, choices)}

Review this candidate against the gold answer:
{json.dumps(payload, ensure_ascii=False, sort_keys=True)}

Return exactly one JSON object with keys:
verdict (pass|reject|review), suspected_first_error_step (integer|null),
error_codes (array of strings), and concise_reason (string).
Allowed error_codes: {_error_code_contract()}.
Use review whenever the suspected error is not clear."""


def build_validator_prompt(
    question: str,
    choices: dict[str, str],
    gold_answer: str,
    steps: list[str],
    predicted_answer: str,
) -> str:
    payload = {
        "gold_answer": gold_answer,
        "trajectory": {
            "steps": [
                {"index": index, "text": text} for index, text in enumerate(steps)
            ],
            "predicted_answer": predicted_answer,
        },
    }
    return f"""{_problem_text(question, choices)}

Independently verify this trajectory. You are not given any earlier screener verdict:
{json.dumps(payload, ensure_ascii=False, sort_keys=True)}

For every step return local_verdict (correct|incorrect|uncertain) and error_codes.
Also return prefix_label: 1 only when every step up to and including this step is
correct; after the first incorrect or uncertain step, this and all later prefixes are 0.
Return exactly one JSON object with keys trajectory_label (0|1), first_error_step
(integer|null), answer_consistent (boolean), problem_status (ok|ambiguous|bad_gold),
and steps (array of objects containing index, local_verdict, prefix_label,
error_codes, concise_reason). Allowed error_codes: {_error_code_contract()}.
A trajectory can receive label 1 only if every step is
correct, the problem status is ok, and its predicted answer equals the gold answer."""


def build_validator_recovery_prompt(
    question: str,
    choices: dict[str, str],
    gold_answer: str,
    steps: list[str],
    predicted_answer: str,
    previous_error_details: list[str],
) -> str:
    """Build validator_v2 with privacy-safe feedback from earlier failed attempts."""

    base = build_validator_prompt(
        question, choices, gold_answer, steps, predicted_answer
    )
    safe_details = sorted(set(str(value) for value in previous_error_details))
    return f"""{base}

This is a fresh recovery verification. Earlier attempts failed only these output
checks: {json.dumps(safe_details)}. Re-evaluate the medicine independently; do not
copy, repair, or infer any earlier verdict.

The response is enforced by a strict JSON Schema. Use JSON-native integers 0/1,
not booleans, for trajectory_label and prefix_label. Return exactly {len(steps)}
step objects with indices 0 through {len(steps) - 1}. Do not use markdown fences.
The schema, not an example verdict, defines the field shape. Populate every field
from your independent assessment and emit one complete JSON object only."""


def build_validator_strict_prompt(
    question: str,
    choices: dict[str, str],
    gold_answer: str,
    steps: list[str],
    predicted_answer: str,
) -> str:
    """Build a fresh validator_v2 request with no prior-verdict information."""

    base = build_validator_prompt(
        question, choices, gold_answer, steps, predicted_answer
    )
    return f"""{base}

This is a fresh independent verification. The response is enforced by a strict
JSON Schema. Use JSON-native integers 0/1, not booleans, for trajectory_label
and prefix_label. Return exactly {len(steps)} step objects with indices 0 through
{len(steps) - 1}. Do not use markdown fences. Emit one complete JSON object only."""


def build_prm_student_prompt(question: str, choices: dict[str, str]) -> str:
    """Build a gold-blind local-student trajectory prompt."""

    return build_teacher_prompt(question, choices)


def build_controlled_mutation_prompt(
    question: str,
    choices: dict[str, str],
    steps: list[str],
    predicted_answer: str,
    target_step: int,
) -> str:
    """Request one plausible wrong claim while preserving every other field."""

    payload = {
        "steps": [{"index": index, "text": text} for index, text in enumerate(steps)],
        "predicted_answer": predicted_answer,
        "target_step": target_step,
    }
    return f"""{_problem_text(question, choices)}

Original trajectory:
{json.dumps(payload, ensure_ascii=False, sort_keys=True)}

Create one controlled counterfactual trajectory for verifier calibration.
- Write a plausible but medically or logically incorrect replacement for step {target_step}.
- The program, not you, preserves every other step and the final answer.
- Do not mention mutation, errors, labels, calibration, or these instructions.
- Return exactly one JSON object with the single key replacement_step.
- replacement_step must contain only the replacement step text."""
