"""Deterministic answer extraction for multiple-choice evaluations.

The extractor deliberately avoids fuzzy matching against option text.  It only
accepts an answer when the model explicitly marks a choice as its answer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Iterable


_FINAL_LINE_RE = re.compile(
    r"^Final Answer:\s*([A-Z])$",
    flags=re.IGNORECASE,
)
_FINAL_MARKER_RE = re.compile(
    r"(?im)^\s*(?:\*\*)?final\s+answer(?:\*\*)?\s*[:：]\s*"
    r"(?:\*\*)?[\(\[]?([A-Z])[\)\]]?(?:\*\*)?[.!。]?\s*$"
)
_ENGLISH_EXPLICIT_RE = re.compile(
    r"(?i)\bthe\s+(?:final\s+)?answer\s+is\s*[:：]?\s*[\(\[]?([A-Z])[\)\]]?"
)
_CHINESE_EXPLICIT_RE = re.compile(
    r"(?:最终答案|答案)\s*(?:是|为|[:：])\s*[\(\[]?([A-Z])[\)\]]?",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class ExtractionResult:
    answer: str | None
    parse_status: str
    format_valid: bool
    candidates: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _normalise_choices(valid_choices: Iterable[str]) -> set[str]:
    choices = {str(choice).strip().upper() for choice in valid_choices}
    if not choices or any(len(choice) != 1 or not choice.isalpha() for choice in choices):
        raise ValueError("valid_choices must contain one-letter option labels")
    return choices


def _result_from_candidates(
    candidates: list[str], valid_choices: set[str], source: str
) -> ExtractionResult:
    normalised = tuple(candidate.upper() for candidate in candidates)
    invalid = tuple(candidate for candidate in normalised if candidate not in valid_choices)
    if invalid:
        return ExtractionResult(None, "invalid_choice", False, normalised)

    unique = set(normalised)
    if len(unique) > 1:
        return ExtractionResult(None, "ambiguous", False, normalised)
    if not normalised:
        return ExtractionResult(None, "missing", False, ())

    status = source if len(normalised) == 1 else f"{source}_repeated"
    return ExtractionResult(normalised[-1], status, False, normalised)


def extract_answer(text: str, valid_choices: Iterable[str]) -> ExtractionResult:
    """Extract an explicitly stated answer without inspecting option text.

    ``format_valid`` is intentionally stricter than successful extraction. It
    requires exactly one ``Final Answer: X`` marker and requires that marker to
    be the last non-empty line of the response.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    choices = _normalise_choices(valid_choices)
    stripped = text.strip()
    if not stripped:
        return ExtractionResult(None, "empty", False, ())

    final_candidates = _FINAL_MARKER_RE.findall(stripped)
    if final_candidates:
        result = _result_from_candidates(final_candidates, choices, "parsed_final_marker")
        if result.answer is None:
            return result

        last_nonempty_line = next(
            (line.strip() for line in reversed(stripped.splitlines()) if line.strip()), ""
        )
        strict_match = _FINAL_LINE_RE.fullmatch(last_nonempty_line)
        format_valid = (
            len(final_candidates) == 1
            and strict_match is not None
            and strict_match.group(1).upper() == result.answer
        )
        return ExtractionResult(
            result.answer,
            "parsed_strict" if format_valid else result.parse_status,
            format_valid,
            result.candidates,
        )

    explicit_candidates = _ENGLISH_EXPLICIT_RE.findall(stripped)
    explicit_candidates.extend(_CHINESE_EXPLICIT_RE.findall(stripped))
    return _result_from_candidates(explicit_candidates, choices, "parsed_explicit")
