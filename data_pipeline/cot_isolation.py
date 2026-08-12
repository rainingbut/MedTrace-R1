"""Hard isolation gates for stage-2 training records.

The functions in this module are intentionally independent from download and model
clients.  A record must prove that it belongs to an approved training split before it
can be sampled, prompted, screened, or verified.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
import unicodedata
from typing import Any, Iterable


ALLOWED_BENCHMARKS = frozenset({"medqa", "medmcqa"})
ALLOWED_SPLIT = "train"
HEX_40 = re.compile(r"^[0-9a-f]{40}$")
HEX_64 = re.compile(r"^[0-9a-f]{64}$")
SIMHASH_BANDS = 4
SIMHASH_BAND_BITS = 16
ANCHOR_COUNT = 4
ANCHOR_STOPWORDS = frozenset(
    {
        "about",
        "after",
        "before",
        "following",
        "from",
        "most",
        "patient",
        "question",
        "that",
        "these",
        "this",
        "which",
        "with",
        "would",
    }
)


def normalise_text(value: str) -> str:
    """Return a stable comparison form without weakening source hashes."""

    value = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(re.findall(r"[a-z0-9]+", value))


def content_sha256(question: str, choices: dict[str, str]) -> str:
    canonical = json.dumps(
        {
            "question": str(question).strip(),
            "choices": {
                str(label).strip().upper(): str(text).strip()
                for label, text in choices.items()
            },
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _tokens(question: str) -> frozenset[str]:
    return frozenset(normalise_text(question).split())


def _simhash(tokens: frozenset[str]) -> int:
    if not tokens:
        return 0
    weights = [0] * 64
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        value = int.from_bytes(digest[:8], "big")
        for bit in range(64):
            weights[bit] += 1 if value & (1 << bit) else -1
    result = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            result |= 1 << bit
    return result


def _anchor_tokens(tokens: frozenset[str]) -> tuple[str, ...]:
    substantive = [
        token
        for token in tokens
        if len(token) >= 4 and token not in ANCHOR_STOPWORDS
    ]
    ranked = sorted(
        substantive,
        key=lambda token: hashlib.sha256(token.encode("utf-8")).digest(),
    )
    return tuple(ranked[:ANCHOR_COUNT])


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


@dataclass(frozen=True)
class OverlapResult:
    kind: str
    evaluation_id: str
    score: float


@dataclass(frozen=True)
class _EvaluationItem:
    record_id: str
    content_digest: str
    question_digest: str
    tokens: frozenset[str]
    simhash: int


class EvaluationIsolationIndex:
    """Index evaluation questions for exact and high-confidence near matches."""

    def __init__(self, records: Iterable[dict[str, Any]]) -> None:
        self._items: list[_EvaluationItem] = []
        self._content: dict[str, int] = {}
        self._questions: dict[str, int] = {}
        self._bands: dict[tuple[int, int], set[int]] = {}
        self._anchors: dict[str, set[int]] = {}
        for record in records:
            choices = _validated_choices(record.get("choices"))
            question = _required_text(record, "question")
            item = _EvaluationItem(
                record_id=str(record.get("id") or f"evaluation_{len(self._items)}"),
                content_digest=content_sha256(question, choices),
                question_digest=hashlib.sha256(
                    normalise_text(question).encode("utf-8")
                ).hexdigest(),
                tokens=_tokens(question),
                simhash=_simhash(_tokens(question)),
            )
            index = len(self._items)
            self._items.append(item)
            self._content.setdefault(item.content_digest, index)
            self._questions.setdefault(item.question_digest, index)
            for band in range(SIMHASH_BANDS):
                value = (
                    item.simhash >> (band * SIMHASH_BAND_BITS)
                ) & ((1 << SIMHASH_BAND_BITS) - 1)
                self._bands.setdefault((band, value), set()).add(index)
            for anchor in _anchor_tokens(item.tokens):
                self._anchors.setdefault(anchor, set()).add(index)

    def find_overlap(self, record: dict[str, Any]) -> OverlapResult | None:
        choices = _validated_choices(record.get("choices"))
        question = _required_text(record, "question")
        digest = content_sha256(question, choices)
        if digest in self._content:
            item = self._items[self._content[digest]]
            return OverlapResult("exact_content", item.record_id, 1.0)

        question_digest = hashlib.sha256(
            normalise_text(question).encode("utf-8")
        ).hexdigest()
        if question_digest in self._questions:
            item = self._items[self._questions[question_digest]]
            return OverlapResult("normalised_question", item.record_id, 1.0)

        tokens = _tokens(question)
        signature = _simhash(tokens)
        candidates: set[int] = set()
        for band in range(SIMHASH_BANDS):
            value = (
                signature >> (band * SIMHASH_BAND_BITS)
            ) & ((1 << SIMHASH_BAND_BITS) - 1)
            candidates.update(self._bands.get((band, value), ()))
        for anchor in _anchor_tokens(tokens):
            candidates.update(self._anchors.get(anchor, ()))
        for index in sorted(candidates):
            item = self._items[index]
            shorter = min(len(tokens), len(item.tokens))
            longer = max(len(tokens), len(item.tokens))
            if longer and shorter / longer < 0.90:
                continue
            score = _jaccard(tokens, item.tokens)
            if score >= 0.90:
                return OverlapResult("near_question", item.record_id, score)
        return None

    def assert_isolated(self, record: dict[str, Any]) -> None:
        overlap = self.find_overlap(record)
        if overlap is not None:
            raise ValueError(
                "training record overlaps evaluation data: "
                f"{overlap.kind} id={overlap.evaluation_id} score={overlap.score:.4f}"
            )


def _required_text(record: dict[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"record has missing or empty {field}")
    return value.strip()


def _validated_choices(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or len(value) < 2:
        raise ValueError("record choices must be a mapping with at least two options")
    choices = {
        str(label).strip().upper(): str(text).strip()
        for label, text in value.items()
    }
    if any(len(label) != 1 or not label.isalpha() for label in choices):
        raise ValueError("record has invalid choice labels")
    if any(not text for text in choices.values()):
        raise ValueError("record has an empty choice")
    return choices


def validate_train_record(
    record: dict[str, Any],
    evaluation_index: EvaluationIsolationIndex | None = None,
) -> None:
    """Reject any record that cannot prove train-only provenance."""

    required = {
        "id",
        "benchmark",
        "split",
        "question",
        "choices",
        "answer",
        "source_revision",
        "source_file_sha256",
        "content_sha256",
    }
    missing = required - set(record)
    if missing:
        raise ValueError(f"training record missing fields: {sorted(missing)}")
    if record["benchmark"] not in ALLOWED_BENCHMARKS:
        raise ValueError(f"unsupported training benchmark: {record['benchmark']!r}")
    if record["split"] != ALLOWED_SPLIT:
        raise ValueError(
            f"stage-2 data requires split='train'; got {record['split']!r}"
        )
    if not HEX_40.fullmatch(str(record["source_revision"])):
        raise ValueError("source_revision must be an exact 40-character commit SHA")
    if not HEX_64.fullmatch(str(record["source_file_sha256"])):
        raise ValueError("source_file_sha256 must be a 64-character SHA256")
    choices = _validated_choices(record["choices"])
    question = _required_text(record, "question")
    answer = str(record["answer"]).strip().upper()
    if answer not in choices:
        raise ValueError("answer is not one of the available choices")
    expected_digest = content_sha256(question, choices)
    if record["content_sha256"] != expected_digest:
        raise ValueError("content_sha256 does not match question and choices")
    if evaluation_index is not None:
        evaluation_index.assert_isolated(record)
