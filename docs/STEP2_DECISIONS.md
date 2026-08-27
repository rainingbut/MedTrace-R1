# Step 2 frozen decisions

Status: the frozen 40-question paid pilot was approved on 2026-08-11. This
approval covers that pilot only; it does not cover larger-scale generation.

## Data boundary

- Only the pinned `train` splits of MedQA and MedMCQA may enter generation,
  screening, validation, SFT, or PRM derivation.
- Evaluation splits, `evaluation/data/eval_data.json`, and legacy
  `data/demo_data.json` are forbidden as training sources.
- Exact, normalised, or high-confidence near overlap with evaluation data is
  rejected before sampling.
- Questions, raw responses, canonical trajectories, derived SFT/PRM JSONL, and
  per-record verification outputs remain private and Git-ignored.
- Current permission boundary is private, non-commercial research. Publication,
  redistribution, or commercial use requires a separate rights review.

## Models and prompts

| Role | Frozen model | Prompt |
|---|---|---|
| Teacher | `qwen3-max-2026-01-23` | `teacher_v1`, gold-blind |
| First-pass screener | `Qwen/Qwen2.5-7B-Instruct` at `a09a35458c702b33eeacc393d103063234e8bc28` | `screener_v1` |
| Independent verifier | OpenRouter `deepseek/deepseek-v4-pro` | `validator_v1` |

The teacher produces one candidate per request. Four independent requests are
made for each question. The screener is conservative, and its verdict is hidden
from the independent verifier.

The verifier routes through OpenRouter using the user's existing balance. Each
request requires ZDR routing, denies data-collecting providers, requires support
for the supplied parameters, and keeps provider fallbacks within the same exact
model slug. The teacher remains on DashScope because the frozen dated Qwen model
is not replaced by an unverified OpenRouter alias.

OpenRouter routing is sorted by price and capped per request at USD 2.10/M input
tokens and USD 4.40/M output tokens. The runner reserves budget at those maximum
rates before every validator request, then replaces the estimate with the
provider-reported `usage.cost` for its durable ledger. A missing cost field is a
failed response, not permission to fall back to an estimated successful charge.

## Pilot

- 40 questions: 20 MedQA train and 20 MedMCQA train.
- Four candidates per question, for 160 candidate trajectories before filters.
- Fixed sampling seed `20260811`; stratify by question length and, for MedMCQA,
  subject where metadata permits.
- API hard cap: CNY 10 equivalent. Local screening cap: one RTX 4090 GPU hour.
- The committed executable flag remains false as a fail-safe. The approved run
  must still pass preflight and use the explicit `--execute-40` CLI flag.

Machine-readable settings are in `configs/cot/pilot_v1.yaml`; the canonical
record contract is `schemas/cot_trajectory_v1.schema.json`, and the private
training-source manifest contract is
`schemas/training_source_manifest_v1.schema.json`.

## Post-pilot PRM amendment (2026-08-17)

The original frozen pilot remains immutable. Its validator recovery fixed the
transport/JSON contract failure mode, but the observed PRM labels are severely
imbalanced at approximately 723 positive and 4 negative prefix records. Six
validator v2 recovery canaries were contract-valid and all positive; recovery
therefore is not treated as a negative-data strategy.

The user approved only the offline first phase of PRM negative enrichment:

- audit structurally usable answer-mismatch and screener-reject candidates;
- assign no automatic negative label to either source;
- define a private candidate contract with no label field;
- accept a strict process negative only when an independent validator returns
  `problem_status=ok`, a concrete first error, and `local_verdict=incorrect` at
  that step;
- route uncertain, ambiguous, and bad-gold outcomes to human review;
- exclude answer-only inconsistency from PRM step negatives;
- keep all original pilot artifacts immutable and all private records ignored.

This approval does not cover GPU inference, API calls, the remaining validator
recovery, a 24-case canary, medium-scale generation, or full-data generation.
The detailed policy is in `docs/STEP2_PRM_NEGATIVE_ENRICHMENT_PLAN.md` and the
machine-readable settings are in
`configs/cot/prm_negative_enrichment_v1.yaml`.

The first aggregate opportunity audit found 11 structurally usable natural
candidates (7 MedQA and 4 MedMCQA), all from answer mismatch and none from a
screener reject. It also found one legacy contract-invalid canonical whose six
derived PRM labels are JSON booleans rather than integer ones. Those records are
not strict training artifacts. The original files remain immutable; any
normalization or validator v2 revalidation requires an isolated, auditable
derivative and a later decision.

## PRM negative canary approval (2026-08-26)

The user approved execution without a second approval gate for exactly 24
candidates: eight existing answer-mismatch, eight local gold-blind student, and
eight local controlled-single-error candidates. Each origin contains four
MedQA and four MedMCQA cases. The OpenRouter hard cap is CNY 20 equivalent with
a CNY 18 stop line; local GPU time is capped at one hour. This approval does not
authorize merging canary labels into training data or larger-scale generation.

## PRM negative validator HTTP 429 recovery approval (2026-08-27)

The 24-case canary produced 21 strict-contract results, 11 strict process
negatives, and three unavailable responses. All three unavailable responses
were OpenRouter HTTP 429 failures without content. The user approved recovery
of exactly those three events with the same validator v2 contract, at most two
requests per event, 60/45/90-second throttling, a CNY 2 hard cap, and a CNY 1.8
stop line. Recovery must not overwrite source events, generate candidates, use
local GPU inference, merge training data, or expand scale.

## Structured PRM human-review approval (2026-08-27)

After the recovery, the canonical canary passed machine quality with 24/24
strict contracts and 12 strict process negatives. The user approved an offline
structured blind-review phase: a private 24-case annotation template, strict
conditional field contracts, an immutable annotation/source/recovery hash lock,
and aggregate scoring at 90% trajectory-label and 80% exact-first-error gates.
Only a conservative candidate metadata list may be written after both gates
pass. No model/API call, training record, merge, or scale expansion is allowed.
