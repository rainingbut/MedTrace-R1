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
