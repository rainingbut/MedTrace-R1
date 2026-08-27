# Step 2: 24-case PRM negative canary runbook

Approval: user-approved on 2026-08-26. No second approval is required for the
commands in this runbook. This approval covers exactly the frozen 24-case
canary; it does not authorize training-data merge or full-scale generation.

## Frozen composition and limits

- 8 existing answer-mismatch candidates: 4 MedQA + 4 MedMCQA;
- 8 local gold-blind student candidates: 4 + 4;
- 8 local controlled single-error candidates: 4 + 4;
- local model: `Qwen/Qwen2.5-7B-Instruct` at revision `a09a...bc28`;
- validator: OpenRouter `deepseek/deepseek-v4-pro`, validator v2 strict schema;
- OpenRouter hard cap CNY 20 equivalent; stop-before line CNY 18;
- local GPU cap one hour;
- all outputs remain private under
  `results/cot/pilot_v1_real/prm_negative_enrichment_v1/`.

The original seven pilot artifacts are hashed before and after execution and
must not change.

## 1. Synchronize and test

```bash
cd /root/autodl-tmp/MedTrace-R1
source .venv/bin/activate
source /etc/network_turbo

git status --short
git fetch origin \
  +refs/heads/step2_genCOTdata:refs/remotes/origin/step2_genCOTdata
git merge --ff-only origin/step2_genCOTdata
git rev-parse HEAD
git status --short

python -m unittest discover
```

Do not use `reset --hard` or delete any ignored result directory if the merge
cannot fast-forward.

## 2. Build the strict, isolated pilot view

This excludes the one legacy boolean-label canonical. It does not modify the
original pilot.

```bash
python -m data_pipeline.build_strict_pilot_view \
  --config configs/cot/prm_negative_canary_v1.yaml
```

Expected counts:

```text
canonical_trajectories: 108
sft_records: 107
prm_records: 721
excluded validator_contract_invalid: 1
```

## 3. Start the pinned local vLLM service

```bash
tmux new -s prm-negative-vllm
cd /root/autodl-tmp/MedTrace-R1
source .venv/bin/activate

export VLLM_RUNTIME_BACKEND=native
export HF_HOME=/root/autodl-tmp/medtrace-cache/huggingface
export MEDTRACE_CACHE_DIR="${HF_HOME}"

./scripts/serve_vllm.sh \
  2>&1 | tee results/prm-negative-vllm.log
```

Detach with `Ctrl+B`, then `D`. After `/v1/models` responds, capture the exact
runtime from another terminal:

```bash
cd /root/autodl-tmp/MedTrace-R1
source .venv/bin/activate

python scripts/capture_runtime.py \
  --backend native \
  --vllm-version 0.24.0 \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --model-revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --output results/runtime/cot_screener_runtime_manifest.json
```

Do not use `--allow-dirty`.

## 4. Load the validator key privately

In the shell that will run the canary:

```bash
read -rsp 'OpenRouter API key: ' OPENROUTER_API_KEY && echo
export OPENROUTER_API_KEY
export MEDTRACE_API_KEY=EMPTY
```

No DashScope key is needed. Do not write keys to config, logs, scripts, or Git.
If execution is moved into a new tmux session, load the key again inside that
session; an already-running tmux server may not inherit a newly exported custom
environment variable.

## 5. Preview and zero-completion preflight

Preview makes no model request:

```bash
python -m data_pipeline.run_prm_negative_canary \
  --config configs/cot/prm_negative_canary_v1.yaml
```

Preflight checks GPU, Git/runtime pins, both `/models` endpoints, keys, strict
source view, and the immutable source identity. It makes no completion request:

```bash
python -m data_pipeline.run_prm_negative_canary \
  --config configs/cot/prm_negative_canary_v1.yaml \
  --preflight-only
```

Continue only when the preflight status is `passed`.

## 6. Execute or resume the approved canary

```bash
tmux new -s prm-negative-24
cd /root/autodl-tmp/MedTrace-R1
source .venv/bin/activate

read -rsp 'OpenRouter API key: ' OPENROUTER_API_KEY && echo
export OPENROUTER_API_KEY
export MEDTRACE_API_KEY=EMPTY
set -o pipefail
mkdir -p \
  results/cot/pilot_v1_real/prm_negative_enrichment_v1/canary_v1

python -m data_pipeline.run_prm_negative_canary \
  --config configs/cot/prm_negative_canary_v1.yaml \
  --execute-canary-24 \
  2>&1 | tee \
  results/cot/pilot_v1_real/prm_negative_enrichment_v1/canary_v1/runner.log
```

If interrupted, restore the two environment variables and run the same command.
Append-only local-generation and validator event logs provide resume state.
Never delete an event log to bypass a manifest or hash error.

All 24 candidates are constructed and their exact origin/benchmark mix is
verified before the first OpenRouter completion is requested.
The first execution persists a one-hour local-generation deadline. Interrupting
and resuming the runner does not reset that deadline; completed candidates can
still proceed to validator calls after local generation is complete.

## 7. Aggregate audit and private blind review

```bash
python -m data_pipeline.audit_prm_negative_canary \
  --config configs/cot/prm_negative_canary_v1.yaml

python -m data_pipeline.review_prm_negative_canary \
  --config configs/cot/prm_negative_canary_v1.yaml
```

Aggregate outputs:

```text
results/cot/pilot_v1_real/prm_negative_enrichment_v1/canary_v1/quality_audit.json
results/cot/pilot_v1_real/prm_negative_enrichment_v1/canary_v1/quality_audit.md
```

Private review outputs:

```text
results/cot/pilot_v1_real/prm_negative_enrichment_v1/canary_v1/human_review_blind.md
results/cot/pilot_v1_real/prm_negative_enrichment_v1/canary_v1/human_review_key.md
```

Complete the blind file before opening the key. Both files contain private
licensed question/trajectory material and must not be committed or shared.

The canary does not automatically merge any record into SFT or PRM training
data, even if machine gates pass.

## 8. Recorded result and approved HTTP 429 recovery

The completed canary produced 21 strict-contract results, 11 strict process
negatives, and 3 unavailable responses. All three unavailable responses were
OpenRouter HTTP 429 failures with no content. The approved isolated recovery
commands are in `docs/STEP2_PRM_NEGATIVE_VALIDATOR_RECOVERY_RUNBOOK.md`.
All three recovery requests later succeeded on their first attempt. The
canonical audit reached 24/24 strict contracts and 12 strict process negatives;
the remaining required gate is the structured blind human review documented in
`docs/STEP2_PRM_NEGATIVE_HUMAN_REVIEW_RUNBOOK.md`.
