# Step 2: three-case PRM validator HTTP 429 recovery

Approval: user-approved on 2026-08-27. This runbook covers only the three
validator events that were unavailable because OpenRouter returned HTTP 429
without content. It does not authorize new candidates, local GPU generation,
training-data merge, or larger-scale generation.

## Frozen limits

- source canary: 24 candidates, 21 strict-contract results, 3 unavailable;
- selection: exactly the 3 unavailable events, all with `http_429` and no content;
- validator: the same `deepseek/deepseek-v4-pro` validator v2 contract;
- maximum two requests per candidate, at most six new requests;
- delays: 60 seconds initially, 45 seconds between candidates, 90 seconds before
  retrying an HTTP 429;
- new API hard cap CNY 2 equivalent; stop-before line CNY 1.8;
- append-only recovery output under the private canary directory.

The original canary and pilot files are hashed and must not change.

## 1. Synchronize and test

```bash
cd /root/autodl-tmp/MedTrace-R1
source /etc/network_turbo
source .venv/bin/activate

git status --short
git fetch origin \
  +refs/heads/step2_genCOTdata:refs/remotes/origin/step2_genCOTdata
git merge --ff-only origin/step2_genCOTdata
git rev-parse HEAD
git status --short

python -m unittest discover
```

Do not use `reset --hard` and do not delete any private canary output if the
merge cannot fast-forward.

## 2. Preview and zero-completion preflight

Load the OpenRouter key in the shell that will execute recovery:

```bash
read -rsp 'OpenRouter API key: ' OPENROUTER_API_KEY && echo
export OPENROUTER_API_KEY
```

Preview reads only local files:

```bash
python -m data_pipeline.run_prm_negative_validator_recovery \
  --config configs/cot/prm_negative_validator_recovery_v1.yaml
```

Expected preview selection:

```text
selected_http_429_events: 3
max_attempts_per_candidate: 2
maximum_new_requests: 6
api_hard_cap_cny_equivalent: 2
budget_stop_limit_cny: 1.8
```

Preflight checks the clean Git state, frozen source failure signatures, source
hashes, API key, and model availability. It makes no completion request:

```bash
python -m data_pipeline.run_prm_negative_validator_recovery \
  --config configs/cot/prm_negative_validator_recovery_v1.yaml \
  --preflight-only
```

Continue only when `status` is `passed` and `selected_events` is 3.

## 3. Execute or resume recovery

Use a tmux session so the throttle waits and validator requests survive an SSH
disconnect:

```bash
tmux new -s prm-negative-recovery-3
```

Inside tmux:

```bash
cd /root/autodl-tmp/MedTrace-R1
source .venv/bin/activate

read -rsp 'OpenRouter API key: ' OPENROUTER_API_KEY && echo
export OPENROUTER_API_KEY

set -o pipefail
mkdir -p \
  results/cot/pilot_v1_real/prm_negative_enrichment_v1/canary_v1/validator_recovery_v1

python -m data_pipeline.run_prm_negative_validator_recovery \
  --config configs/cot/prm_negative_validator_recovery_v1.yaml \
  --execute-recovery-3 \
  2>&1 | tee \
  results/cot/pilot_v1_real/prm_negative_enrichment_v1/canary_v1/validator_recovery_v1/runner.log
```

The runner waits before the first request. If interrupted, load the key again
and run the same command. Successful candidates are never requested again, and
the per-candidate attempt count cannot exceed two. Never delete an event log or
manifest to bypass a source, attempt, or budget gate.

## 4. Canonical aggregate audit

After the runner finishes:

```bash
python -m data_pipeline.audit_prm_negative_validator_recovery \
  --config configs/cot/prm_negative_validator_recovery_v1.yaml
```

The target machine gate is:

```text
Canonical strict contract valid: 24 / 24
Integrity passed: true
Machine quality passed: true
```

The audit uses original complete results unchanged and substitutes only a
strict-contract-valid recovery for each original unavailable result. It never
merges records into SFT or PRM training data.

## 5. Structured human review

The recovery passed with 24/24 canonical strict contracts and 12 strict process
negatives. Complete and hash-lock the approved structured blind review before
generating or opening the recovered key. Follow:

```text
docs/STEP2_PRM_NEGATIVE_HUMAN_REVIEW_RUNBOOK.md
```

The key command now fails closed unless the completed annotation file has a
valid immutable lock. It never overwrites `human_review_blind.md`.
