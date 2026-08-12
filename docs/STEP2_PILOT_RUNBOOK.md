# Step 2: 40-question CoT pilot runbook

This runbook executes the approved paid pilot only. It does not authorize the
later full-data generation stage.

## Frozen run

- Input: 40 private train questions, 20 MedQA and 20 MedMCQA.
- Sampling: four independent teacher requests per question (160 candidates).
- Teacher: `qwen3-max-2026-01-23`, gold-blind `teacher_v1`.
- Local screener: `Qwen/Qwen2.5-7B-Instruct` at revision
  `a09a35458c702b33eeacc393d103063234e8bc28`, `screener_v1`.
- Independent verifier: OpenRouter `deepseek/deepseek-v4-pro`, `validator_v1`,
  with per-request zero-data-retention routing.
- API hard cap: CNY 10 equivalent; the runner stops before CNY 9.
- Private outputs: `data/source/` and `results/cot/` are Git-ignored.

Do not substitute a model, revision, split, prompt version, or output directory
when resuming this run.

## 1. Prepare the AutoDL host

Use the accepted RTX 4090 AutoDL instance. The Windows development host's RTX
3050 (4 GB) cannot serve the frozen 7B screener.

```bash
cd /root/autodl-tmp/MedTrace-R1
source .venv/bin/activate
source /etc/network_turbo

export VLLM_RUNTIME_BACKEND=native
export HF_HOME=/root/autodl-tmp/medtrace-cache/huggingface
export MEDTRACE_CACHE_DIR="${HF_HOME}"

git rev-parse HEAD
git status --short
python -m unittest discover
nvidia-smi
df -h / /root/autodl-tmp
```

The code snapshot used for the pilot must be clean and must contain the Step 2
runner. Do not use `--allow-dirty` for the official pilot runtime manifest.

## 2. Rebuild and verify private inputs on AutoDL

Private data is deliberately absent from Git. Either copy the already verified
`data/source/` directory through a private channel, or deterministically rebuild
it on AutoDL:

```bash
python -m data_pipeline.prepare_training_sources
python -m data_pipeline.verify_training_sources
python -m data_pipeline.select_cot_pilot

sha256sum data/source/pilot/pilot_v1_questions.jsonl
```

The question-file SHA-256 must be:

```text
eb7f4c2ba71d4651929af89558bfa64c295dcdab902e218483f931c040df9f76
```

Selection also binds the current `configs/cot/pilot_v1.yaml` hash into the
private pilot manifest. The real runner rejects a stale manifest.

## 3. Start the pinned local screener

In a dedicated tmux session:

```bash
tmux new -s cot-screener
cd /root/autodl-tmp/MedTrace-R1
source .venv/bin/activate
export VLLM_RUNTIME_BACKEND=native
export HF_HOME=/root/autodl-tmp/medtrace-cache/huggingface
export MEDTRACE_CACHE_DIR="${HF_HOME}"
./scripts/serve_vllm.sh 2>&1 | tee results/cot-screener-vllm.log
```

Detach with `Ctrl+B`, then `D`. In a second shell, after `/v1/models` responds,
capture the exact running process:

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

## 4. Load keys without printing them

Run this in the shell that will launch the pilot:

```bash
read -rsp 'DashScope API key: ' DASHSCOPE_API_KEY && echo
export DASHSCOPE_API_KEY
read -rsp 'OpenRouter API key: ' OPENROUTER_API_KEY && echo
export OPENROUTER_API_KEY
export MEDTRACE_API_KEY=EMPTY
```

Do not put keys in YAML, logs, shell scripts, Git, or the runtime manifest.

## 5. Zero-generation preflight

This checks the private manifest and its hashes, key presence, GPU capacity,
runtime pin, vLLM model endpoint, and both paid-provider model endpoints. It does
not request a completion.

```bash
python -m data_pipeline.run_cot_pilot_real --preflight-only
```

Continue only if the JSON result says `"status": "passed"` and names all three
frozen models.

## 6. Execute or resume the approved 40-question pilot

Use another named tmux session:

```bash
tmux new -s cot-pilot-40
cd /root/autodl-tmp/MedTrace-R1
source .venv/bin/activate
export DASHSCOPE_API_KEY
export OPENROUTER_API_KEY
export MEDTRACE_API_KEY=EMPTY
mkdir -p results/cot/pilot_v1_real
set -o pipefail
python -m data_pipeline.run_cot_pilot_real --execute-40 \
  2>&1 | tee results/cot/pilot_v1_real/runner.log
```

If interrupted, restore the three environment variables and run the same
command. The runner resumes from append-only per-phase event logs and refuses a
changed config or question file. Never delete an event log to bypass a resume
error.

The run is complete only when
`results/cot/pilot_v1_real/metadata.json` has `"status": "complete"`. Inspect:

```bash
python -m json.tool results/cot/pilot_v1_real/metadata.json
wc -l \
  results/cot/pilot_v1_real/teacher_events.jsonl \
  results/cot/pilot_v1_real/screener_events.jsonl \
  results/cot/pilot_v1_real/validator_events.jsonl \
  results/cot/pilot_v1_real/canonical_trajectories.jsonl \
  results/cot/pilot_v1_real/sft_verified.jsonl \
  results/cot/pilot_v1_real/process_train.jsonl
```

Stop after this pilot and review its quality/cost report before any larger CoT
generation stage.
