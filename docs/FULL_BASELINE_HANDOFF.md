# MEDTRACE-R1 full-baseline handoff

## Purpose and current stopping point

This document hands off the first official Qwen2.5-7B-Instruct baseline from
the completed 200-record pilot to the full 5,456-record evaluation.

The full run has **not started**. The existing pilot contains 100 MedQA and 100
MedMCQA predictions and must be resumed rather than rerun when the same AutoDL
instance and vLLM process are still alive.

## Frozen repository and runtime

| Item | Frozen value |
|---|---|
| Branch | `baseevalframe` |
| Git commit | `8459db5504723a8ac59f152d376582597c8aa288` |
| Provider | AutoDL container instance |
| AutoDL container UUID | `smyulw5z7c-d584d27f` |
| GPU | NVIDIA GeForce RTX 4090, 24,564 MiB |
| Driver / advertised CUDA | 580.105.08 / 13.0 |
| Python | 3.12.13 |
| PyTorch | 2.11.0+cu130 |
| Transformers | 5.14.1 |
| vLLM | 0.24.0 |
| Model | `Qwen/Qwen2.5-7B-Instruct` |
| Model/tokenizer revision | `a09a35458c702b33eeacc393d103063234e8bc28` |
| Precision | BF16, no quantisation |
| Maximum context | 4096 |
| Maximum concurrent sequences | 8 |
| Git worktree at pilot capture | clean |

Repository location on AutoDL:

```text
/root/autodl-tmp/MedTrace-R1
```

Native environment and model cache:

```text
/root/autodl-tmp/MedTrace-R1/.venv
/root/autodl-tmp/medtrace-cache/huggingface
```

## Completed gates

- `uv pip check`: all 193 installed packages compatible.
- CUDA available and BF16 supported on the RTX 4090.
- Linux Bash syntax checks passed.
- 31/31 unit tests passed.
- Pinned Parquet provenance verification passed for every benchmark row.
- Prepared benchmark contains 5,456 unique records:
  - MedQA test: 1,273;
  - MedMCQA validation: 4,183.
- Combined input SHA256:
  `2667539b38e43c538da784803a8b19b3ef9309fd65a26ffe65944d511977b6df`.
- vLLM `/version` and `/v1/models` health checks passed.
- Pilot completed without API errors, truncation, OOM, traceback, or CUDA error.

## Pilot result

| Metric | MedQA (100) | MedMCQA (100) | Overall (200) |
|---|---:|---:|---:|
| Accuracy | 0.69 | 0.51 | 0.60 |
| Parse rate | 0.99 | 1.00 | 0.995 |
| Format rate | 0.99 | 1.00 | 0.995 |
| Truncation rate | 0.00 | 0.00 | 0.00 |
| API errors | 0 | 0 | 0 |

Pilot wall time was 54.546752 seconds. At an AutoDL rate of CNY 2.5/hour,
the measured projection was:

```text
projected_compute_hours: 0.413
projected_hours_with_safety_factor: 0.517
projected_compute_cost: CNY 1.03
budget_with_safety_factor: CNY 1.29
recommended_booking_hours: 2
```

These 100-record per-benchmark accuracies are operational pilot observations,
not reportable final baseline values.

### Audited format failure

One MedQA record (`medqa_test_000072_c998d6d8b6`) ended with an option letter
plus option text instead of exactly `Final Answer: X`. Strict rejection was
correct. The selected letter also differed from the benchmark answer, so
extracting it would not change accuracy. Do not loosen the parser or tune the
prompt based on this held-out record.

## Non-negotiable controls

- Do not change the model, revision, prompt, parser, decoding, concurrency, or
  benchmark ordering.
- Do not rerun the 200 pilot records from scratch when a valid same-runtime
  resume is available.
- Do not create a second full-run directory accidentally.
- Do not mix predictions across a changed instance, vLLM process, dependency
  environment, Git commit, config, input, or runtime manifest.
- Do not publish raw benchmark prompts or predictions until the recorded
  licensing questions are resolved.

## Resume decision gate

A new chat/session alone does not change the experiment. Resume the existing
pilot only if the same AutoDL instance and original vLLM process are still
running.

Run:

```bash
cd /root/autodl-tmp/MedTrace-R1
source .venv/bin/activate

export VLLM_RUNTIME_BACKEND=native
export HF_HOME=/root/autodl-tmp/medtrace-cache/huggingface
export MEDTRACE_CACHE_DIR="${HF_HOME}"

git rev-parse HEAD
git status --short
tmux ls
curl -s http://127.0.0.1:8000/version | python -m json.tool
curl -s http://127.0.0.1:8000/v1/models | python -m json.tool

RUN_DIR="$(ls -dt results/baseline/*/ | head -n 1)"
echo "${RUN_DIR}"
wc -l "${RUN_DIR}/predictions.jsonl"
```

Required output:

- Git commit is `8459db5504723a8ac59f152d376582597c8aa288`;
- `git status --short` is empty;
- `medtrace-vllm` exists in `tmux ls`;
- API reports vLLM 0.24.0 and the frozen model;
- the selected pilot predictions file has exactly 200 lines.

Also verify that the live server PID is the captured PID when the process was
not restarted:

```bash
python - "${RUN_DIR}/metadata.json" <<'PY'
import json
from pathlib import Path
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    metadata = json.load(handle)

manifest = metadata["runtime_manifest"]
pid = int(manifest["process"]["pid"])
pid_path = Path(f"/proc/{pid}")
print("captured_pid:", pid)
print("pid_alive:", pid_path.exists())
print("runtime_manifest_sha256:", metadata["runtime_manifest_sha256"])
assert manifest["git_commit"] == "8459db5504723a8ac59f152d376582597c8aa288"
assert manifest["model_revision"] == "a09a35458c702b33eeacc393d103063234e8bc28"
assert manifest["packages"]["vllm"] == "0.24.0"
assert pid_path.exists(), "captured vLLM process is no longer alive"
command = (pid_path / "cmdline").read_bytes().replace(b"\0", b" ").decode()
print("live_command:", command)
assert "vllm serve Qwen/Qwen2.5-7B-Instruct" in command
assert "a09a35458c702b33eeacc393d103063234e8bc28" in command
PY
```

If the instance or vLLM process was restarted, stop and ask for a new-run plan.
Do not silently resume the old pilot with a new runtime.

## Start the full resume

After the decision gate passes, create a second tmux session:

```bash
tmux new -s medtrace-full
```

Inside it:

```bash
cd /root/autodl-tmp/MedTrace-R1
source .venv/bin/activate

export VLLM_RUNTIME_BACKEND=native
export HF_HOME=/root/autodl-tmp/medtrace-cache/huggingface
export MEDTRACE_CACHE_DIR="${HF_HOME}"

RUN_DIR="$(ls -dt results/baseline/*/ | head -n 1)"
echo "Resuming: ${RUN_DIR}"

./scripts/resume_full_baseline.sh "${RUN_DIR}" \
  2>&1 | tee results/full-baseline-console.log
```

Detach with `Ctrl+B`, then `D`. The evaluator should recognise 200 completed
records and append the remaining 5,256.

## Monitor

In another terminal:

```bash
cd /root/autodl-tmp/MedTrace-R1
RUN_DIR="$(ls -dt results/baseline/*/ | head -n 1)"

watch -n 10 "
date
wc -l '${RUN_DIR}/predictions.jsonl'
nvidia-smi --query-gpu=temperature.gpu,power.draw,memory.used,utilization.gpu --format=csv,noheader
tail -n 8 results/full-baseline-console.log
"
```

Check server errors separately:

```bash
grep -Ei "out of memory|oom|traceback|fatal|cuda error" \
  results/vllm-server.log || true
```

If only the evaluation process stops while the original vLLM process remains
healthy, rerun the same resume command with `tee -a`. The evaluator will reuse
completed predictions after validating its hashes. If the vLLM process or
runtime changes, stop and create a new-run plan instead.

## Final acceptance

After the resume exits successfully:

```bash
RUN_DIR="$(ls -dt results/baseline/*/ | head -n 1)"

wc -l "${RUN_DIR}/predictions.jsonl"
cat "${RUN_DIR}/metrics.json"
cat "${RUN_DIR}/metadata.json"
```

Verify exact counts and unique IDs:

```bash
python - "${RUN_DIR}/predictions.jsonl" <<'PY'
import json
import sys
from collections import Counter

with open(sys.argv[1], encoding="utf-8") as handle:
    rows = [json.loads(line) for line in handle if line.strip()]

ids = [row["id"] for row in rows]
benchmarks = Counter(row["benchmark"] for row in rows)
duplicates = [item for item, count in Counter(ids).items() if count > 1]

print("total:", len(rows))
print("benchmarks:", dict(benchmarks))
print("unique_ids:", len(set(ids)))
print("duplicate_ids:", duplicates[:20])

assert len(rows) == 5456
assert benchmarks == {"medqa": 1273, "medmcqa": 4183}
assert len(set(ids)) == 5456
PY
```

Final error check:

```bash
grep -Ei "out of memory|oom|traceback|fatal|cuda error" \
  results/vllm-server.log \
  results/full-baseline-console.log || true
```

Do not stop the server until the final metrics, metadata, count check, and error
check have been reviewed.

## Private backup and shutdown

After final acceptance, create a private archive outside the repository:

```bash
RUN_DIR="$(ls -dt results/baseline/*/ | head -n 1)"
ARCHIVE="/root/autodl-tmp/medtrace-baseline-8459db5-$(date -u +%Y%m%dT%H%M%SZ).tar.gz"

tar -czf "${ARCHIVE}" \
  "${RUN_DIR}" \
  results/runtime/runtime_manifest.json \
  results/vllm-server.log \
  results/full-baseline-console.log \
  data/benchmark/manifest.json \
  data/benchmark/provenance.json

sha256sum "${ARCHIVE}"
ls -lh "${ARCHIVE}"
```

Download the archive privately. Do not commit raw predictions. Once the backup
is confirmed:

```bash
./scripts/stop_vllm.sh
nvidia-smi
tmux kill-session -t medtrace-vllm 2>/dev/null || true
tmux kill-session -t medtrace-full 2>/dev/null || true
```

Then shut down or release the AutoDL instance as appropriate. Paid expanded
storage may continue billing while the instance is powered off.

## Prompt for the next assistant session

Copy the following into the new session:

```text
Continue the MEDTRACE-R1 first official full baseline from the handoff document
docs/FULL_BASELINE_HANDOFF.md. Read that document completely before advising or
taking action. The 200-record pilot is complete, but the 5,456-record full run
has not started. First walk me through the resume decision gate and verify that
the original AutoDL instance and captured vLLM process are still alive. Do not
change the model, revision, prompt, parser, decoding configuration, benchmark
data, runtime manifest, or existing RUN_DIR. Wait for my confirmation before
each new execution phase.
```
