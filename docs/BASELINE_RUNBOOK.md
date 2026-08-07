# Qwen2.5-7B-Instruct baseline runbook

This runbook produces the first official MEDTRACE-R1 baseline. It uses BF16
weights without quantisation and evaluates the pinned model revision on MedQA
test and MedMCQA validation.

## Frozen components

| Component | Frozen value |
|---|---|
| Model | `Qwen/Qwen2.5-7B-Instruct` |
| Model/tokenizer revision | `a09a35458c702b33eeacc393d103063234e8bc28` |
| Model license | Apache-2.0 |
| vLLM | `0.24.0` |
| Docker manifest digest | `sha256:251eba5cc7c12fed0b75da22a9240e582b1c9e39f6fbc064f86781b963bd814f` |
| Precision | BF16, no quantisation |
| Maximum context | 4096 tokens |
| Maximum concurrent sequences | 8 |
| Generation | temperature 0, top-p 1.0, maximum 1024 new tokens, seed 42 |

Docker hosts use the frozen official image. AutoDL container instances use the
native Python environment from `requirements.txt`, because AutoDL does not
support Docker inside a container instance. `scripts/capture_runtime.py`
records the selected backend, actual package versions, GPU, driver, Git commit,
and either the Docker digest or the native requirements hash and `pip freeze`.

## Recommended rental

For the current inference-only baseline, rent **one RTX 4090 24 GB**. The model
weights occupy roughly 15 GB in BF16; the frozen 4096-token context and
eight-sequence cap leave the remaining VRAM for KV cache and runtime overhead.
The 200-question pilot is still an OOM gate.

- Preferred cost/performance: one RTX 4090 24 GB.
- Safer fallback after an OOM: one L40S 48 GB or A100 40/80 GB.
- Avoid cards below 24 GB and avoid quantisation for the official baseline.
- CPU/RAM: at least 8 vCPU and 32 GB host RAM.
- AutoDL system disk: expand the default 30 GB system disk to at least 60 GB.
- AutoDL data disk: at least 50 GB; use `/root/autodl-tmp` for the repository,
  Hugging Face cache, and results. Expand to 80 GB if keeping multiple models.

This recommendation applies only to baseline inference. SFT and GRPO/DAPO
will receive separate memory plans; do not assume one 4090 can run the final RL
pipeline.

Do not use a quantised model for the official BF16 baseline.

## 1. Prepare an AutoDL container instance

Select an Ubuntu/Miniconda image, Python 3.12, and a host whose NVIDIA driver
supports the CUDA backend selected by vLLM. AutoDL instances are containers and
cannot run Docker internally, so use the native backend:

```bash
cd /root/autodl-tmp
git clone https://github.com/rainingbut/MedTrace-R1.git
cd MedTrace-R1
git checkout <the-AutoDL-baseline-commit>
test -z "$(git status --porcelain)"

python -m pip install --upgrade uv
uv venv --python 3.12 --seed --managed-python .venv
source .venv/bin/activate
uv pip install -r requirements.txt --torch-backend=auto

export VLLM_RUNTIME_BACKEND=native
export HF_HOME=/root/autodl-tmp/medtrace-cache/huggingface
export MEDTRACE_CACHE_DIR="${HF_HOME}"

python - <<'PY'
import torch, transformers, vllm
print("torch", torch.__version__, "CUDA", torch.version.cuda)
print("transformers", transformers.__version__)
print("vllm", vllm.__version__)
assert vllm.__version__ == "0.24.0"
assert torch.cuda.is_available()
assert torch.cuda.is_bf16_supported()
PY
nvidia-smi

bash -n scripts/*.sh

python data_pipeline/verify_benchmark_provenance.py
python data_pipeline/prepare_benchmarks.py
```

The provenance check uses the network and downloads the two pinned benchmark
Parquet splits into `.cache/provenance`; it downloads no model weights and does
not modify the committed provenance report.

To keep the backend and cache variables in new terminals, either export them
again or add those two exports to the instance's shell startup file.

## 2. Start the pinned server

### AutoDL native backend

In a persistent `tmux` session:

```bash
cd /root/autodl-tmp/MedTrace-R1
source .venv/bin/activate
export VLLM_RUNTIME_BACKEND=native
export MEDTRACE_CACHE_DIR=/root/autodl-tmp/medtrace-cache/huggingface
./scripts/serve_vllm.sh 2>&1 | tee results/vllm-server.log
```

The first start downloads approximately 15.2 GB of model files to the AutoDL
data disk.

### Docker-capable host

Inspect the runtime constants, then pull the immutable image:

```bash
source configs/runtime/baseline_vllm.sh
docker pull "${VLLM_IMAGE}"
```

Start the server in a persistent `tmux` session:

```bash
./scripts/serve_vllm.sh 2>&1 | tee results/vllm-server.log
```

The first start downloads approximately 15.2 GB of model files. Keep the cache
directory on persistent storage if the provider separates system and data
disks.

## 3. Run the stratified 200-question pilot

In a second terminal:

```bash
source .venv/bin/activate
./scripts/run_baseline_pilot.sh
```

The pilot evaluates 100 MedQA and 100 MedMCQA records. These predictions are
the first 200 records of the official run and will be reused by the resume
command. Do not tune prompts, decoding parameters, or reward weights based on
pilot accuracy.

Operational acceptance gates:

- API error count is zero;
- no OOM or server restart occurred;
- truncation rate is zero, or is investigated before continuing;
- the runtime manifest reports the expected model revision, image, vLLM
  version, BF16-capable GPU, and a clean Git commit;
- `predictions.jsonl`, `metadata.json`, and `metrics.json` are present.

Parse and format rates describe model behaviour. If they reveal a protocol
problem, stop and create a development-set experiment rather than tuning on
these held-out benchmark labels.

## 4. Estimate time and cost

```bash
python scripts/estimate_full_run.py \
  --run-dir results/baseline/<pilot-run-directory> \
  --hourly-rate <provider-price-per-hour>
```

The estimate uses measured concurrent wall-clock time, scales it to all 5456
records, adds a 25% safety factor, and recommends a booking duration. Provider
pricing is deliberately supplied at run time rather than hard-coded.

## 5. Resume the same run over the full benchmark

Keep the same server, GPU instance, runtime manifest, Git commit, and config:

```bash
./scripts/resume_full_baseline.sh \
  results/baseline/<pilot-run-directory>
```

The evaluator verifies configuration, input, and runtime-manifest hashes before
reusing the 200 pilot predictions. It appends the remaining records and reports
the complete MedQA and MedMCQA metrics separately.

If the instance or container image changes, start a new run instead of mixing
two runtime manifests.

## 6. Preserve results before terminating the instance

Copy the following off the rental instance:

```text
results/baseline/<run>/predictions.jsonl
results/baseline/<run>/metrics.json
results/baseline/<run>/metadata.json
results/runtime/runtime_manifest.json
results/vllm-server.log
data/benchmark/manifest.json
data/benchmark/provenance.json
```

Record the provider, GPU hourly rate, invoice/runtime, and any OOM or retry
events in the experiment report. Then stop the server:

```bash
./scripts/stop_vllm.sh
```

Do not write the baseline numbers into the résumé until these artifacts have
been reviewed and committed to an experiment report.

## Primary references

- [Qwen2.5-7B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [vLLM official Docker deployment](https://docs.vllm.ai/en/latest/deployment/docker/)
- [vLLM serve arguments](https://docs.vllm.ai/en/latest/cli/serve/)
