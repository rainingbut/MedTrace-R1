#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=../configs/runtime/baseline_vllm.sh
source "${REPO_ROOT}/configs/runtime/baseline_vllm.sh"
cd -- "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
PILOT_PER_BENCHMARK="${PILOT_PER_BENCHMARK:-100}"
RUNTIME_BACKEND="$(resolve_vllm_runtime_backend)"

"${PYTHON_BIN}" data_pipeline/prepare_benchmarks.py
"${PYTHON_BIN}" scripts/wait_for_server.py \
  --base-url "http://127.0.0.1:${VLLM_PORT}/v1" \
  --model "${SERVED_MODEL_NAME}"
capture_args=(
  --backend "${RUNTIME_BACKEND}"
  --vllm-version "${VLLM_VERSION}"
  --model-id "${MODEL_ID}"
  --model-revision "${MODEL_REVISION}"
)
if [[ "${RUNTIME_BACKEND}" == docker ]]; then
  capture_args+=(--image "${VLLM_IMAGE}")
else
  capture_args+=(
    --pid-file results/runtime/vllm-server.pid
    --requirements-file requirements.txt
  )
fi
"${PYTHON_BIN}" scripts/capture_runtime.py "${capture_args[@]}"
"${PYTHON_BIN}" evaluation/run_eval.py \
  --config configs/eval/qwen2_5_7b_instruct.yaml \
  --limit-per-benchmark "${PILOT_PER_BENCHMARK}"
