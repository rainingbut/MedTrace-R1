#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=../configs/runtime/baseline_vllm.sh
source "${REPO_ROOT}/configs/runtime/baseline_vllm.sh"

RUNTIME_BACKEND="$(resolve_vllm_runtime_backend)"
if [[ -d /root/autodl-tmp ]]; then
  DEFAULT_CACHE_DIR="/root/autodl-tmp/medtrace-cache/huggingface"
else
  DEFAULT_CACHE_DIR="${REPO_ROOT}/.cache/huggingface"
fi
MEDTRACE_CACHE_DIR="${MEDTRACE_CACHE_DIR:-${DEFAULT_CACHE_DIR}}"
mkdir -p -- "${MEDTRACE_CACHE_DIR}"
mkdir -p -- "${REPO_ROOT}/results/runtime"

server_args=(
  --revision "${MODEL_REVISION}"
  --tokenizer-revision "${MODEL_REVISION}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --dtype bfloat16
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --generation-config vllm
  --seed "${VLLM_SEED}"
)

echo "Starting ${MODEL_ID}@${MODEL_REVISION} with vLLM ${VLLM_VERSION}"
echo "Runtime backend: ${RUNTIME_BACKEND}"
echo "Model cache: ${MEDTRACE_CACHE_DIR}"

if [[ "${RUNTIME_BACKEND}" == native ]]; then
  if ! command -v vllm >/dev/null 2>&1; then
    echo "vllm is not installed; install requirements.txt in the active environment" >&2
    exit 1
  fi
  export HF_HOME="${MEDTRACE_CACHE_DIR}"
  printf '%s\n' "$$" > "${REPO_ROOT}/results/runtime/vllm-server.pid"
  exec vllm serve "${MODEL_ID}" \
    --host 0.0.0.0 \
    --port "${VLLM_PORT}" \
    "${server_args[@]}"
fi

docker_args=(
  run --rm
  --name medtrace-vllm
  --gpus all
  --ipc=host
  --publish "${VLLM_PORT}:8000"
  --volume "${MEDTRACE_CACHE_DIR}:/root/.cache/huggingface"
)
if [[ -n "${HF_TOKEN:-}" ]]; then
  docker_args+=(--env HF_TOKEN)
fi

exec docker "${docker_args[@]}" "${VLLM_IMAGE}" \
  --model "${MODEL_ID}" \
  "${server_args[@]}"
