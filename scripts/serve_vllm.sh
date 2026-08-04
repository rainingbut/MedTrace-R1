#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=../configs/runtime/baseline_vllm.sh
source "${REPO_ROOT}/configs/runtime/baseline_vllm.sh"

MEDTRACE_CACHE_DIR="${MEDTRACE_CACHE_DIR:-${REPO_ROOT}/.cache/huggingface}"
mkdir -p -- "${MEDTRACE_CACHE_DIR}"

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

echo "Starting ${MODEL_ID}@${MODEL_REVISION} with vLLM ${VLLM_VERSION}"
echo "Model cache: ${MEDTRACE_CACHE_DIR}"
exec docker "${docker_args[@]}" "${VLLM_IMAGE}" \
  --model "${MODEL_ID}" \
  --revision "${MODEL_REVISION}" \
  --tokenizer-revision "${MODEL_REVISION}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --dtype bfloat16 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --generation-config vllm \
  --seed "${VLLM_SEED}"
