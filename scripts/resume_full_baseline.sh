#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 results/baseline/<pilot-run-directory>" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=../configs/runtime/baseline_vllm.sh
source "${REPO_ROOT}/configs/runtime/baseline_vllm.sh"
cd -- "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_DIR="$1"

"${PYTHON_BIN}" scripts/wait_for_server.py \
  --base-url "http://127.0.0.1:${VLLM_PORT}/v1" \
  --model "${SERVED_MODEL_NAME}"
"${PYTHON_BIN}" evaluation/run_eval.py \
  --config configs/eval/qwen2_5_7b_instruct.yaml \
  --run-dir "${RUN_DIR}"
