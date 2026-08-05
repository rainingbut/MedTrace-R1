# Pinned inference runtime for the MEDTRACE-R1 baseline.
VLLM_VERSION="0.24.0"
VLLM_IMAGE="vllm/vllm-openai@sha256:251eba5cc7c12fed0b75da22a9240e582b1c9e39f6fbc064f86781b963bd814f"
MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION="a09a35458c702b33eeacc393d103063234e8bc28"
SERVED_MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
VLLM_PORT="8000"
MAX_MODEL_LEN="4096"
MAX_NUM_SEQS="8"
GPU_MEMORY_UTILIZATION="0.90"
VLLM_SEED="42"

# `auto` uses Docker when a working daemon exists and otherwise runs the
# locally installed vLLM command. AutoDL container instances select `native`.
VLLM_RUNTIME_BACKEND="${VLLM_RUNTIME_BACKEND:-auto}"

resolve_vllm_runtime_backend() {
  case "${VLLM_RUNTIME_BACKEND}" in
    docker|native)
      printf '%s\n' "${VLLM_RUNTIME_BACKEND}"
      ;;
    auto)
      if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
        printf '%s\n' docker
      else
        printf '%s\n' native
      fi
      ;;
    *)
      echo "Unsupported VLLM_RUNTIME_BACKEND: ${VLLM_RUNTIME_BACKEND}" >&2
      return 2
      ;;
  esac
}
