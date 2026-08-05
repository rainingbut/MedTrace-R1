#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PID_FILE="${REPO_ROOT}/results/runtime/vllm-server.pid"

if [[ -f "${PID_FILE}" ]]; then
  read -r server_pid < "${PID_FILE}"
  if [[ "${server_pid}" =~ ^[0-9]+$ ]] && kill -0 "${server_pid}" 2>/dev/null; then
    server_command="$(ps -ww -p "${server_pid}" -o args=)"
    if [[ "${server_command}" != *"vllm serve"* ]]; then
      echo "Refusing to stop PID ${server_pid}: it is not a vLLM server" >&2
      exit 1
    fi
    kill "${server_pid}"
    echo "Stopped native vLLM process ${server_pid}"
  else
    echo "Native vLLM PID file is stale"
  fi
  rm -f -- "${PID_FILE}"
elif command -v docker >/dev/null 2>&1 && \
  docker container inspect medtrace-vllm >/dev/null 2>&1; then
  docker stop medtrace-vllm
else
  echo "medtrace-vllm is not running"
fi
