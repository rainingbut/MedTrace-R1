#!/usr/bin/env bash
set -Eeuo pipefail

if docker container inspect medtrace-vllm >/dev/null 2>&1; then
  docker stop medtrace-vllm
else
  echo "medtrace-vllm is not running"
fi
