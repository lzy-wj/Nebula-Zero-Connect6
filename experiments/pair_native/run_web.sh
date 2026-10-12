#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${NEBULA_PAIR_WEB_PYTHON:-python}"
GPU="${NEBULA_PAIR_WEB_GPU:-0}"
CPUSET="${NEBULA_PAIR_WEB_CPUSET:-}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export NEBULA_PAIR_WEB_DEVICE="${NEBULA_PAIR_WEB_DEVICE:-cuda:0}"
export NEBULA_PAIR_WEB_HOST="${NEBULA_PAIR_WEB_HOST:-127.0.0.1}"
export NEBULA_PAIR_WEB_PORT="${NEBULA_PAIR_WEB_PORT:-18473}"
export NEBULA_CUDA_GRAPH="${NEBULA_CUDA_GRAPH:-0}"

cd "${REPO_ROOT}"
if command -v taskset >/dev/null 2>&1 && [[ -n "${CPUSET}" ]]; then
  exec taskset -c "${CPUSET}" "${PYTHON_BIN}" -m experiments.pair_native.web_app
fi
exec "${PYTHON_BIN}" -m experiments.pair_native.web_app
